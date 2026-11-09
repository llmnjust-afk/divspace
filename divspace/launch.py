"""Fill every GPU with an independent task.

Each task is a full ``divspace`` CLI invocation. Workers are pinned with
``CUDA_VISIBLE_DEVICES`` so that a task never sees more than one card, and the
scheduler keeps exactly one running task per GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .utils import write_json


@dataclass
class Task:
    index: int
    cmd: list[str]
    gpu: int | None = None
    status: str = "pending"
    started_at: float | None = None
    finished_at: float | None = None
    log: str | None = None
    field_extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "cmd": self.cmd,
            "gpu": self.gpu,
            "status": self.status,
            "seconds": (
                round(self.finished_at - self.started_at, 2)
                if self.started_at and self.finished_at
                else None
            ),
            "log": self.log,
            **self.field_extra,
        }


def load_tasks(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["tasks"] if isinstance(payload, dict) else payload


def plan(models: list[str], stages: list[dict], shards: int) -> dict:
    """Cross a model list with a list of per-model argument templates."""
    tasks = []
    for model in models:
        for stage in stages:
            for shard_id in range(shards):
                cmd = list(stage["cmd"]) + ["--model", model, "--device", "cuda:0"]
                cmd += ["--shard-id", str(shard_id), "--num-shards", str(shards)]
                cmd += stage.get("extra", [])
                out = stage["out"].format(model=model.replace("/", "__"), shard=shard_id)
                if "--out" not in cmd:
                    cmd += ["--out", out]
                tasks.append({"cmd": cmd, "out": out})
    return {"gpus": len(models) * shards, "tasks": tasks}


def run(tasks: list[dict], gpus: list[int], log_dir: Path, poll: float = 2.0) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = [Task(index=i, cmd=t["cmd"]) for i, t in enumerate(tasks)]
    pending = queue
    running: list[tuple[Task, subprocess.Popen]] = []
    free = list(gpus)
    failures = 0

    while pending or running:
        while pending and free:
            task = pending.pop(0)
            gpu = free.pop(0)
            task.gpu = gpu
            task.status = "running"
            task.started_at = time.time()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env.setdefault("PYTHONUNBUFFERED", "1")
            log_path = log_dir / f"task{task.index:03d}_gpu{gpu}.log"
            handle = log_path.open("w", encoding="utf-8")
            handle.write(json.dumps({"gpu": gpu, "cmd": task.cmd}) + "\n")
            handle.flush()
            process = subprocess.Popen(
                [sys.executable, "-m", "divspace.cli", *task.cmd],
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            process._log_handle = handle
            task.log = str(log_path)
            running.append((task, process))
            print(f"[gpu{gpu}] start task {task.index}: {' '.join(task.cmd[:3])} ...", flush=True)

        time.sleep(poll)

        still_running = []
        for task, process in running:
            code = process.poll()
            if code is None:
                still_running.append((task, process))
                continue
            process._log_handle.close()
            task.finished_at = time.time()
            task.status = "done" if code == 0 else "failed"
            if code != 0:
                failures += 1
                print(f"[gpu{task.gpu}] task {task.index} FAILED (exit {code}) -> {task.log}", flush=True)
            else:
                print(
                    f"[gpu{task.gpu}] task {task.index} done in "
                    f"{task.finished_at - task.started_at:.0f}s",
                    flush=True,
                )
            free.append(task.gpu)
        running = still_running

    write_json(log_dir / "manifest.json", {"tasks": [t.as_dict() for t in queue]})
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="divspace-launch", description=__doc__)
    parser.add_argument("--tasks", required=True, help="JSON file with a task list")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    if args.dry_run:
        for i, task in enumerate(tasks):
            print(f"{i:03d} {' '.join(task['cmd'])}")
        return 0
    failures = run(tasks, args.gpus, Path(args.log_dir))
    print(f"{len(tasks) - failures}/{len(tasks)} tasks succeeded")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
