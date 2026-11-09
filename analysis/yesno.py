"""Answer-form control: how often each steering cell answers "Yes"."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

YES = re.compile(r"^\W*(yes|yeah|yep|sure|correct|true)\b", re.IGNORECASE)
NO = re.compile(r"^\W*(no|nope|nah|false|incorrect)\b", re.IGNORECASE)


def classify(text: str) -> str:
    text = text.strip()
    if YES.match(text):
        return "yes"
    if NO.match(text):
        return "no"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="jsonl produced by `steer`")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"yes": 0, "no": 0, "other": 0})
    for line in Path(args.results).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        counts[row.get("cell", "unknown")][classify(row.get("text", ""))] += 1

    rows = []
    print(f"{'cell':28s} {'n':>4s} {'yes_rate':>9s} {'no_rate':>8s}")
    for cell in sorted(counts):
        entry = counts[cell]
        total = sum(entry.values())
        yes = entry["yes"] / total if total else 0.0
        no = entry["no"] / total if total else 0.0
        rows.append({"cell": cell, "n": total, "yes_rate": yes, "no_rate": no,
                     "other_rate": entry["other"] / total if total else 0.0})
        print(f"{cell:28s} {total:4d} {yes:9.3f} {no:8.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
