# 内部审核：divspace 与 ACL 2027 投稿状态

> **内部文件。** 记录缺陷、失效结论与差距清单，供后续重跑和改稿使用。
> 提交匿名评审前请从评审快照中排除本文件，或改写为「设计取舍」说明。

- 审核日期：2026-11-09
- 审核对象：commit `cb05f23` 的代码与 `paper/` 草稿
- 修复提交：`8c7fcde`
- 测试状态：`71 passed`（`python3 -m pytest tests/ -q`），`pyflakes` 干净

---

## 0. 结论

| 层面 | 状态 |
| --- | --- |
| 代码 | 发现 3 个决定性缺陷、3 个次要缺陷，**已全部修复并加回归测试** |
| 论文 | **`paper/main.pdf` 中的核心结论目前不成立**，两条主线依赖坏代码的输出，必须重跑后重写 |
| 可投稿性 | 重跑约 1 天（有 GPU），改稿约 2 周，另有 4 项内容缺口 |
| 风险 | 重跑后若组内余弦显著为正，标题与主结论都要改；「不存在」叙事不再成立 |

一句话：**代码已修好，论文还没修好，且修好的代码尚未在真模型上跑过。**

---

## 1. 缺陷清单

### D1（决定性）within-prompt 分组按样本索引，而非按离散度类别

- 位置：`divspace/stability.py::_direction_from`
- 原实现：

  ```python
  keep  = sorted(set(highs) | set(lows))                 # 按样本索引排序
  split = len([i for i in keep if i in set(highs)])
  return unit(mean_diff(acts[:split], acts[split:]))     # 取「索引最小的 k 条」
  ```

- 症状：两类样本被索引顺序混在一起，正类均值里混入低离散度响应，反之亦然。估计出的方向等于**真实轴的随机符号版本**。
- 证据：在激活严格等于已知离散度的理想数据上跑 200 次随机索引排列，`cos` 均值 **−0.11**，且 `|cos| > 0.5` 出现 **200/200**——即方向幅度对，符号随机。两个独立半区各自随机取号，余弦必然 ≈ 0。
- 影响：Table 1 的 Within 列（−0.037…+0.063）、§5 全节、「pooled +0.999 vs within −0.037」这一核心对比、摘要与标题。
- 修复：`ordered = highs + lows; split = len(highs)`，与 `local.py::fit_one` 的正确写法一致。
- 测试：`test_within_prompt_direction_recovers_a_planted_axis`（20 次随机排列，要求 `min cos > 0.9`）。

### D2（决定性）shuffled 安慰剂与真实方向完全相同

- 位置：`divspace/intervene/steer.py::shuffled_direction`
- 原实现：把 `negative` 做 `randperm` 后取 `mean(0)`。**均值对置换不变**，所以置换后的负类均值与原来一模一样。
- 证据：`cos(shuffled, real) = 1.000000`。
- 影响：Table 2 的 "shuffled 1.46 vs fitted 1.35" 是同一条方向的两次运行；§7「contrast contributes essentially nothing」是同义反复；摘要与结论中相应句子失效。
- 附带发现：`1.35` 与 `1.46` 实为同一条件下的两次采样，可直接用作**运行间噪声估计 ≈ 0.1 VENDI**。
- 概念问题：差异均值估计器根本不使用「配对」，全文 "destroying the dispersion pairing" 的表述在方法上不成立。正确安慰剂应重抽 **类归属**（保持两类样本量不变）。
- 修复：打乱池化后的类标签归属；若仍按配对置换，方向在数学上不可能改变。
- 测试：`test_shuffled_control_keeps_the_norm_and_drops_the_contrast`。

### D3（决定性）centroid 臂未归一化

- 位置：`divspace/intervene/steer.py::control_directions`
- 原实现：`torch.cat([positive, negative]).mean(0)`，直接作为方向返回。
- 症状：其余三臂都是单位向量，centroid 是**均值激活**，范数与残差流同量级（Qwen2.5-7B 第 14 层残差范数约 **1488**）。在默认 `--strength absolute` 下，同一个系数对 centroid 意味着比其他臂大 2–3 个数量级的扰动。
- 影响：「四个臂是同等大小的扰动」这一论断不成立；centroid 臂实际是灾难性干预。
- 修复：`unit(mean)`；新增测试要求所有安慰剂臂范数为 1。

### D4（次要）对照臂只取前 32 对，真实方向用全部

- 位置：`control_directions` 中的 `artifact.positive_pairs[:32]`
- 影响：安慰剂与真实臂的样本量不一致，比较被混淆。
- 修复：使用全部存储的配对激活。

### D5（次要）缺少 `pca1` 对照

- 症状：消融只投影掉**单一方向**，正文却写作「removing the response subspace」。无法区分「去掉任何主方向都会崩」与「这条方向特殊」。
- 修复：新增 `pca1` 臂（响应激活的第一主方向），并纳入 `--control` 选项。
- 仍需：`pca1` 与 `centroid` 的**消融**结果尚未跑。

### D6（次要）附录两两余弦表不可复现

- 症状：`tab:consistency` 的数字在仓库里找不到任何生成脚本（`grep -r pairwise` 无结果），属会话内的临时计算。
- 修复：`stability-suite` 现在导出每 prompt 的方向向量，`analysis/pairwise_consistency.py` 生成该表。

### D7（披露问题，非 bug）注入位置覆盖 prompt token

- `select_positions` 默认 `"all"`，所有脚本都未传 `--positions`，因此干预同时作用于提示词与生成 token。
- 方向是在**响应**激活上拟合的，注入却在全部位置，正文未披露。

### D8（披露问题，非 bug）正对照的打分方式

- `analysis/yesno.py` 用正则匹配整段文本的首词，正文写作「measuring the yes-rate of the first token」。需统一表述（并说明 `other` 计入分母）。

---

## 2. 论文中已失效的断言

| 位置 | 现文 | 问题 | 处置 |
| --- | --- | --- | --- |
| `abstract.tex` | 「split-half cosine … −0.037 when the two halves come from the same prompt … never exceeds \|0.12\|」 | D1 产物 | 重跑后替换 |
| `abstract.tex` | 「destroying the pairing that defines the contrast does the same」 | D2 产物 | 重跑后替换 |
| 标题 | "Pooled Contrast Directions Are Sums of Orthogonal Prompt-Local Vectors" | 正交性从未被证明；若组内拟合是噪声，两两余弦≈0 是必然结果，不能作为「正交分量」的证据 | 建议改为身份方向框架 |
| `stability.tex` 表 1 | Within / CI / p 三列 | D1 产物 | 重跑后替换 |
| `stability.tex` §5 正文 | 「the only difference is whether the two halves may contain responses to different prompts」 | **事实错误**：pooled 每臂≈128 条，within 每臂 3 条，差 40 倍预算 | 补预算曲线后改写 |
| `mechanism.tex` 表 2 与 §7 | shuffled＝1.46 作为「对照同样崩溃」 | D2 产物 | 用新 shuffled + pca1 重写 |
| `interventions.tex` | 「the four arms behave as four arbitrary perturbations of similar size」 | D3 产物 | 重跑 centroid |
| `interventions.tex` | 「a sign test rejects symmetry at p < 10⁻⁵」 | 35 个 cell 嵌套于 7 个模型，非独立 | 改为描述性统计 |
| `interventions.tex` | 摘要以 `ρ=−0.30, p=0.016` 领衔 | 每模型仅 5 个系数点，单模型 ρ=−0.30 时 p≈0.62 | 降级为辅助证据 |
| `setup.tex` | 「seven models … 35 dose levels」 | 正对照只在 3/7 模型真正通过 | 按模型分别陈述 |
| `mechanism.tex:19` | `Orth.\\ \\` | 表格多一行空行 | 修排版 |
| `mechanism.tex` | pythia-2.8b「+0.191（[−0.307,−0.074]）」 | 正负号自相矛盾 | 重写区间表述 |

---

## 3. 待重跑实验

| # | 实验 | 命令 | 估计 | 决定什么 |
| --- | --- | --- | --- | --- |
| R1 | 预算匹配稳定性（7 模型） | `bash scripts/16_stability_suite.sh` | 单卡 3–4 h | §5 全部数字；「欠功效」质疑是否成立 |
| R2 | 消融对照（shuffled / pca1 / centroid） | 待写 `scripts/17_ablation_controls.sh` | 单卡 1–2 h | §7 机制论断能否保留 |
| R3 | 第二个 seed 的主表单元 | 复用 `steer`，`--seed 1` | 1 h | 误差棒；目前只有 ±0.1 的间接估计 |
| R4 | 公开 prompt benchmark 复跑关键单元 | 待定 | 3–5 h | 审稿意见「自拟 prompt」 |

R1 会一次性产出：修好的组内余弦、`m = 1…40` 的预算曲线、between-prompt 与 pooled 的余弦、每 prompt 方向向量。

---

## 4. ACL 2027 差距清单

### A. 实验（必须）

1. R1 与 R2 重跑，替换 Table 1、Table 2 及全部相关正文。
2. 预算曲线必须进正文（一张小图或一列数字），否则「组内为零」无法与「样本太少」区分。
3. between-prompt 余弦必须进正文：它是身份解释的**直接**证据，比推理链更稳。
4. 消融需要至少三个对照臂（random / pca1 / centroid），并说明投影的是单方向不是子空间。

### B. 写作（必须）

1. 重写 §5 与 §7；摘要、结论、标题同步。
2. 统计表述降级：删去 `p=0.016` 作为亮点，`p<10⁻⁵` 改为描述性。
3. 披露：注入位置（全部 token）、正对照打分方式、单 seed、n=8 的 VENDI 偏差。
4. 修排版（`Orth.\\ \\`）与 pythia-2.8b 的区间表述。

### C. 内容缺口（审稿人会追）

1. **prompt 来源**：现在是 5 组自拟共 40 条（OPEN 10 / STORY 8 / EXPLAIN 8 / ADVICE 8 / LIST 6），`prompts.py` 自己的 docstring 就写着「投稿前换成公开 benchmark」。
2. **VENDI 敏感性**：只用一种句向量模型；建议第二嵌入模型或改用 token 级指标交叉验证。
3. **先例引用**：正文只说「很自然可以问」，没有一篇真正提出「diversity direction」的文献；容易被判为稻草人。要么补文献，要么明确写成「对通用做法的外推检验」。
4. **规模**：最大 7B、单一温度 0.9；需在 Limitations 里说明系数标定依赖残差范数，跨规模需重标。

### D. 合规

1. 匿名代码/数据链接（anonymous.4open.science）+ 说明如何复现每一张表。
2. ACL reproducibility checklist。
3. 确认 `docs/AUDIT.md` 与实验分支不进入匿名快照。

---

## 5. 时间线（今天 2026-11-09，ARR 截稿 2027-01-04）

| 周 | 任务 |
| --- | --- |
| W1（11-09 → 11-16） | 装环境、R1+R2 重跑、判定结论方向 |
| W2（11-17 → 11-23） | 重写 §5/§7/摘要/标题，重算全部表 |
| W3（11-24 → 11-30） | 公开 benchmark + 嵌入敏感性（R4） |
| W4（12-01 → 12-14） | 全文改写与内审，补先例引用 |
| W5（12-15 → 12-28） | 匿名化、复现包、checklist、外部试读 |
| W6（12-29 → 01-04） | 定稿提交，留 3 天缓冲 |

---

## 6. 复现与验证

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest tests/ -q          # 71 passed at 8c7fcde

# R1：预算匹配稳定性 + 两张表
MODELS="EleutherAI/pythia-1b Qwen/Qwen2.5-7B-Instruct" GPUS=2 \
  bash scripts/16_stability_suite.sh
# 产出 results/STABILITY_SUITE.md 与 results/PAIRWISE.md
```

修复的验证方式不是「测试通过」，而是三条可独立复算的事实：

1. 植入已知轴后，修好的 `_direction_from` 在 20 次随机排列下 `min cos > 0.9`（旧代码给出随机符号）。
2. `cos(shuffled, real)` 现在随 seed 变化且无信息标签下 `|cos| < 0.6`（旧代码恒为 1.0）。
3. 所有安慰剂臂范数为 1（旧 centroid 范数为均值激活范数）。

---

## 7. 变更记录

| commit | 内容 |
| --- | --- |
| `8c7fcde` | 修复 D1/D2/D3/D4，新增 `pca1`、`divspace/budget.py`、`stability-suite`、两个分析脚本、`scripts/16_stability_suite.sh`、6 个回归测试；README 同步 |
| `cb05f23` | 修正 CAA 引用（Rimsky et al. 2024）与标题字段 |
| `09ce424` | 引用元数据校正 |
