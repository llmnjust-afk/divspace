# Dose-response and specificity

## Within-model dose-response (Spearman rho between coefficient and dVendi)

| model | arm | n cells | rho | p |
| --- | --- | --- | --- | --- |
| EleutherAI_pythia-1b | real | 5 | -0.60 | 0.285 |
| EleutherAI_pythia-1b | random | 5 | +0.10 | 0.873 |
| EleutherAI_pythia-2.8b | real | 5 | -0.10 | 0.873 |
| EleutherAI_pythia-2.8b | random | 5 | -0.50 | 0.391 |
| EleutherAI_pythia-410m | real | 5 | -0.40 | 0.505 |
| EleutherAI_pythia-410m | random | 5 | -0.60 | 0.285 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | real | 5 | -0.30 | 0.624 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | random | 5 | +0.00 | 1.000 |
| Qwen_Qwen2.5-1.5B-Instruct | real | 5 | -0.30 | 0.624 |
| Qwen_Qwen2.5-1.5B-Instruct | random | 5 | -0.70 | 0.188 |
| Qwen_Qwen2.5-7B-Instruct | real | 5 | -0.30 | 0.624 |
| Qwen_Qwen2.5-7B-Instruct | random | 5 | -0.40 | 0.505 |
| allenai_OLMo-2-1124-7B-Instruct | real | 5 | -0.30 | 0.624 |
| allenai_OLMo-2-1124-7B-Instruct | random | 5 | +1.00 | 0.000 |

## Across-model summary

- real: median Spearman rho = -0.30 over 7 models, Wilcoxon p = 0.016
- random: median Spearman rho = -0.40 over 7 models, Wilcoxon p = 0.562
- real arm: 0/7 models have a positive (diversity-increasing) slope

## Fitted direction minus random direction at matched coefficients

| model | mean dVendi (fitted) - dVendi (random) | n coefs |
| --- | --- | --- |
| EleutherAI_pythia-1b | -0.243 | 5 |
| EleutherAI_pythia-2.8b | +0.013 | 5 |
| EleutherAI_pythia-410m | +0.261 | 5 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | -0.072 | 5 |
| Qwen_Qwen2.5-1.5B-Instruct | -0.060 | 5 |
| Qwen_Qwen2.5-7B-Instruct | -0.122 | 5 |
| allenai_OLMo-2-1124-7B-Instruct | -0.348 | 5 |

Across 7 models the fitted direction is -0.082 Vendi relative to a random vector (Wilcoxon p = 0.297).
