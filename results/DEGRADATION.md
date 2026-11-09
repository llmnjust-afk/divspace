# Diversity change versus text degradation

For every cell we report the Vendi change beside the relative change in mean length and in trigram repetition. Cells that move length or repetition by more than 10% are classed as degraded; the last two columns re-test the Vendi change on the surviving cells only.

| model | arm | coef | dVendi | z | dLen | dRep3 | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_pythia-1b | real | 1.0 | -0.010 | -0.05 | +2% | +8% | clean |
| EleutherAI_pythia-1b | real | 2.0 | -0.329 | -0.96 | +1% | +4% | clean |
| EleutherAI_pythia-1b | real | 4.0 | -0.191 | -0.83 | -0% | +47% | degraded |
| EleutherAI_pythia-1b | real | 8.0 | -0.152 | -0.65 | -1% | +53% | degraded |
| EleutherAI_pythia-1b | real | 16.0 | -0.336 | -1.25 | -2% | +119% | degraded |
| EleutherAI_pythia-1b | random | 1.0 | +0.098 | +0.47 | -1% | +20% | degraded |
| EleutherAI_pythia-1b | random | 2.0 | +0.030 | +0.15 | +0% | +2% | clean |
| EleutherAI_pythia-1b | random | 4.0 | -0.070 | -0.32 | -1% | +5% | clean |
| EleutherAI_pythia-1b | random | 8.0 | -0.021 | -0.09 | +1% | -8% | clean |
| EleutherAI_pythia-1b | random | 16.0 | +0.160 | +0.80 | -2% | -47% | degraded |
| EleutherAI_pythia-2.8b | real | 1.0 | -0.051 | -0.32 | -1% | +8% | clean |
| EleutherAI_pythia-2.8b | real | 2.0 | -0.010 | -0.08 | -4% | +21% | degraded |
| EleutherAI_pythia-2.8b | real | 4.0 | -0.039 | -0.36 | -2% | +12% | degraded |
| EleutherAI_pythia-2.8b | real | 8.0 | +0.003 | +0.02 | -5% | +41% | degraded |
| EleutherAI_pythia-2.8b | real | 16.0 | -0.081 | -0.67 | -7% | +69% | degraded |
| EleutherAI_pythia-2.8b | random | 1.0 | +0.013 | +0.11 | -1% | +30% | degraded |
| EleutherAI_pythia-2.8b | random | 2.0 | -0.122 | -0.79 | -2% | +15% | degraded |
| EleutherAI_pythia-2.8b | random | 4.0 | +0.032 | +0.25 | -1% | +27% | degraded |
| EleutherAI_pythia-2.8b | random | 8.0 | -0.032 | -0.16 | +0% | +24% | degraded |
| EleutherAI_pythia-2.8b | random | 16.0 | -0.132 | -1.12 | -1% | +13% | degraded |
| EleutherAI_pythia-410m | real | 1.0 | -0.203 | -1.11 | -1% | -4% | clean |
| EleutherAI_pythia-410m | real | 2.0 | -0.282 | -1.02 | -3% | +21% | degraded |
| EleutherAI_pythia-410m | real | 4.0 | -0.315 | -1.64 | -1% | +26% | degraded |
| EleutherAI_pythia-410m | real | 8.0 | -0.031 | -0.27 | -4% | +10% | degraded |
| EleutherAI_pythia-410m | real | 16.0 | -0.430 | -1.73 | -13% | +23% | degraded |
| EleutherAI_pythia-410m | random | 1.0 | -0.448 | -2.02 | -1% | +25% | degraded |
| EleutherAI_pythia-410m | random | 2.0 | -0.249 | -0.89 | -3% | +2% | clean |
| EleutherAI_pythia-410m | random | 4.0 | -0.012 | -0.09 | -3% | +27% | degraded |
| EleutherAI_pythia-410m | random | 8.0 | -0.760 | -2.73 | -9% | +47% | degraded |
| EleutherAI_pythia-410m | random | 16.0 | -1.097 | -3.44 | -27% | +133% | degraded |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | real | 1.0 | -0.155 | -0.79 | -2% | -1% | clean |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | real | 2.0 | -0.035 | -0.23 | -1% | -5% | clean |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | real | 4.0 | -0.276 | -1.90 | -2% | -5% | clean |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | real | 8.0 | -0.147 | -1.06 | -2% | -2% | clean |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | real | 16.0 | -0.269 | -1.95 | -3% | -6% | clean |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | random | 1.0 | -0.072 | -0.50 | -1% | -2% | clean |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | random | 2.0 | -0.087 | -0.59 | -2% | -2% | clean |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | random | 4.0 | -0.091 | -0.47 | -0% | +2% | clean |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | random | 8.0 | -0.204 | -1.21 | -0% | -5% | clean |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | random | 16.0 | -0.067 | -0.57 | +0% | +5% | clean |
| Qwen_Qwen2.5-1.5B-Instruct | real | 1.0 | -0.031 | -0.17 | +0% | -0% | clean |
| Qwen_Qwen2.5-1.5B-Instruct | real | 2.0 | -0.134 | -0.82 | -1% | -9% | clean |
| Qwen_Qwen2.5-1.5B-Instruct | real | 4.0 | -0.332 | -1.51 | -7% | -2% | clean |
| Qwen_Qwen2.5-1.5B-Instruct | real | 8.0 | -0.199 | -1.04 | -8% | -10% | degraded |
| Qwen_Qwen2.5-1.5B-Instruct | real | 16.0 | -0.070 | -0.47 | -21% | -32% | degraded |
| Qwen_Qwen2.5-1.5B-Instruct | random | 1.0 | +0.034 | +0.26 | -0% | -2% | clean |
| Qwen_Qwen2.5-1.5B-Instruct | random | 2.0 | -0.173 | -0.96 | +2% | -0% | clean |
| Qwen_Qwen2.5-1.5B-Instruct | random | 4.0 | +0.016 | +0.10 | +3% | +2% | clean |
| Qwen_Qwen2.5-1.5B-Instruct | random | 8.0 | -0.156 | -0.71 | +3% | +6% | clean |
| Qwen_Qwen2.5-1.5B-Instruct | random | 16.0 | -0.188 | -1.06 | +5% | +5% | clean |
| Qwen_Qwen2.5-7B-Instruct | real | 1.0 | -0.199 | -1.74 | -0% | -3% | clean |
| Qwen_Qwen2.5-7B-Instruct | real | 2.0 | -0.066 | -0.48 | +0% | -3% | clean |
| Qwen_Qwen2.5-7B-Instruct | real | 4.0 | -0.035 | -0.30 | -0% | +2% | clean |
| Qwen_Qwen2.5-7B-Instruct | real | 8.0 | -0.109 | -0.71 | +0% | +7% | clean |
| Qwen_Qwen2.5-7B-Instruct | real | 16.0 | -0.294 | -1.54 | -1% | -3% | clean |
| Qwen_Qwen2.5-7B-Instruct | random | 1.0 | -0.006 | -0.05 | +1% | -5% | clean |
| Qwen_Qwen2.5-7B-Instruct | random | 2.0 | -0.034 | -0.28 | +0% | -3% | clean |
| Qwen_Qwen2.5-7B-Instruct | random | 4.0 | -0.039 | -0.30 | -0% | -8% | clean |
| Qwen_Qwen2.5-7B-Instruct | random | 8.0 | +0.028 | +0.19 | -1% | -7% | clean |
| Qwen_Qwen2.5-7B-Instruct | random | 16.0 | -0.042 | -0.35 | -1% | -16% | degraded |
| allenai_OLMo-2-1124-7B-Instruct | real | 1.0 | +0.051 | +0.22 | -2% | +6% | clean |
| allenai_OLMo-2-1124-7B-Instruct | real | 2.0 | -0.054 | -0.28 | -1% | -1% | clean |
| allenai_OLMo-2-1124-7B-Instruct | real | 4.0 | +0.043 | +0.21 | -2% | +12% | degraded |
| allenai_OLMo-2-1124-7B-Instruct | real | 8.0 | +0.079 | +0.40 | -4% | -20% | degraded |
| allenai_OLMo-2-1124-7B-Instruct | real | 16.0 | -1.043 | -3.26 | -41% | -27% | degraded |

## Vendi change on non-degraded cells only

| model | arm | clean cells | mean dVendi | max |z| |
| --- | --- | --- | --- | --- |
| EleutherAI_pythia-1b | random | 3 | -0.020 | 0.32 |
| EleutherAI_pythia-1b | real | 2 | -0.169 | 0.96 |
| EleutherAI_pythia-2.8b | real | 1 | -0.051 | 0.32 |
| EleutherAI_pythia-410m | random | 1 | -0.249 | 0.89 |
| EleutherAI_pythia-410m | real | 1 | -0.203 | 1.11 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | random | 5 | -0.104 | 1.21 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | real | 5 | -0.176 | 1.95 |
| Qwen_Qwen2.5-1.5B-Instruct | random | 5 | -0.093 | 1.06 |
| Qwen_Qwen2.5-1.5B-Instruct | real | 3 | -0.166 | 1.51 |
| Qwen_Qwen2.5-7B-Instruct | random | 4 | -0.013 | 0.30 |
| Qwen_Qwen2.5-7B-Instruct | real | 5 | -0.141 | 1.74 |
| allenai_OLMo-2-1124-7B-Instruct | real | 2 | -0.002 | 0.28 |
