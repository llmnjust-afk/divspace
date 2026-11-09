# Pooled versus within-prompt stability

Pooled: cosine between directions fitted on two halves of the whole sample set (all prompts mixed). Within: cosine between two halves of one prompt's own samples, averaged over prompts; brackets are 95% t-intervals over prompts.

| model | pooled (middle layer) | within-prompt mean | 95% CI | prompts | cos > 0.5 | t vs 0 | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_pythia-1b | +0.999 | -0.119 | [-0.327, +0.089] | 40 | 13/40 | -1.12 | 0.268 |
| EleutherAI_pythia-2.8b | +0.532 | +0.035 | [-0.029, +0.100] | 40 | 1/40 | +1.08 | 0.287 |
| EleutherAI_pythia-410m | +0.999 | -0.037 | [-0.222, +0.149] | 40 | 12/40 | -0.39 | 0.701 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | -0.635 | -0.021 | [-0.111, +0.069] | 40 | 3/40 | -0.46 | 0.651 |
| Qwen_Qwen2.5-1.5B-Instruct | -0.482 | +0.063 | [-0.014, +0.141] | 40 | 1/40 | +1.59 | 0.119 |
| Qwen_Qwen2.5-7B-Instruct | -0.758 | -0.029 | [-0.115, +0.056] | 40 | 1/40 | -0.67 | 0.507 |
| allenai_OLMo-2-1124-7B-Instruct | -0.534 | -0.068 | [-0.149, +0.014] | 40 | 1/40 | -1.63 | 0.111 |

A direction that could support steering needs a cosine close to 1: only then does a fit on one sample half predict what the other half measures.
