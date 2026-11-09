# divspace experiment summary

## Diversity at the best intervention per file

| model | condition | cell | vendi | pc1_ratio | participation_ratio | distinct_2 | self_bleu | completion_rate | mean_words |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_pythia-160m | ablate | steer_L9_c+0.00 | 6.270 | 0.357 | 4.834 | 0.894 | 0.001 | 0.212 | 118.734 |
| EleutherAI_pythia-160m | control_orthogonal | orthogonal_L3_c+2.00 | 5.924 | 0.393 | 4.546 | 0.904 | 0.000 | 0.228 | 116.559 |
| EleutherAI_pythia-160m | control_random | random_L3_c-2.00 | 5.862 | 0.408 | 4.535 | 0.903 | 0.000 | 0.194 | 117.947 |
| EleutherAI_pythia-160m | control_shuffled | shuffled_L3_c+2.00 | 5.878 | 0.386 | 4.696 | 0.897 | 0.000 | 0.166 | 117.328 |
| EleutherAI_pythia-160m | sweep | baseline | 5.937 | 0.380 | 4.742 | 0.898 | 0.001 | 0.166 | 118.088 |
| EleutherAI_pythia-160m | sweep | steer_L3_c+1.00 | 6.170 | 0.368 | 4.739 | 0.900 | 0.001 | 0.263 | 116.953 |
| EleutherAI_pythia-1b | ablate | steer_L12_c+0.00 | 5.069 | 0.562 | 3.069 | 0.510 | 0.013 | 0.169 | 117.947 |
| EleutherAI_pythia-1b | control_orthogonal | orthogonal_L4_c+2.00 | 6.202 | 0.429 | 4.116 | 0.730 | 0.009 | 0.150 | 121.681 |
| EleutherAI_pythia-1b | control_random | random_L12_c+2.00 | 6.111 | 0.404 | 4.407 | 0.726 | 0.008 | 0.163 | 120.266 |
| EleutherAI_pythia-1b | control_shuffled | shuffled_L12_c-2.00 | 6.137 | 0.409 | 4.256 | 0.726 | 0.009 | 0.134 | 120.597 |
| EleutherAI_pythia-1b | sweep | baseline | 6.126 | 0.419 | 4.290 | 0.731 | 0.008 | 0.128 | 120.747 |
| EleutherAI_pythia-1b | sweep | steer_L4_c-3.00 | 6.222 | 0.394 | 4.429 | 0.740 | 0.009 | 0.163 | 121.331 |
| EleutherAI_pythia-2.8b | ablate | steer_L24_c+0.00 | 6.311 | 0.392 | 4.486 | 0.760 | 0.004 | 0.175 | 110.403 |
| EleutherAI_pythia-2.8b | control_orthogonal | orthogonal_L8_c+2.00 | 6.367 | 0.372 | 4.685 | 0.747 | 0.008 | 0.150 | 122.237 |
| EleutherAI_pythia-2.8b | control_random | random_L24_c+2.00 | 6.230 | 0.373 | 4.731 | 0.740 | 0.009 | 0.153 | 121.653 |
| EleutherAI_pythia-2.8b | control_shuffled | shuffled_L8_c-2.00 | 6.356 | 0.336 | 4.967 | 0.755 | 0.010 | 0.153 | 121.316 |
| EleutherAI_pythia-2.8b | sweep | baseline | 6.231 | 0.365 | 4.806 | 0.745 | 0.009 | 0.178 | 120.822 |
| EleutherAI_pythia-2.8b | sweep | steer_L8_c-3.00 | 6.294 | 0.367 | 4.812 | 0.758 | 0.006 | 0.194 | 117.881 |
| EleutherAI_pythia-410m | ablate | steer_L18_c+0.00 | 5.809 | 0.476 | 3.752 | 0.711 | 0.005 | 0.163 | 119.188 |
| EleutherAI_pythia-410m | control_orthogonal | orthogonal_L12_c+2.00 | 6.138 | 0.417 | 4.214 | 0.747 | 0.007 | 0.163 | 120.109 |
| EleutherAI_pythia-410m | control_random | random_L6_c-2.00 | 6.119 | 0.378 | 4.547 | 0.744 | 0.007 | 0.150 | 120.669 |
| EleutherAI_pythia-410m | control_shuffled | shuffled_L12_c-2.00 | 6.213 | 0.389 | 4.565 | 0.772 | 0.006 | 0.134 | 123.062 |
| EleutherAI_pythia-410m | sweep | baseline | 5.931 | 0.449 | 4.015 | 0.751 | 0.006 | 0.184 | 119.503 |
| EleutherAI_pythia-410m | sweep | steer_L12_c-3.00 | 6.139 | 0.419 | 4.204 | 0.761 | 0.007 | 0.172 | 118.884 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | ablate | steer_L18_c+0.00 | 6.308 | 0.307 | 5.275 | 0.722 | 0.068 | 0.209 | 121.541 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | control_orthogonal | orthogonal_L6_c-2.00 | 6.254 | 0.322 | 5.149 | 0.681 | 0.101 | 0.181 | 122.275 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | control_random | random_L12_c+2.00 | 6.255 | 0.317 | 5.187 | 0.676 | 0.103 | 0.175 | 121.572 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | control_shuffled | shuffled_L6_c+2.00 | 6.216 | 0.324 | 5.081 | 0.681 | 0.101 | 0.203 | 122.122 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | sweep | baseline | 6.160 | 0.333 | 5.011 | 0.682 | 0.103 | 0.184 | 122.312 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | sweep | steer_L12_c-1.00 | 6.296 | 0.315 | 5.233 | 0.679 | 0.103 | 0.188 | 121.556 |
| Qwen_Qwen2.5-1.5B-Instruct | ablate | steer_L21_c+0.00 | 6.193 | 0.338 | 4.929 | 0.808 | 0.052 | 0.175 | 127.600 |
| Qwen_Qwen2.5-1.5B-Instruct | control_orthogonal | orthogonal_L21_c+2.00 | 6.124 | 0.354 | 4.774 | 0.770 | 0.070 | 0.166 | 128.141 |
| Qwen_Qwen2.5-1.5B-Instruct | control_random | random_L14_c-2.00 | 6.107 | 0.350 | 4.809 | 0.771 | 0.069 | 0.181 | 126.281 |
| Qwen_Qwen2.5-1.5B-Instruct | control_shuffled | shuffled_L7_c-2.00 | 6.129 | 0.348 | 4.861 | 0.762 | 0.076 | 0.172 | 128.009 |
| Qwen_Qwen2.5-1.5B-Instruct | sweep | baseline | 6.100 | 0.349 | 4.797 | 0.763 | 0.076 | 0.150 | 127.669 |
| Qwen_Qwen2.5-1.5B-Instruct | sweep | steer_L7_c-3.00 | 6.196 | 0.335 | 4.948 | 0.762 | 0.076 | 0.181 | 128.559 |
| Qwen_Qwen2.5-7B-Instruct | ablate | steer_L21_c+0.00 | 6.157 | 0.327 | 5.035 | 0.665 | 0.141 | 0.062 | 128.072 |
| Qwen_Qwen2.5-7B-Instruct | control_orthogonal | orthogonal_L14_c-2.00 | 6.102 | 0.333 | 4.925 | 0.648 | 0.166 | 0.081 | 126.641 |
| Qwen_Qwen2.5-7B-Instruct | control_random | random_L14_c+2.00 | 6.100 | 0.336 | 4.921 | 0.654 | 0.159 | 0.053 | 126.419 |
| Qwen_Qwen2.5-7B-Instruct | control_shuffled | shuffled_L21_c-2.00 | 6.131 | 0.331 | 4.983 | 0.642 | 0.170 | 0.066 | 126.503 |
| Qwen_Qwen2.5-7B-Instruct | sweep | baseline | 6.055 | 0.333 | 4.911 | 0.643 | 0.169 | 0.113 | 126.247 |
| Qwen_Qwen2.5-7B-Instruct | sweep | steer_L7_c+3.00 | 6.144 | 0.329 | 4.947 | 0.651 | 0.162 | 0.053 | 127.156 |
| allenai_OLMo-2-1124-7B-Instruct | ablate | steer_L24_c+0.00 | 6.133 | 0.353 | 4.779 | 0.693 | 0.113 | 0.113 | 125.162 |
| allenai_OLMo-2-1124-7B-Instruct | control_orthogonal | orthogonal_L8_c-2.00 | 6.160 | 0.342 | 4.882 | 0.672 | 0.127 | 0.122 | 126.147 |
| allenai_OLMo-2-1124-7B-Instruct | control_random | random_L8_c+2.00 | 6.141 | 0.338 | 4.949 | 0.698 | 0.112 | 0.087 | 124.597 |
| allenai_OLMo-2-1124-7B-Instruct | control_shuffled | shuffled_L8_c+2.00 | 6.225 | 0.324 | 5.030 | 0.692 | 0.112 | 0.097 | 124.931 |
| allenai_OLMo-2-1124-7B-Instruct | sweep | baseline | 6.140 | 0.336 | 4.922 | 0.693 | 0.113 | 0.103 | 125.722 |
| allenai_OLMo-2-1124-7B-Instruct | sweep | steer_L16_c-2.00 | 6.202 | 0.324 | 5.056 | 0.671 | 0.124 | 0.106 | 126.416 |

## Attribution

| model | top layers (sign, magnitude) |
| --- | --- |
| EleutherAI_pythia-160m | L3 (+0.375), L6 (-0.232), L9 (-0.031) |
| EleutherAI_pythia-1b | L8 (-0.077), L12 (-0.076), L4 (-0.045) |
| EleutherAI_pythia-2.8b | L16 (-0.136), L24 (-0.132), L8 (-0.046) |
| EleutherAI_pythia-410m | L12 (-0.124), L18 (-0.120), L6 (+0.042) |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | L18 (-0.043), L12 (-0.004), L6 (+0.001) |
| Qwen_Qwen2.5-1.5B-Instruct | L21 (-0.075), L7 (-0.003), L14 (+0.001) |
| Qwen_Qwen2.5-7B-Instruct | L21 (-0.020), L7 (+0.009), L14 (+0.004) |
| allenai_OLMo-2-1124-7B-Instruct | L16 (+0.009), L24 (-0.002), L8 (+0.001) |

## Direction stability

| model | layer | split-half cosine |
| --- | --- | --- |
| EleutherAI_pythia-160m | 3 | 0.970 |
| EleutherAI_pythia-160m | 6 | 0.937 |
| EleutherAI_pythia-160m | 9 | 0.738 |
| EleutherAI_pythia-1b | 12 | 0.998 |
| EleutherAI_pythia-1b | 4 | 1.000 |
| EleutherAI_pythia-1b | 8 | 0.999 |
| EleutherAI_pythia-2.8b | 16 | 0.532 |
| EleutherAI_pythia-2.8b | 24 | 0.377 |
| EleutherAI_pythia-2.8b | 8 | 0.914 |
| EleutherAI_pythia-410m | 12 | 0.999 |
| EleutherAI_pythia-410m | 18 | 0.995 |
| EleutherAI_pythia-410m | 6 | 0.999 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | 12 | -0.635 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | 18 | -0.714 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | 6 | -0.589 |
| Qwen_Qwen2.5-1.5B-Instruct | 14 | -0.482 |
| Qwen_Qwen2.5-1.5B-Instruct | 21 | -0.620 |
| Qwen_Qwen2.5-1.5B-Instruct | 7 | -0.466 |
| Qwen_Qwen2.5-7B-Instruct | 14 | -0.758 |
| Qwen_Qwen2.5-7B-Instruct | 21 | -0.780 |
| Qwen_Qwen2.5-7B-Instruct | 7 | -0.708 |
| allenai_OLMo-2-1124-7B-Instruct | 16 | -0.534 |
| allenai_OLMo-2-1124-7B-Instruct | 24 | -0.614 |
| allenai_OLMo-2-1124-7B-Instruct | 8 | -0.511 |

