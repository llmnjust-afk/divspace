# divspace experiment summary

## Diversity at the best intervention per file

| model | condition | cell | vendi | pc1_ratio | participation_ratio | distinct_2 | self_bleu | completion_rate | mean_words |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_pythia-160m | ablate | steer_L9_c+0.00 | 6.270 | 0.357 | 4.834 | 0.894 | 0.001 | 0.212 | 118.734 |
| EleutherAI_pythia-160m | ablate_orthogonal | ablate_orthogonal_L3_c+0.00 | 5.965 | 0.426 | 4.121 | 0.715 | 0.001 | 0.153 | 125.250 |
| EleutherAI_pythia-160m | ablate_random | ablate_random_L3_c+0.00 | 6.287 | 0.338 | 4.934 | 0.904 | 0.000 | 0.159 | 120.169 |
| EleutherAI_pythia-160m | ablate_shuffled | ablate_shuffled_L9_c+0.00 | 6.252 | 0.360 | 4.876 | 0.892 | 0.001 | 0.206 | 120.541 |
| EleutherAI_pythia-160m | control_orthogonal | orthogonal_L3_c+2.00 | 5.924 | 0.393 | 4.546 | 0.904 | 0.000 | 0.228 | 116.559 |
| EleutherAI_pythia-160m | control_random | random_L3_c-2.00 | 5.862 | 0.408 | 4.535 | 0.903 | 0.000 | 0.194 | 117.947 |
| EleutherAI_pythia-160m | control_shuffled | shuffled_L3_c+2.00 | 5.878 | 0.386 | 4.696 | 0.897 | 0.000 | 0.166 | 117.328 |
| EleutherAI_pythia-160m | lt_control_orthogonal | orthogonal_L3_c-2.00 | 5.947 | 0.381 | 4.569 | 0.766 | 0.004 | 0.184 | 117.828 |
| EleutherAI_pythia-160m | lt_control_random | random_L3_c+2.00 | 5.763 | 0.405 | 4.401 | 0.770 | 0.004 | 0.178 | 119.394 |
| EleutherAI_pythia-160m | lt_control_shuffled | shuffled_L6_c-2.00 | 5.697 | 0.451 | 4.007 | 0.771 | 0.005 | 0.163 | 119.091 |
| EleutherAI_pythia-160m | sweep | baseline | 5.937 | 0.380 | 4.742 | 0.898 | 0.001 | 0.166 | 118.088 |
| EleutherAI_pythia-160m | sweep | steer_L3_c+1.00 | 6.170 | 0.368 | 4.739 | 0.900 | 0.001 | 0.263 | 116.953 |
| EleutherAI_pythia-160m | sweep_lt | baseline | 6.019 | 0.373 | 4.587 | 0.778 | 0.004 | 0.150 | 120.947 |
| EleutherAI_pythia-160m | sweep_lt | steer_L3_c+1.00 | 5.952 | 0.395 | 4.416 | 0.768 | 0.005 | 0.178 | 120.584 |
| EleutherAI_pythia-1b | ablate | steer_L12_c+0.00 | 5.069 | 0.562 | 3.069 | 0.510 | 0.013 | 0.169 | 117.947 |
| EleutherAI_pythia-1b | ablate_orthogonal | ablate_orthogonal_L4_c+0.00 | 5.693 | 0.456 | 3.760 | 0.516 | 0.011 | 0.394 | 82.684 |
| EleutherAI_pythia-1b | ablate_random | ablate_random_L12_c+0.00 | 6.219 | 0.396 | 4.523 | 0.735 | 0.007 | 0.147 | 121.153 |
| EleutherAI_pythia-1b | ablate_shuffled | ablate_shuffled_L12_c+0.00 | 5.624 | 0.475 | 3.635 | 0.578 | 0.008 | 0.175 | 121.947 |
| EleutherAI_pythia-1b | control_orthogonal | orthogonal_L4_c+2.00 | 6.202 | 0.429 | 4.116 | 0.730 | 0.009 | 0.150 | 121.681 |
| EleutherAI_pythia-1b | control_random | random_L12_c+2.00 | 6.111 | 0.404 | 4.407 | 0.726 | 0.008 | 0.163 | 120.266 |
| EleutherAI_pythia-1b | control_shuffled | shuffled_L12_c-2.00 | 6.137 | 0.409 | 4.256 | 0.726 | 0.009 | 0.134 | 120.597 |
| EleutherAI_pythia-1b | local | baseline | 6.296 | 0.397 | 4.355 | 0.739 | 0.007 | 0.119 | 122.106 |
| EleutherAI_pythia-1b | local | localrandom_L8_c-2.00 | 6.319 | 0.409 | 4.185 | 0.734 | 0.006 | 0.156 | 122.381 |
| EleutherAI_pythia-1b | lt_control_orthogonal | orthogonal_L4_c-2.00 | 5.240 | 0.432 | 3.843 | 0.226 | 0.102 | 0.153 | 118.828 |
| EleutherAI_pythia-1b | lt_control_random | random_L8_c-2.00 | 5.284 | 0.446 | 3.755 | 0.241 | 0.108 | 0.138 | 118.150 |
| EleutherAI_pythia-1b | lt_control_shuffled | shuffled_L8_c-2.00 | 5.266 | 0.444 | 3.778 | 0.249 | 0.109 | 0.159 | 117.694 |
| EleutherAI_pythia-1b | sweep | baseline | 6.126 | 0.419 | 4.290 | 0.731 | 0.008 | 0.128 | 120.747 |
| EleutherAI_pythia-1b | sweep | steer_L4_c-3.00 | 6.222 | 0.394 | 4.429 | 0.740 | 0.009 | 0.163 | 121.331 |
| EleutherAI_pythia-1b | sweep_lt | baseline | 5.145 | 0.438 | 3.755 | 0.241 | 0.104 | 0.147 | 117.350 |
| EleutherAI_pythia-1b | sweep_lt | steer_L8_c-2.00 | 5.287 | 0.434 | 3.881 | 0.248 | 0.110 | 0.156 | 116.672 |
| EleutherAI_pythia-2.8b | ablate | steer_L24_c+0.00 | 6.311 | 0.392 | 4.486 | 0.760 | 0.004 | 0.175 | 110.403 |
| EleutherAI_pythia-2.8b | ablate_orthogonal | ablate_orthogonal_L16_c+0.00 | 5.987 | 0.451 | 3.896 | 0.755 | 0.003 | 0.131 | 110.150 |
| EleutherAI_pythia-2.8b | ablate_random | ablate_random_L8_c+0.00 | 6.343 | 0.342 | 5.000 | 0.750 | 0.009 | 0.181 | 120.425 |
| EleutherAI_pythia-2.8b | ablate_shuffled | ablate_shuffled_L24_c+0.00 | 6.602 | 0.319 | 5.292 | 0.772 | 0.004 | 0.153 | 127.675 |
| EleutherAI_pythia-2.8b | control_orthogonal | orthogonal_L8_c+2.00 | 6.367 | 0.372 | 4.685 | 0.747 | 0.008 | 0.150 | 122.237 |
| EleutherAI_pythia-2.8b | control_random | random_L24_c+2.00 | 6.230 | 0.373 | 4.731 | 0.740 | 0.009 | 0.153 | 121.653 |
| EleutherAI_pythia-2.8b | control_shuffled | shuffled_L8_c-2.00 | 6.356 | 0.336 | 4.967 | 0.755 | 0.010 | 0.153 | 121.316 |
| EleutherAI_pythia-2.8b | lt_control_orthogonal | orthogonal_L16_c-2.00 | 4.715 | 0.564 | 2.893 | 0.295 | 0.088 | 0.216 | 117.791 |
| EleutherAI_pythia-2.8b | lt_control_random | random_L24_c+2.00 | 4.715 | 0.542 | 3.073 | 0.273 | 0.103 | 0.166 | 117.694 |
| EleutherAI_pythia-2.8b | lt_control_shuffled | shuffled_L24_c+2.00 | 4.648 | 0.551 | 3.023 | 0.280 | 0.116 | 0.191 | 118.544 |
| EleutherAI_pythia-2.8b | sweep | baseline | 6.231 | 0.365 | 4.806 | 0.745 | 0.009 | 0.178 | 120.822 |
| EleutherAI_pythia-2.8b | sweep | steer_L8_c-3.00 | 6.294 | 0.367 | 4.812 | 0.758 | 0.006 | 0.194 | 117.881 |
| EleutherAI_pythia-2.8b | sweep_lt | baseline | 4.723 | 0.542 | 3.075 | 0.288 | 0.102 | 0.184 | 119.300 |
| EleutherAI_pythia-2.8b | sweep_lt | steer_L24_c+2.00 | 4.634 | 0.566 | 2.918 | 0.299 | 0.104 | 0.197 | 119.284 |
| EleutherAI_pythia-410m | ablate | steer_L18_c+0.00 | 5.809 | 0.476 | 3.752 | 0.711 | 0.005 | 0.163 | 119.188 |
| EleutherAI_pythia-410m | ablate_orthogonal | ablate_orthogonal_L18_c+0.00 | 5.398 | 0.507 | 3.359 | 0.633 | 0.005 | 0.181 | 103.725 |
| EleutherAI_pythia-410m | ablate_random | ablate_random_L12_c+0.00 | 6.092 | 0.405 | 4.330 | 0.750 | 0.007 | 0.150 | 118.859 |
| EleutherAI_pythia-410m | ablate_shuffled | ablate_shuffled_L6_c+0.00 | 5.262 | 0.518 | 3.446 | 0.298 | 0.005 | 0.097 | 72.678 |
| EleutherAI_pythia-410m | control_orthogonal | orthogonal_L12_c+2.00 | 6.138 | 0.417 | 4.214 | 0.747 | 0.007 | 0.163 | 120.109 |
| EleutherAI_pythia-410m | control_random | random_L6_c-2.00 | 6.119 | 0.378 | 4.547 | 0.744 | 0.007 | 0.150 | 120.669 |
| EleutherAI_pythia-410m | control_shuffled | shuffled_L12_c-2.00 | 6.213 | 0.389 | 4.565 | 0.772 | 0.006 | 0.134 | 123.062 |
| EleutherAI_pythia-410m | local | baseline | 6.243 | 0.390 | 4.504 | 0.756 | 0.006 | 0.141 | 121.819 |
| EleutherAI_pythia-410m | local | localother_L12_c-2.00 | 6.340 | 0.359 | 4.776 | 0.761 | 0.005 | 0.141 | 123.535 |
| EleutherAI_pythia-410m | lt_control_orthogonal | orthogonal_L18_c-2.00 | 5.422 | 0.433 | 3.862 | 0.224 | 0.041 | 0.231 | 115.484 |
| EleutherAI_pythia-410m | lt_control_random | random_L18_c+2.00 | 5.482 | 0.435 | 3.853 | 0.241 | 0.045 | 0.237 | 116.197 |
| EleutherAI_pythia-410m | lt_control_shuffled | shuffled_L18_c-2.00 | 5.390 | 0.439 | 3.812 | 0.220 | 0.053 | 0.188 | 120.259 |
| EleutherAI_pythia-410m | sweep | baseline | 5.931 | 0.449 | 4.015 | 0.751 | 0.006 | 0.184 | 119.503 |
| EleutherAI_pythia-410m | sweep | steer_L12_c-3.00 | 6.139 | 0.419 | 4.204 | 0.761 | 0.007 | 0.172 | 118.884 |
| EleutherAI_pythia-410m | sweep_lt | baseline | 5.389 | 0.446 | 3.723 | 0.218 | 0.054 | 0.225 | 116.744 |
| EleutherAI_pythia-410m | sweep_lt | steer_L18_c-1.00 | 5.495 | 0.415 | 3.948 | 0.229 | 0.052 | 0.231 | 116.925 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | ablate | steer_L18_c+0.00 | 6.308 | 0.307 | 5.275 | 0.722 | 0.068 | 0.209 | 121.541 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | ablate_orthogonal | ablate_orthogonal_L12_c+0.00 | 6.210 | 0.325 | 5.054 | 0.688 | 0.085 | 0.181 | 120.922 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | ablate_random | ablate_random_L6_c+0.00 | 6.235 | 0.319 | 5.165 | 0.678 | 0.105 | 0.169 | 122.750 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | ablate_shuffled | ablate_shuffled_L12_c+0.00 | 6.313 | 0.325 | 5.039 | 0.693 | 0.058 | 0.300 | 117.875 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | control_orthogonal | orthogonal_L6_c-2.00 | 6.254 | 0.322 | 5.149 | 0.681 | 0.101 | 0.181 | 122.275 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | control_random | random_L12_c+2.00 | 6.255 | 0.317 | 5.187 | 0.676 | 0.103 | 0.175 | 121.572 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | control_shuffled | shuffled_L6_c+2.00 | 6.216 | 0.324 | 5.081 | 0.681 | 0.101 | 0.203 | 122.122 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | lt_control_orthogonal | orthogonal_L12_c+2.00 | 5.828 | 0.362 | 4.549 | 0.514 | 0.237 | 0.175 | 121.994 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | lt_control_random | random_L6_c+2.00 | 5.856 | 0.361 | 4.569 | 0.518 | 0.237 | 0.209 | 121.278 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | lt_control_shuffled | shuffled_L6_c+2.00 | 5.831 | 0.378 | 4.462 | 0.523 | 0.231 | 0.188 | 121.812 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | sweep | baseline | 6.160 | 0.333 | 5.011 | 0.682 | 0.103 | 0.184 | 122.312 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | sweep | steer_L12_c-1.00 | 6.296 | 0.315 | 5.233 | 0.679 | 0.103 | 0.188 | 121.556 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | sweep_lt | baseline | 5.775 | 0.371 | 4.504 | 0.518 | 0.236 | 0.172 | 121.737 |
| HuggingFaceTB_SmolLM2-1.7B-Instruct | sweep_lt | steer_L18_c-3.00 | 5.931 | 0.350 | 4.701 | 0.516 | 0.240 | 0.216 | 122.159 |
| Qwen_Qwen2.5-1.5B-Instruct | ablate | steer_L21_c+0.00 | 6.193 | 0.338 | 4.929 | 0.808 | 0.052 | 0.175 | 127.600 |
| Qwen_Qwen2.5-1.5B-Instruct | ablate_orthogonal | ablate_orthogonal_L7_c+0.00 | 6.152 | 0.343 | 4.865 | 0.761 | 0.074 | 0.163 | 127.369 |
| Qwen_Qwen2.5-1.5B-Instruct | ablate_random | ablate_random_L21_c+0.00 | 6.155 | 0.336 | 4.938 | 0.762 | 0.075 | 0.153 | 128.734 |
| Qwen_Qwen2.5-1.5B-Instruct | ablate_shuffled | ablate_shuffled_L21_c+0.00 | 6.023 | 0.366 | 4.710 | 0.810 | 0.048 | 0.191 | 127.200 |
| Qwen_Qwen2.5-1.5B-Instruct | control_orthogonal | orthogonal_L21_c+2.00 | 6.124 | 0.354 | 4.774 | 0.770 | 0.070 | 0.166 | 128.141 |
| Qwen_Qwen2.5-1.5B-Instruct | control_random | random_L14_c-2.00 | 6.107 | 0.350 | 4.809 | 0.771 | 0.069 | 0.181 | 126.281 |
| Qwen_Qwen2.5-1.5B-Instruct | control_shuffled | shuffled_L7_c-2.00 | 6.129 | 0.348 | 4.861 | 0.762 | 0.076 | 0.172 | 128.009 |
| Qwen_Qwen2.5-1.5B-Instruct | local | baseline | 6.025 | 0.348 | 4.825 | 0.762 | 0.073 | 0.172 | 127.612 |
| Qwen_Qwen2.5-1.5B-Instruct | local | localrandom_L14_c+2.00 | 6.181 | 0.325 | 5.059 | 0.755 | 0.077 | 0.178 | 127.403 |
| Qwen_Qwen2.5-1.5B-Instruct | lt_control_orthogonal | orthogonal_L21_c+2.00 | 5.713 | 0.392 | 4.306 | 0.630 | 0.174 | 0.197 | 127.969 |
| Qwen_Qwen2.5-1.5B-Instruct | lt_control_random | random_L14_c+2.00 | 5.859 | 0.367 | 4.541 | 0.618 | 0.188 | 0.159 | 128.691 |
| Qwen_Qwen2.5-1.5B-Instruct | lt_control_shuffled | shuffled_L7_c+2.00 | 5.791 | 0.389 | 4.407 | 0.620 | 0.182 | 0.181 | 127.287 |
| Qwen_Qwen2.5-1.5B-Instruct | sweep | baseline | 6.100 | 0.349 | 4.797 | 0.763 | 0.076 | 0.150 | 127.669 |
| Qwen_Qwen2.5-1.5B-Instruct | sweep | steer_L7_c-3.00 | 6.196 | 0.335 | 4.948 | 0.762 | 0.076 | 0.181 | 128.559 |
| Qwen_Qwen2.5-1.5B-Instruct | sweep_lt | baseline | 5.747 | 0.387 | 4.365 | 0.625 | 0.184 | 0.188 | 127.003 |
| Qwen_Qwen2.5-1.5B-Instruct | sweep_lt | steer_L21_c+2.00 | 5.782 | 0.385 | 4.393 | 0.631 | 0.172 | 0.175 | 127.016 |
| Qwen_Qwen2.5-7B-Instruct | ablate | steer_L21_c+0.00 | 6.157 | 0.327 | 5.035 | 0.665 | 0.141 | 0.062 | 128.072 |
| Qwen_Qwen2.5-7B-Instruct | ablate_orthogonal | ablate_orthogonal_L21_c+0.00 | 6.055 | 0.337 | 4.883 | 0.639 | 0.169 | 0.078 | 126.787 |
| Qwen_Qwen2.5-7B-Instruct | ablate_random | ablate_random_L14_c+0.00 | 6.129 | 0.328 | 4.988 | 0.645 | 0.167 | 0.075 | 126.297 |
| Qwen_Qwen2.5-7B-Instruct | ablate_shuffled | ablate_shuffled_L21_c+0.00 | 6.099 | 0.338 | 4.886 | 0.680 | 0.130 | 0.075 | 126.803 |
| Qwen_Qwen2.5-7B-Instruct | control_orthogonal | orthogonal_L14_c-2.00 | 6.102 | 0.333 | 4.925 | 0.648 | 0.166 | 0.081 | 126.641 |
| Qwen_Qwen2.5-7B-Instruct | control_random | random_L14_c+2.00 | 6.100 | 0.336 | 4.921 | 0.654 | 0.159 | 0.053 | 126.419 |
| Qwen_Qwen2.5-7B-Instruct | control_shuffled | shuffled_L21_c-2.00 | 6.131 | 0.331 | 4.983 | 0.642 | 0.170 | 0.066 | 126.503 |
| Qwen_Qwen2.5-7B-Instruct | local | baseline | 6.081 | 0.343 | 4.843 | 0.643 | 0.168 | 0.059 | 126.044 |
| Qwen_Qwen2.5-7B-Instruct | local | localrandom_L14_c-2.00 | 6.077 | 0.341 | 4.884 | 0.643 | 0.165 | 0.087 | 126.459 |
| Qwen_Qwen2.5-7B-Instruct | lt_control_orthogonal | orthogonal_L14_c-2.00 | 5.892 | 0.353 | 4.615 | 0.519 | 0.293 | 0.069 | 126.506 |
| Qwen_Qwen2.5-7B-Instruct | lt_control_random | random_L21_c+2.00 | 5.851 | 0.352 | 4.610 | 0.517 | 0.294 | 0.066 | 126.600 |
| Qwen_Qwen2.5-7B-Instruct | lt_control_shuffled | shuffled_L14_c-2.00 | 5.895 | 0.351 | 4.652 | 0.522 | 0.295 | 0.075 | 126.609 |
| Qwen_Qwen2.5-7B-Instruct | sweep | baseline | 6.055 | 0.333 | 4.911 | 0.643 | 0.169 | 0.113 | 126.247 |
| Qwen_Qwen2.5-7B-Instruct | sweep | steer_L7_c+3.00 | 6.144 | 0.329 | 4.947 | 0.651 | 0.162 | 0.053 | 127.156 |
| Qwen_Qwen2.5-7B-Instruct | sweep_lt | baseline | 5.806 | 0.377 | 4.383 | 0.519 | 0.292 | 0.072 | 126.684 |
| Qwen_Qwen2.5-7B-Instruct | sweep_lt | steer_L7_c+1.00 | 5.825 | 0.365 | 4.493 | 0.517 | 0.293 | 0.081 | 126.681 |
| allenai_OLMo-2-1124-7B-Instruct | ablate | steer_L24_c+0.00 | 6.133 | 0.353 | 4.779 | 0.693 | 0.113 | 0.113 | 125.162 |
| allenai_OLMo-2-1124-7B-Instruct | ablate_orthogonal | ablate_orthogonal_L24_c+0.00 | 6.166 | 0.330 | 4.972 | 0.694 | 0.111 | 0.087 | 125.541 |
| allenai_OLMo-2-1124-7B-Instruct | ablate_random | ablate_random_L8_c+0.00 | 6.206 | 0.330 | 4.984 | 0.693 | 0.111 | 0.103 | 125.356 |
| allenai_OLMo-2-1124-7B-Instruct | ablate_shuffled | ablate_shuffled_L16_c+0.00 | 6.193 | 0.329 | 4.999 | 0.695 | 0.111 | 0.094 | 124.953 |
| allenai_OLMo-2-1124-7B-Instruct | control_orthogonal | orthogonal_L8_c-2.00 | 6.160 | 0.342 | 4.882 | 0.672 | 0.127 | 0.122 | 126.147 |
| allenai_OLMo-2-1124-7B-Instruct | control_random | random_L8_c+2.00 | 6.141 | 0.338 | 4.949 | 0.698 | 0.112 | 0.087 | 124.597 |
| allenai_OLMo-2-1124-7B-Instruct | control_shuffled | shuffled_L8_c+2.00 | 6.225 | 0.324 | 5.030 | 0.692 | 0.112 | 0.097 | 124.931 |
| allenai_OLMo-2-1124-7B-Instruct | lt_control_orthogonal | orthogonal_L8_c-2.00 | 5.842 | 0.369 | 4.509 | 0.539 | 0.248 | 0.097 | 127.144 |
| allenai_OLMo-2-1124-7B-Instruct | lt_control_random | random_L24_c+2.00 | 5.800 | 0.386 | 4.385 | 0.553 | 0.237 | 0.103 | 126.463 |
| allenai_OLMo-2-1124-7B-Instruct | lt_control_shuffled | shuffled_L8_c-2.00 | 5.865 | 0.360 | 4.589 | 0.545 | 0.246 | 0.084 | 127.506 |
| allenai_OLMo-2-1124-7B-Instruct | sweep | baseline | 6.140 | 0.336 | 4.922 | 0.693 | 0.113 | 0.103 | 125.722 |
| allenai_OLMo-2-1124-7B-Instruct | sweep | steer_L16_c-2.00 | 6.202 | 0.324 | 5.056 | 0.671 | 0.124 | 0.106 | 126.416 |
| allenai_OLMo-2-1124-7B-Instruct | sweep_lt | baseline | 5.794 | 0.378 | 4.447 | 0.561 | 0.234 | 0.078 | 126.069 |
| allenai_OLMo-2-1124-7B-Instruct | sweep_lt | steer_L16_c-1.00 | 5.874 | 0.359 | 4.611 | 0.562 | 0.229 | 0.116 | 126.341 |

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

