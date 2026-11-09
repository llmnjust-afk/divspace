# divspace experiment summary

## Diversity at the best intervention per file

| model | condition | cell | vendi | pc1_ratio | participation_ratio | distinct_2 | self_bleu | n_clusters | completion_rate | mean_words |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_pythia-410m | ablate | steer_L12_c+0.00 | 5.362 | 0.440 | 3.910 | 0.125 | 0.014 | 7.800 | 0.287 | 43.862 |
| EleutherAI_pythia-410m | control_orthogonal | orthogonal_L12_c+2.00 | 6.501 | 0.375 | 4.574 | 0.800 | 0.003 | 8.000 | 0.163 | 74.963 |
| EleutherAI_pythia-410m | control_random | random_L6_c-1.00 | 6.432 | 0.394 | 4.565 | 0.823 | 0.006 | 8.000 | 0.188 | 74.588 |
| EleutherAI_pythia-410m | control_shuffled | shuffled_L12_c+2.00 | 6.449 | 0.437 | 3.980 | 0.792 | 0.008 | 8.000 | 0.225 | 75.513 |
| EleutherAI_pythia-410m | sweep | baseline | 6.057 | 0.482 | 3.527 | 0.795 | 0.003 | 7.900 | 0.163 | 73.588 |
| EleutherAI_pythia-410m | sweep | steer_L12_c+1.00 | 6.256 | 0.411 | 4.245 | 0.810 | 0.003 | 8.000 | 0.163 | 72.987 |

## Attribution

| model | top layers (sign, magnitude) |
| --- | --- |
| EleutherAI_pythia-410m | L18 (+0.257), L6 (+0.091), L12 (-0.033) |

## Direction stability

| model | layer | split-half cosine |
| --- | --- | --- |
| EleutherAI_pythia-410m | 12 | 0.679 |
| EleutherAI_pythia-410m | 18 | 0.469 |
| EleutherAI_pythia-410m | 6 | 0.622 |

