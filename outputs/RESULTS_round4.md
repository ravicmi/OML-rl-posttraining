## Reference arms (earlier rounds)

| Arm | Seeds | MATH-500 p@1 | MATH-500 p@8 | GSM8K p@1 | AIME24 p@1 | Truncated | Zero-signal |
|---|---|---|---|---|---|---|---|
| drgrpo | 10 | 60.2 ± 0.9 | 81.4 ± 1.0 | 71.3 ± 1.2 | 2.8 ± 1.1 | 13% | 57% |
| drgrpo_filter | 10 | 61.3 ± 0.8 | 81.7 ± 0.6 | 72.6 ± 1.0 | 2.3 ± 0.9 | 13% | 43% |
| pr_mid | 4 | 61.8 ± 0.8 | 81.5 ± 0.9 | 73.0 ± 0.8 | 3.1 ± 0.2 | 11% | 39% |

## A. 16- and 32-sample pass rates (Dr.GRPO, 500 steps)

| Arm | Seeds | MATH-500 p@1 | MATH-500 p@8 | GSM8K p@1 | AIME24 p@1 | Truncated | Zero-signal |
|---|---|---|---|---|---|---|---|
| filter, 8-sample labels | 10 | 61.3 ± 0.8 | 81.7 ± 0.6 | 72.6 ± 1.0 | 2.3 ± 0.9 | 13% | 43% |
| filter, 16-sample labels | 4 | 61.1 ± 0.7 | 82.8 ± 0.7 | 72.2 ± 1.1 | 2.8 ± 0.7 | 13% | 44% |
| filter, 32-sample labels | 4 | 61.8 ± 1.2 | 82.2 ± 1.2 | 71.9 ± 1.5 | 2.0 ± 0.9 | 13% | 50% |
| mid band, 8-sample labels | 4 | 61.8 ± 0.8 | 81.5 ± 0.9 | 73.0 ± 0.8 | 3.1 ± 0.2 | 11% | 39% |
| mid band, 16-sample labels | 4 | 61.8 ± 1.0 | 81.2 ± 2.2 | 72.6 ± 1.3 | 2.6 ± 0.9 | 12% | 41% |
| mid band, 32-sample labels | 4 | 62.2 ± 1.0 | 81.9 ± 1.6 | 73.6 ± 0.7 | 3.6 ± 0.9 | 13% | 42% |

| Comparison | n seeds | Mean diff (pt) | s.e. | t | Seeds won |
|---|---|---|---|---|---|
| pr16_filter - drgrpo_filter | 4 | -0.18 | 0.83 | -0.2 | 1/4 |
| pr32_filter - drgrpo_filter | 4 | +0.59 | 1.14 | +0.5 | 3/4 |
| pr16_mid - pr_mid | 4 | -0.01 | 0.41 | -0.0 | 1/4 |
| pr32_mid - pr_mid | 4 | +0.35 | 0.52 | +0.7 | 2/4 |
| pr16_filter - drgrpo | 4 | +1.44 | 0.92 | +1.6 | 4/4 |
| pr32_filter - drgrpo | 4 | +2.21 | 0.95 | +2.3 | 4/4 |
| pr16_mid - drgrpo | 4 | +2.18 | 0.82 | +2.7 | 4/4 |
| pr32_mid - drgrpo | 4 | +2.53 | 1.01 | +2.5 | 4/4 |
| pr32_filter - pr16_filter | 4 | +0.77 | 0.50 | +1.6 | 3/4 |
| pr32_mid - pr16_mid | 4 | +0.36 | 0.83 | +0.4 | 2/4 |

## B. Completion cap 1536 (train) vs 1024

| Arm | Seeds | MATH-500 p@1 | MATH-500 p@8 | GSM8K p@1 | AIME24 p@1 | Truncated | Zero-signal |
|---|---|---|---|---|---|---|---|
| drgrpo | 10 | 60.2 ± 0.9 | 81.4 ± 1.0 | 71.3 ± 1.2 | 2.8 ± 1.1 | 13% | 57% |
| cap_drgrpo | 3 | 59.9 ± 1.2 | 79.7 ± 0.7 | 71.6 ± 0.6 | 2.8 ± 1.0 | 24% | 59% |
| drgrpo_filter | 10 | 61.3 ± 0.8 | 81.7 ± 0.6 | 72.6 ± 1.0 | 2.3 ± 0.9 | 13% | 43% |
| cap_filter | 4 | 61.9 ± 0.6 | 80.4 ± 1.1 | 74.7 ± 0.8 | 2.0 ± 1.0 | 23% | 48% |
| pr_mid | 4 | 61.8 ± 0.8 | 81.5 ± 0.9 | 73.0 ± 0.8 | 3.1 ± 0.2 | 11% | 39% |
| cap_mid | 3 | 62.3 ± 1.3 | 80.3 ± 0.8 | 74.8 ± 1.1 | 2.4 ± 0.6 | 22% | 45% |

| Comparison | n seeds | Mean diff (pt) | s.e. | t | Seeds won |
|---|---|---|---|---|---|
| cap_drgrpo - drgrpo | 3 | -0.33 | 0.71 | -0.5 | 1/3 |
| cap_filter - drgrpo_filter | 4 | +0.65 | 0.66 | +1.0 | 2/4 |
| cap_mid - pr_mid | 3 | +0.72 | 1.09 | +0.7 | 2/3 |
| cap_filter - cap_drgrpo | 3 | +1.82 | 0.95 | +1.9 | 2/3 |
| cap_mid - cap_drgrpo | 3 | +2.45 | 1.05 | +2.3 | 3/3 |

## C. Learning-rate sweep (2e-6 = default; same seeds 0,1 for the paired rows)

| Arm | Seeds | MATH-500 p@1 | MATH-500 p@8 | GSM8K p@1 | AIME24 p@1 | Truncated | Zero-signal |
|---|---|---|---|---|---|---|---|
| drgrpo lr=1e-6 | 2 | 57.1 ± 0.8 | 79.2 ± 0.6 | 68.6 ± 1.0 | 3.1 ± 0.3 | 16% | 55% |
| drgrpo lr=2e-6 | 10 | 60.2 ± 0.9 | 81.4 ± 1.0 | 71.3 ± 1.2 | 2.8 ± 1.1 | 13% | 57% |
| drgrpo lr=4e-6 | 2 | 60.2 ± 0.4 | 81.1 ± 0.7 | 72.4 ± 2.0 | 1.2 ± 1.2 | 59% | 60% |
| drgrpo lr=8e-6 | 1 | 36.4 | 61.4 | 55.5 | 0.0 | 12% | 60% |
| filter lr=1e-6 | 2 | 58.7 ± 0.9 | 80.9 ± 0.1 | 69.4 ± 0.2 | 2.5 ± 0.0 | 17% | 34% |
| filter lr=2e-6 | 10 | 61.3 ± 0.8 | 81.7 ± 0.6 | 72.6 ± 1.0 | 2.3 ± 0.9 | 13% | 43% |
| filter lr=4e-6 | 2 | 59.1 ± 4.2 | 78.6 ± 3.4 | 73.0 ± 2.1 | 2.3 ± 0.3 | 13% | 46% |
| filter lr=8e-6 | 2 | 43.0 ± 11.0 | 66.4 ± 12.2 | 67.4 ± 5.9 | 0.4 ± 0.6 | 58% | 41% |

| Comparison | n seeds | Mean diff (pt) | s.e. | t | Seeds won |
|---|---|---|---|---|---|
| drgrpo: lr 1e-6 - 2e-6 | 2 | -1.89 | 0.56 | -3.4 | 0/2 |
| drgrpo: lr 4e-6 - 2e-6 | 2 | +1.20 | 0.82 | +1.5 | 2/2 |
| filter: lr 1e-6 - 2e-6 | 2 | -2.02 | 0.15 | -13.5 | 0/2 |
| filter: lr 4e-6 - 2e-6 | 2 | -1.64 | 2.44 | -0.7 | 1/2 |
| filter: lr 8e-6 - 2e-6 | 2 | -17.66 | 8.26 | -2.1 | 0/2 |
| filter - drgrpo at lr 1e-6 | 2 | +1.51 | 0.09 | +17.3 | 2/2 |
| filter - drgrpo at lr 4e-6 | 2 | -1.19 | 2.64 | -0.5 | 1/2 |

## Per-seed MATH-500 pass@1

| Arm | seeds: values |
|---|---|
| pr16_filter | s0=62.1, s1=60.9, s2=60.7, s3=60.7 |
| pr32_filter | s0=62.6, s1=62.6, s2=61.9, s3=60.1 |
| pr16_mid | s0=61.9, s1=60.4, s2=62.1, s3=62.7 |
| pr32_mid | s0=63.4, s1=62.4, s2=61.9, s3=61.0 |
| cap_drgrpo | s1=61.2, s2=59.3, s3=59.0 |
| cap_filter | s0=62.5, s1=61.2, s2=61.7, s3=62.1 |
| cap_mid | s1=62.4, s2=61.0, s3=63.5 |
| lr1e-6_drgrpo | s0=56.6, s1=57.7 |
| lr1e-6_filter | s0=58.0, s1=59.3 |
| lr4e-6_drgrpo | s0=59.9, s1=60.6 |
| lr4e-6_filter | s0=56.1, s1=62.0 |
| lr8e-6_drgrpo | s0=36.4 |
| lr8e-6_filter | s0=50.8, s1=35.3 |
