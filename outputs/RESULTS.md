# Qwen3-0.6B RL post-training ablations: collated results

Setup: Qwen3-0.6B, MATH train (7,498 problems), verifiable boxed-answer reward, 16 prompts x 8 samples per step, lr 2e-6, 1024-token completion cap, TRL GRPOTrainer + vLLM. Eval: temperature 0.7, 1536 tokens, 8 samples/problem. pass@1 in %, mean ± std across seeds.

## 1. 100-step screening grid (1 seed, 4 samples/problem)

| Run | MATH-500 p@1 | GSM8K p@1 | AIME24 p@1 | Truncated (MATH-500) | Train reward first→last |
|---|---|---|---|---|---|
| base_qwen3_0p6b | 50.0 | 60.5 | 0.8 | 6.6% | - |
| dapo_none | 52.2 | 64.3 | 3.3 | 15.0% | 0.42→0.46 |
| drgrpo_e2h | 51.5 | 62.5 | 0.0 | 12.7% | 0.67→0.20 |
| drgrpo_h2e | 54.3 | 64.2 | 2.5 | 13.2% | 0.18→0.71 |
| drgrpo_none | 54.0 | 62.8 | 0.8 | 11.2% | 0.44→0.48 |
| grpo_none | 52.8 | 62.6 | 1.7 | 10.2% | 0.43→0.46 |
| gspo_none | 54.0 | 65.5 | 1.7 | 13.6% | 0.41→0.47 |

## 2. 500-step comparison: algorithm and filter (4 arms x up to 10 seeds)

### Final results at step 500

| Arm | Seeds | MATH-500 p@1 | MATH-500 p@8 | GSM8K p@1 | AIME24 p@1 | Train reward (last 50) | Zero-signal groups | Truncated |
|---|---|---|---|---|---|---|---|---|
| Baseline (no RL, n=8) | - | 49.4 | 76.2 | 60.1 | 2.5 | - | - | - |
| Dr.GRPO | 10 | 60.2 ± 0.9 | 81.4 ± 1.0 | 71.3 ± 1.2 | 2.8 ± 1.1 | 0.54 | 57% | 26% |
| GSPO | 10 | 60.9 ± 1.1 | 81.2 ± 0.7 | 72.6 ± 0.9 | 2.5 ± 0.8 | 0.54 | 60% | 34% |
| DAPO | 10 | 60.2 ± 1.3 | 80.4 ± 1.1 | 72.5 ± 1.2 | 2.3 ± 1.1 | 0.52 | 61% | 36% |
| Dr.GRPO + pass-rate filter (0<p<1) | 10 | 61.3 ± 0.8 | 81.7 ± 0.6 | 72.6 ± 1.0 | 2.3 ± 0.9 | 0.71 | 43% | 12% |

Paired same-seed difference vs `drgrpo` (math500 pass@1, percentage points):

| Arm | n seeds | Mean diff | s.e. | t | Seeds won |
|---|---|---|---|---|---|
| gspo | 10 | +0.68 | 0.42 | 1.6 | 7/10 |
| dapo | 10 | -0.01 | 0.48 | -0.0 | 5/10 |
| drgrpo_filter | 10 | +1.12 | 0.35 | 3.2 | 8/10 |

Paired same-seed difference vs `drgrpo_filter` (math500 pass@1, percentage points):

| Arm | n seeds | Mean diff | s.e. | t | Seeds won |
|---|---|---|---|---|---|
| gspo | 10 | -0.44 | 0.45 | -1.0 | 3/10 |
| dapo | 10 | -1.13 | 0.43 | -2.6 | 2/10 |

Paired same-seed difference vs `drgrpo` (gsm8k pass@1, percentage points):

| Arm | n seeds | Mean diff | s.e. | t | Seeds won |
|---|---|---|---|---|---|
| gspo | 10 | +1.36 | 0.51 | 2.7 | 6/10 |
| dapo | 10 | +1.24 | 0.50 | 2.5 | 8/10 |
| drgrpo_filter | 10 | +1.39 | 0.51 | 2.7 | 8/10 |

#### Per-seed MATH-500 pass@1 (%)

| Arm | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
|---|---|---|---|---|---|---|---|---|---|---|
| drgrpo | 57.9 | 60.2 | 60.6 | 59.8 | 61.4 | 60.7 | 60.5 | 60.3 | 60.1 | 60.4 |
| gspo | 60.0 | 62.9 | 61.8 | 61.1 | 60.0 | 61.7 | 59.6 | 60.9 | 60.9 | 59.7 |
| dapo | 59.4 | 59.3 | 61.9 | 60.7 | 58.5 | 60.5 | 59.7 | 58.6 | 61.5 | 61.6 |
| drgrpo_filter | 60.2 | 61.2 | 60.7 | 62.8 | 60.8 | 60.6 | 62.2 | 61.5 | 61.3 | 61.8 |

#### MATH-500 pass@1 (%) at checkpoints (mean over seeds; n=4 samples at checkpoints, n=8 at step 500)

| Arm | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| drgrpo | 55.5 | 57.9 | 59.2 | 59.2 | 60.2 |
| gspo | 56.0 | 58.6 | 60.1 | 60.7 | 60.9 |
| dapo | 56.5 | 57.8 | 59.1 | 59.7 | 60.2 |
| drgrpo_filter | 56.3 | 59.3 | 60.5 | 61.0 | 61.3 |

## 3. Pass-rate / curriculum study (Dr.GRPO, 500 steps)

### Final results at step 500 (only arms with finished runs are listed)

| Arm | Seeds | MATH-500 p@1 | MATH-500 p@8 | GSM8K p@1 | AIME24 p@1 | Train reward (last 50) | Zero-signal groups | Truncated |
|---|---|---|---|---|---|---|---|---|
| Baseline (no RL, n=8) | - | 49.4 | 76.2 | 60.1 | 2.5 | - | - | - |
| no filter (7,498) | 10 | 60.2 ± 0.9 | 81.4 ± 1.0 | 71.3 ± 1.2 | 2.8 ± 1.1 | 0.54 | 57% | 26% |
| filter 0<p<1 (3,993) | 10 | 61.3 ± 0.8 | 81.7 ± 0.6 | 72.6 ± 1.0 | 2.3 ± 0.9 | 0.71 | 43% | 12% |
| random subset (3,993) | 4 | 60.3 ± 1.5 | 81.2 ± 0.8 | 71.1 ± 1.6 | 2.5 ± 0.3 | 0.55 | 58% | 28% |
| drop never-solved (5,156) | 4 | 61.3 ± 1.2 | 81.8 ± 1.7 | 72.7 ± 1.5 | 2.4 ± 1.0 | 0.77 | 53% | 11% |
| drop always-solved (6,335) | 4 | 60.7 ± 1.3 | 82.3 ± 0.9 | 72.4 ± 0.9 | 2.0 ± 1.0 | 0.48 | 53% | 33% |
| hard band p=1-2/8 (1,238) | 4 | 61.5 ± 1.2 | 82.5 ± 1.6 | 72.6 ± 0.9 | 2.8 ± 0.9 | 0.47 | 25% | 27% |
| mid band p=3-5/8 (1,427) | 4 | 61.8 ± 0.8 | 81.5 ± 0.9 | 73.0 ± 0.8 | 3.1 ± 0.2 | 0.81 | 39% | 8% |
| easy band p=6-7/8 (1,328) | 4 | 60.2 ± 1.0 | 80.1 ± 1.1 | 72.4 ± 1.1 | 2.7 ± 0.4 | 0.95 | 74% | 2% |
| filter + easy->hard | 4 | 61.3 ± 0.8 | 81.3 ± 1.7 | 72.1 ± 2.2 | 2.6 ± 0.4 | 0.43 | 27% | 35% |
| easy->hard, no filter | 4 | 60.3 ± 1.2 | 82.5 ± 0.7 | 71.2 ± 1.1 | 2.7 ± 0.2 | 0.11 | 68% | 55% |

Paired same-seed difference vs `drgrpo_filter` (math500 pass@1, percentage points):

| Arm | n seeds | Mean diff | s.e. | t | Seeds won |
|---|---|---|---|---|---|
| pr_rand | 4 | -0.89 | 1.00 | -0.9 | 1/4 |
| pr_drop0 | 4 | +0.09 | 0.95 | 0.1 | 3/4 |
| pr_drop8 | 4 | -0.55 | 1.09 | -0.5 | 2/4 |
| pr_hard | 4 | +0.21 | 0.67 | 0.3 | 2/4 |
| pr_mid | 4 | +0.56 | 0.86 | 0.7 | 2/4 |
| pr_easy | 4 | -1.01 | 0.76 | -1.3 | 1/4 |
| pr_fe2h | 4 | +0.09 | 0.64 | 0.1 | 3/4 |
| pr_e2h | 4 | -0.95 | 0.32 | -3.0 | 0/4 |

Paired same-seed difference vs `drgrpo` (math500 pass@1, percentage points):

| Arm | n seeds | Mean diff | s.e. | t | Seeds won |
|---|---|---|---|---|---|
| pr_rand | 4 | +0.73 | 0.83 | 0.9 | 3/4 |
| pr_drop0 | 4 | +1.71 | 0.43 | 3.9 | 4/4 |
| pr_drop8 | 4 | +1.07 | 0.69 | 1.5 | 3/4 |
| pr_hard | 4 | +1.83 | 0.83 | 2.2 | 3/4 |
| pr_mid | 4 | +2.18 | 0.83 | 2.6 | 4/4 |
| pr_easy | 4 | +0.61 | 0.57 | 1.1 | 3/4 |
| pr_fe2h | 4 | +1.71 | 0.28 | 6.2 | 4/4 |
| pr_e2h | 4 | +0.67 | 0.71 | 0.9 | 2/4 |

#### MATH-500 pass@1 (%) at checkpoints (mean over seeds; n=4 samples at checkpoints, n=8 at step 500)

| Arm | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| drgrpo | 55.5 | 57.9 | 59.2 | 59.2 | 60.2 |
| drgrpo_filter | 56.3 | 59.3 | 60.5 | 61.0 | 61.3 |
| pr_rand | 56.0 | 57.6 | 59.0 | 60.0 | 60.3 |
| pr_drop0 | 57.3 | 59.4 | 60.5 | 61.7 | 61.3 |
| pr_drop8 | 57.1 | 58.0 | 59.7 | 60.5 | 60.7 |
| pr_hard | 54.7 | 58.5 | 60.0 | 61.1 | 61.5 |
| pr_mid | 58.0 | 60.1 | 61.9 | 61.5 | 61.8 |
| pr_easy | 57.5 | 57.9 | 60.3 | 59.6 | 60.2 |

## 4. Not yet run

RLOO, PPO, OPSD, SDPO (need custom trainers); learning-rate and completion-cap pilot; runs longer than 500 steps.
