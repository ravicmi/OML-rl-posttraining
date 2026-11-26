# Presentation guide: RL post-training of Qwen3-0.6B on math

Covers only experiments that finished: the 100-step screen, the 500-step algorithm comparison (4 arms x 10 seeds), the pass-rate / curriculum study (8 arms x 4 seeds), the full-MATH-test re-evaluation, and the pass-rate-noise analysis. Cap pilot, learning-rate sweep and 16/32-sample retraining runs are still running and are NOT included.

Math is written in LaTeX ($...$). Numbers come from `outputs/RESULTS.md` and `outputs/eval_full/`.

---

## 0. The 60-second story

1. Start from Qwen3-0.6B (already instruction-tuned), thinking mode off. Give it math problems and a **binary, automatically checkable reward**: 1 if the last `\boxed{}` equals the gold answer, else 0.
2. Train with GRPO-family policy gradient for 500 steps. Every variant lifts MATH-500 pass@1 from **49.4 to about 60-61** (+11 points) and GSM8K from 60 to 71-73.
3. The **loss variant barely matters** (Dr.GRPO, GRPO-style GSPO, DAPO are within noise). What matters is **which problems you train on**.
4. A GRPO group only teaches something if its 8 samples contain both a success and a failure. About 57% of groups are all-fail or all-pass and give **zero gradient**. Keeping only problems the starting model solves sometimes (0 < p < 1) removes much of that waste and gives **+1.1 points (10 seeds, t = 3.2)**. A random subset of the same size gives nothing, so it is difficulty, not data size.
5. The best slice is the "mid" band (pass rate 3/8 to 5/8), where signal is theoretically largest: 61.8 (+2.2 over plain Dr.GRPO, 4/4 seeds). Ordering easy to hard does not help.
6. Pass rates measured with 8 samples are noisy (38% of "never solved" problems are solvable when 32 samples are used), so the curriculum was built from noisy labels.

---

## 1. What came before (background)

| Stage | Idea | Key limitation it leaves |
|---|---|---|
| Pre-training + SFT | Learn to imitate text / demonstrations: maximise $\log \pi_\theta(y\mid x)$ | Imitation: cannot exceed the demonstrations, learns style not verified correctness |
| RLHF with PPO | Maximise a learned reward with a clipped policy-gradient and a learned value network (critic) | Critic is as big as the policy (memory), hard to train, reward model can be gamed |
| REINFORCE / RLOO | Policy gradient with a baseline from other samples, no critic | High variance if baseline is poor |
| **RLVR** (verifiable rewards) | Reward = programmatic check (final answer, unit tests). No reward model | Reward is sparse: binary, only at the end |
| **GRPO** (DeepSeekMath 2024, used in DeepSeek-R1 2025) | Sample a **group** of $G$ answers per prompt, use the group mean as baseline. No critic | Normalisation choices introduce biases (below) |
| Dr.GRPO, DAPO, GSPO (2025) | Fix specific GRPO issues: length / difficulty bias, clipping, sequence-level ratios | Mostly validated on large models and long chain-of-thought |
| Curriculum learning (Bengio 2009), dynamic sampling (DAPO) | Order or filter data by difficulty | Need difficulty estimates; online filtering wastes rollouts |

**Our project question:** for a *small* model (0.6B) with a fixed compute budget, (a) does the choice of GRPO-family loss matter, and (b) does **curriculum / data selection by difficulty** matter more?

---

## 2. Theory and mathematics

### 2.1 The objective

Policy $\pi_\theta$ (the LLM), prompt $x\sim\mathcal D$, completion $o\sim\pi_\theta(\cdot\mid x)$, reward $r(x,o)\in\{0,1\}$.

$$J(\theta)=\mathbb E_{x\sim \mathcal D}\,\mathbb E_{o\sim\pi_\theta(\cdot|x)}\,[\,r(x,o)\,]=\mathbb E_x[\,p_\theta(x)\,],\qquad p_\theta(x)=\Pr_{o\sim \pi_\theta}[\text{correct}]$$

So for binary reward, $J$ is simply the **average pass rate** over the training problems. The policy-gradient theorem (REINFORCE):

$$\nabla_\theta p_\theta(x)=\mathbb E_{o}\big[r(x,o)\,\nabla_\theta\log\pi_\theta(o\mid x)\big]=\mathbb E_{o}\big[(r-b)\,\nabla_\theta\log\pi_\theta(o\mid x)\big]$$

for any baseline $b$ that does not depend on $o$ (it subtracts zero in expectation but reduces variance). The score function $\nabla\log\pi_\theta(o|x)=\sum_t\nabla\log\pi_\theta(o_t\mid x,o_{<t})$ is summed over tokens.

### 2.2 GRPO: baseline from the group

For each prompt sample $G$ completions $o_1..o_G$ (we use $G=8$), rewards $r_i$, group mean $\mu=\frac1G\sum_i r_i$, std $\sigma$:

$$\hat A_i = \frac{r_i-\mu}{\sigma+\varepsilon}\quad(\text{GRPO}),\qquad \hat A_i=r_i-\mu\quad(\text{Dr.GRPO})$$

Per-token importance ratio $\rho_{i,t}=\pi_\theta(o_{i,t}|\cdot)/\pi_{\theta_{old}}(o_{i,t}|\cdot)$, PPO clipping $\varepsilon$:

$$\mathcal L_{\text{GRPO}}=-\frac1G\sum_{i=1}^{G}\frac1{|o_i|}\sum_{t=1}^{|o_i|}\Big[\min\big(\rho_{i,t}\hat A_i,\ \text{clip}(\rho_{i,t},1\pm\varepsilon)\hat A_i\big)-\beta\,\mathrm{KL}(\pi_\theta\|\pi_{ref})\Big]$$

**Unbiasedness of the group baseline (Dr.GRPO).** Since $r_i-\mu=\frac{G-1}{G}\,(r_i-\mu_{-i})$ where $\mu_{-i}$ is the mean of the other $G-1$ samples (independent of $o_i$):

$$\mathbb E\Big[\tfrac1G\textstyle\sum_i (r_i-\mu)\nabla\log\pi(o_i)\Big]=\tfrac{G-1}{G}\,\nabla_\theta p_\theta(x)$$

i.e. the unscaled group-mean baseline gives the true policy gradient up to a constant $\frac{G-1}{G}=0.875$. The std-scaling in GRPO breaks this, because the weight changes with the prompt's difficulty (see 2.4).

### 2.3 The key fact: no mixed outcomes means no gradient

If all $G$ rewards are equal ($r_i=\mu$ for all $i$) then $\hat A_i=0$ and the group contributes **exactly zero gradient**. For a prompt with true pass rate $p$:

$$\Pr[\text{group informative}]=1-p^{G}-(1-p)^{G}$$

For $G=8$:

| true $p$ | 0.02 | 0.05 | 0.10 | 0.25 | 0.50 | 0.75 | 0.95 |
|---|---|---|---|---|---|---|---|
| P(informative) | 0.15 | 0.34 | 0.57 | 0.90 | 0.99 | 0.90 | 0.34 |

Even among informative groups the **size** of the signal peaks at $p=0.5$. By Cauchy-Schwarz, for a unit direction $u$:

$$\big|u^\top\nabla_\theta p_\theta\big|=\big|\mathrm{Cov}(r,\,u^\top\nabla\log\pi)\big|\le\sqrt{p(1-p)}\;\sqrt{\mathbb E\,(u^\top\nabla\log\pi)^2}$$

and also $\mathbb E[(r_i-\mu)^2]=\frac{G-1}{G}\,p(1-p)$. So the usable signal scales like $\sqrt{p(1-p)}$ (variance $p(1-p)$), which is **maximal at $p=\tfrac12$ and zero at $p\in\{0,1\}$**. This is the theoretical reason to expect that training on mid-difficulty problems is most efficient. The "learnability" score $p(1-p)$ is used for this purpose in the curriculum-RL literature.

**Expected useful prompts per step.** With 16 prompts per step, informative prompts $=16\times(1-\text{zero-signal fraction})$. Measured in our logs: plain Dr.GRPO has 57% zero-signal groups, so about **6.9** informative prompts per step; with the filter, 43%, so about **9.1** (+32% effective batch at identical compute).

### 2.4 The algorithm variants we compare (what each changes)

| Variant | Loss normalisation | Advantage | KL ($\beta$) | Other |
|---|---|---|---|---|
| **GRPO** | per sequence: $\frac1{|o_i|}\sum_t$ | std-scaled | 0.04 | |
| **Dr.GRPO** | constant: $\frac{1}{G\,L_{\max}}\sum_i\sum_t$ | $r_i-\mu$ (no std) | 0 | |
| **DAPO** | token-level over batch: $\frac{1}{\sum_i|o_i|}\sum_i\sum_t$ | std-scaled | 0 | clip-higher $\varepsilon_{hi}=0.28$; truncated samples masked out |
| **GSPO** | sequence-level ratio $\rho_i=\exp\big(\frac1{|o_i|}\sum_t\log\frac{\pi_\theta}{\pi_{old}}\big)$ | std-scaled | 0 | tiny clip $3\!\times\!10^{-4}$/$4\!\times\!10^{-4}$ |

Two biases Dr.GRPO targets:

* **Length bias.** In GRPO a wrong answer ($\hat A<0$) is divided by its length: per-token penalty $\propto 1/|o_i|$, so long wrong answers are punished less per token, and short correct answers are rewarded more per token. This incentivises long, rambling wrong answers.
* **Difficulty bias.** Dividing by $\sigma=\sqrt{p(1-p)}$ weights each prompt by $1/\sigma$, over-weighting the very easy and very hard prompts. For $G=8$ with one success: unscaled advantage of the success is $+0.875$, std-scaled it is $\approx+2.6$ ($\sigma\approx0.33$).

**Important detail of our setup.** We do one optimiser step per rollout batch (`num_iterations=1`), so the sampling policy equals the current policy, $\rho=1$ at the gradient step and **clipping is inactive**. Then clip-higher (DAPO) and the sequence-level clipping of GSPO cannot act, and the variants differ only by normalisation, advantage scaling, KL and masking. This is a good explanation for why they behave alike here (inference from the configuration; the trainer's internals were not instrumented).

### 2.5 Metrics

* **pass@1** = mean accuracy per sample. **pass@k** (unbiased estimator, $n$ samples of which $c$ correct):
$$\text{pass@}k=\mathbb E_x\Big[1-\frac{\binom{n-c_x}{k}}{\binom{n}{k}}\Big]$$
pass@8 tells whether the model *can* solve the problem; pass@1 tells whether it *reliably* does.
* **Paired same-seed difference.** The same seed gives the same prompt order, so we compare arm $A$ vs $B$ seed by seed: $d_s=m^A_s-m^B_s$, $t=\bar d/(s_d/\sqrt n)$, $s.e.=s_d/\sqrt n$. Pairing removes shared prompt-order noise.
* **Noise level:** seed-to-seed std is about 1 point. Detectable gap is about 1.5 points with 4 seeds and about 1.1 with 10.

---

## 3. Hypotheses

| # | Hypothesis | Prediction |
|---|---|---|
| H1 | RL with a verifiable reward improves a post-SFT 0.6B model | pass@1 up on MATH/GSM8K; AIME stays near zero (too hard) |
| H2 | RL mostly **sharpens** rather than adds capability | pass@1 rises much more than pass@8 |
| H3 | Loss details (length norm, std scaling, token- vs sequence-level, clipping) change the result | differences between GRPO-family losses |
| H4 | Zero-signal groups waste compute; filtering to $0<p<1$ helps | filter > no filter; random subset of same size does **not** help |
| H5 | The most informative problems are at $p\approx0.5$ (Sec. 2.3) | mid band > hard band > easy band |
| H6 | Order matters (easy to hard curriculum) | e2h > shuffled |
| H7 | The 8-sample difficulty estimate is accurate enough | (tested; it is not) |

---

## 4. Dataset and setup

**Training data.** MATH training split (Hendrycks et al. 2021): 7,498 competition problems (2 of the original 7,500 are dropped because no answer could be extracted), 7 subjects, Levels 1 to 5, solution ends with `\boxed{answer}`. Prompt: system message "Please reason step by step, and put your final answer within \boxed{}" + problem, chat template, **thinking off**.

**Evaluation sets** (none overlaps training):

| Set | Size | Role |
|---|---|---|
| MATH-500 | 500 | main in-distribution test (the PRM800K test subset) |
| **MATH-rest** | 4,500 | the other MATH test problems; larger, lower-noise confirmation |
| GSM8K test | 1,319 | easier, out-of-distribution (grade-school word problems) |
| AIME 2024 | 30 | very hard; a ceiling check (1 problem = 3.3 points per sample) |

**Reward.** Extract the last brace-balanced `\boxed{...}`; correct if it matches the gold after normalisation or math-verify symbolic equivalence. No boxed answer or truncated output scores 0.

**Training hyperparameters (all arms).** Qwen3-0.6B, TRL GRPOTrainer with vLLM rollouts, 16 prompts x 8 samples = 128 completions per step, 500 steps, lr $2\!\times\!10^{-6}$, rollout temperature 1.0, completion cap 1024 tokens, bf16, one H100 per run (about 2.5 to 3 hours). 500 steps x 16 = 8,000 prompt draws = 1.07 epochs on the full set, but 2.0 epochs on the 3,993-problem filter set and about 5.6 epochs on the 1,427-problem mid band.

**Evaluation protocol.** Temperature 0.7, top-p 0.95, 1536 tokens, thinking off; 8 samples/problem at the end (4 at intermediate checkpoints; 4 on MATH-rest).

**Difficulty labels.** For every training problem, the *starting* model is sampled 8 times (temperature 1.0, 1024 tokens); $\hat p_x=k/8$. Counts: 2,342 problems at 0/8, 1,163 at 8/8, 3,993 in between, 1,427 in 3/8 to 5/8.

**Compute.** 7 runs at 100 steps, 72 runs at 500 steps, plus about 100 evaluations, on a shared AWS HyperPod H100 cluster with a custom job queue. A crash at one node killed some runs, which were re-run.

---

## 5. Experiments and results

### Experiment A: 100-step screen (1 seed)
Question: any large effect worth following up? Grid of {GRPO, Dr.GRPO, DAPO, GSPO} and Dr.GRPO with easy-to-hard / hard-to-easy by MATH level.

| Run | MATH-500 | GSM8K |
|---|---|---|
| base | 50.0 | 60.5 |
| GRPO | 52.8 | 62.6 |
| Dr.GRPO | 54.0 | 62.8 |
| DAPO | 52.2 | 64.3 |
| GSPO | 54.0 | 65.5 |
| Dr.GRPO easy to hard | 51.5 | 62.5 |
| Dr.GRPO hard to easy | 54.3 | 64.2 |

**Takeaway:** every RL run beats the base by 2 to 4 points after only 100 steps, but gaps between arms (1 to 3 points) are about the seed noise, so one seed cannot rank them. It motivated 500 steps and many seeds. Note: raw train reward is not comparable across curricula (e2h reward falls 0.67 to 0.20 simply because the problems get harder).

### Experiment B: 500 steps, 4 arms x 10 seeds (MATH-500, 8 samples)

| Arm | pass@1 | pass@8 | GSM8K | AIME24 | Zero-signal groups | Train samples clipped at the 1024 cap (last 50 steps) |
|---|---|---|---|---|---|---|
| Base (no RL) | 49.4 | 76.2 | 60.1 | 2.5 | | 6.6% |
| Dr.GRPO | 60.2 +- 0.9 | 81.4 | 71.3 | 2.8 | 57% | 26% |
| GSPO | 60.9 +- 1.1 | 81.2 | 72.6 | 2.5 | 60% | 34% |
| DAPO | 60.2 +- 1.3 | 80.4 | 72.5 | 2.3 | 61% | 36% |
| Dr.GRPO + filter | **61.3 +- 0.8** | 81.7 | 72.6 | 2.3 | 43% | 12% |

Paired vs Dr.GRPO (MATH-500): GSPO +0.68 (t = 1.6), DAPO -0.01 (t = 0.0), **filter +1.12 (s.e. 0.35, t = 3.2, 8/10 seeds)**. On GSM8K, GSPO, DAPO and filter are each about +1.3 (t about 2.5 to 2.7).

Findings:
1. **H1 supported:** +11 points on MATH-500, +11 to 12 on GSM8K. AIME unchanged (about 2.5, i.e. less than one problem), so the capability ceiling of a 0.6B model is not moved.
2. **H2 supported:** pass@1 +10.8 but pass@8 only +5.2. The model is making answers it could already produce *reliable* (sharpening), not learning fundamentally new skills.
3. **H3 not supported on MATH-500:** the three losses are within noise (consistent with Sec. 2.4: on-policy single step makes them nearly equivalent). DAPO and GSPO have more training samples hitting the 1024 cap (34 to 36%) than Dr.GRPO (26%), which is consistent with the length-bias issue.
4. **H4 first signal:** the filter is the only change with a clear paired gain; it reduces zero-signal groups (57% to 43%) and the share of training samples cut at the 1024 cap (26% to 12%).

### Experiment C: pass-rate / curriculum study (Dr.GRPO, 4 seeds each)

| Arm | Train set size | pass@1 (MATH-500) | vs Dr.GRPO (paired) | Zero-signal | Truncated |
|---|---|---|---|---|---|
| no filter | 7,498 | 60.2 | | 57% | 26% |
| random subset | 3,993 | 60.3 | +0.73 (t=0.9) | 58% | 28% |
| drop never-solved ($p>0$) | 5,156 | 61.3 | +1.71 (t=3.9, 4/4) | 53% | 11% |
| drop always-solved ($p<1$) | 6,335 | 60.7 | +1.07 (t=1.5) | 53% | 33% |
| filter $0<p<1$ | 3,993 | 61.3 | +1.12 (10 seeds, t=3.2) | 43% | 12% |
| hard band (1/8 to 2/8) | 1,238 | 61.5 | +1.83 (t=2.2) | 25% | 27% |
| **mid band (3/8 to 5/8)** | 1,427 | **61.8** | **+2.18 (t=2.6, 4/4)** | 39% | 8% |
| easy band (6/8 to 7/8) | 1,328 | 60.2 | +0.61 (t=1.1) | 74% | 2% |
| filter + easy to hard (3 stages) | 3,993 | 61.3 | +1.71 (t=6.2, 4/4) | 27% | 35% |
| easy to hard (no filter) | 7,498 | 60.3 | +0.67 (t=0.9) | 68% | 55% |

Findings:
1. **Selection, not size, matters (H4):** random subset of the same size gives nothing (60.3).
2. **Most of the gain comes from removing the never-solved problems:** drop-0 alone gets 61.3 (t = 3.9). Dropping the always-solved ones gives less (+1.07, not significant). Never-solved prompts are the largest block of zero-gradient groups (all-fail), so dropping them frees the most batch slots (the exact mechanism was not isolated).
3. **Band results (H5, partly):** ordering by pass rate is mid (61.8) > hard (61.5) > easy (60.2). The mid band is best nominally and has the lowest truncation among the harder arms (8%) and the highest training reward without being trivial (0.81). The hard band has even fewer zero-signal groups (25%), so zero-signal fraction alone does not predict the ranking. The mid band uses only 1,427 problems (about 5.6 epochs) and still wins, so more data is not the bottleneck. The easy band is the worst (74% zero-signal groups: the model already solves them).
4. **But the top arms are statistically tied** with each other (paired differences vs filter: mid +0.56, hard +0.21, drop-0 +0.09; s.e. 0.7 to 1.0). With 4 seeds we can say "hard/mid/filter/drop-0 all beat plain Dr.GRPO by about 1.5 to 2 points" but **cannot rank them**.
5. **Ordering does not help (H6 not supported):** easy to hard without a filter is -0.95 vs filter (t = -3.0, 0/4); with a filter it equals the filter (+0.09). Caveat: multi-stage runs restart the trainer for each stage (optimiser state reset), a confound.
6. **pass@8 barely changes across arms (80 to 82.5):** curriculum improves reliability, not the ceiling.
7. Multiple comparisons: with 8 comparisons at 4 seeds, only the strongest t-values (drop-0 3.9; fe2h 6.2) are safely beyond chance. Treat the rest as suggestive.

### Experiment D: full MATH test re-evaluation (4,500 unseen problems, 4 samples)

| Arm | seeds | pass@1 |
|---|---|---|
| base | 1 | 49.6 |
| mid band | 4 | 62.4 |
| filter + e2h | 4 | 62.4 |
| hard band | 4 | 62.1 |
| Dr.GRPO + filter | 10 | 61.8 |
| drop never-solved | 4 | 61.7 |
| GSPO | 10 | 61.3 |
| easy band | 4 | 61.3 |
| drop always-solved | 4 | 61.1 |
| easy to hard | 4 | 60.7 |
| DAPO | 10 | 60.7 |
| Dr.GRPO | 10 | 60.6 |
| random subset | 4 | 60.5 |

Same story with 9x more test problems: RL +11 points, filter +1.2 over Dr.GRPO (the MATH-500 result was not a fluke of a 500-problem test set), random subset gets nothing, bands and filter at the top. Eval noise is now small (4,500 problems), so the remaining spread is **training-seed variance** (std about 0.5 to 1.3).

### Experiment E: how reliable is the 8-sample difficulty label?
Same model, same problems, 8 vs 16 vs 32 samples (7,498 problems).

| | 8 samples | 16 | 32 |
|---|---|---|---|
| never solved | 2,342 | 1,927 | 1,558 |
| always solved | 1,163 | 764 | 493 |
| in between | 3,993 | 4,807 | 5,447 |
| mid band 3/8 to 5/8 | 1,427 | 1,241 | 1,138 |

* Of the 2,342 problems scored 0/8, **888 (38%)** are solvable at 32 samples (mean true pass rate about 0.09).
* Of the 1,163 scored 8/8, **64%** are not always solved.
* Of the 1,427 mid-band problems by 8 samples, only **702** are still mid-band at 32 samples.
* Mean absolute error between $\hat p_8$ and $\hat p_{32}$: 0.088.

Why: $\mathrm{Var}(\hat p)=p(1-p)/n$, so at $n=8$, $p=0.5$ the s.e. is 0.18. A problem with $p=0.05$ scores 0/8 with probability $0.95^8=66\%$. **Implication:** all curricula so far were built on noisy labels, so their true effect is probably underestimated; the 16 and 32-sample retraining runs (in progress) will test it.

---

### Experiment F: do better difficulty labels help? (16 and 32 samples instead of 8; 4 seeds)

| Labels | filter | mid band |
|---|---|---|
| 8 samples | 61.3 (10 seeds) | 61.8 |
| 16 samples | 61.1 | 61.8 |
| 32 samples | 61.8 | 62.2 |

Paired vs the 8-sample version: all within noise (filter 16: -0.18, 32: +0.59; mid 16: -0.01, 32: +0.35; s.e. 0.4 to 1.1). Against plain Dr.GRPO every new arm wins on 4/4 seeds (+1.4 to +2.5). **Conclusion:** the noisy 8-sample labels were not the bottleneck; cleaner labels may add a small gain (32-sample mid +0.35, 32-sample filter +0.59), but this is not detectable with 4 seeds. Labels at 32 samples remove more problems (never-solved drops 2,342 to 1,558), yet accuracy is the same.

### Experiment G: training completion cap 1536 instead of 1024 (3 to 4 seeds)

| Arm | cap 1024 | cap 1536 | paired diff |
|---|---|---|---|
| Dr.GRPO | 60.2 (10 seeds) | 59.9 (3 seeds) | -0.33 (t = -0.5) |
| filter | 61.3 | 61.9 | +0.65 (t = 1.0) |
| mid band | 61.8 | 62.3 (3 seeds) | +0.72 (t = 0.7) |

A longer cap does **not** help meaningfully. It does change behaviour: eval truncation at 1536 rises from 13% to 23-24%, pass@8 drops about 1 to 2 points (81.4 to 79.7 for Dr.GRPO), GSM8K rises about 2 points for filter/mid (72.6 to 74.7, 73.0 to 74.8). The curriculum advantage persists under the longer cap: filter - Dr.GRPO +1.8, mid - Dr.GRPO +2.5 (3 seeds). Two runs (`cap_drgrpo_s0`, `cap_mid_s0`) were killed twice (rc=137) and are not finished.

### Experiment H: learning-rate sweep (2 seeds, so only large effects are visible)

| lr | Dr.GRPO | filter |
|---|---|---|
| 1e-6 | 57.1 | 58.7 |
| **2e-6 (default)** | 60.2 | **61.3** |
| 4e-6 | 60.2 (+1.2 paired on seeds 0,1) | 59.1 (s0 56.1, s1 62.0) |
| 8e-6 | 36.4 (1 seed) | 43.0 (50.8, 35.3) |

* 1e-6 is too slow for 500 steps (-1.9 to -2.0, consistent on 2/2 seeds).
* 4e-6 is on the edge: Dr.GRPO roughly equal, filter unstable (one seed collapses to 56). For Dr.GRPO at 4e-6, 65% of training samples hit the cap and 59% of eval answers are truncated: the model drifts to very long outputs.
* 8e-6 destabilises training (accuracy 35 to 51, below the 60 of the default); for the filter arm 58% of training samples hit the cap (the single Dr.GRPO run did not show this, so treat it as one seed).
* So the default 2e-6 is close to the best of the grid; it is **not** under-tuned in the direction of a higher rate. The filter advantage holds at 1e-6 (+1.5, 2/2) and is lost at 4e-6 only because of one unstable run. Two lr runs per cell are too few for finer claims.

---

## 6. Conclusions

1. **RL with verifiable reward works on a 0.6B model:** +11 points on MATH (both test sets) and GSM8K, at about 3 GPU-hours per run. It does not help AIME and adds only about 5 points to pass@8, so it **sharpens** existing ability.
2. **The loss function is a second-order choice** here. Dr.GRPO, GSPO and DAPO are statistically tied on MATH. With one update per rollout, the clipping tricks never activate.
3. **Data selection is a first-order choice:** training only on problems with $0<p<1$ gives +1.1 to +1.2 points (10 seeds, two test sets). The mechanism is concrete: fewer zero-gradient groups (57% to 43%), +32% informative prompts per step, shorter outputs and less truncation.
4. **Which problems:** drop the never-solved first; the sweet spot is mid difficulty, as $p(1-p)$ theory predicts, but with 4 seeds it cannot be separated from hard-only or the plain filter.
5. **Order does not matter, selection does:** easy to hard adds nothing over the filter and hurts without it.
6. **Difficulty labels are noisy** at 8 samples; improving them is the next lever.

### Limitations (say these yourself, it builds credibility)
* One model (0.6B) and one dataset; not shown to transfer to larger models or other domains.
* 500 steps (about 1 epoch); lr ($2\times10^{-6}$) and 1024-token cap were not tuned. At eval, 13% of RL-trained answers (7% for the base) are cut at 1536 tokens, so part of the remaining error is truncation, not wrong math. (The "Truncated" column in the results tables is the share of *training* samples cut at the 1024 cap.)
* Four seeds per curriculum arm: can detect only about 1.5 point gaps; 8 comparisons raise the false-positive chance.
* Difficulty is measured once on the *starting* model and goes stale as the policy improves (DAPO's dynamic sampling does this online at the cost of extra rollouts).
* Multi-stage curricula reset optimiser state per stage.
* Qwen3 may have seen MATH in pre-training (contamination), which could inflate absolute numbers but not the comparisons between arms.
* The mechanism claims (clipping inactive; effective batch size) are inferred from configuration and logs, not from a dedicated ablation.

---

## 7. Intuitive summary (for the audience)

* **RL here is practice with an answer key.** The model tries a problem 8 times and learns from the differences between its attempts.
* **If it gets all 8 wrong or all 8 right, there is nothing to learn from.** That is like a student spending half their time on problems far beyond them or ones they always get right.
* **Picking the problems where it sometimes succeeds** (about half the time) is the most efficient practice. Random fewer problems do not help; the right problems do.
* **Practising easy first, then hard, doesn't add anything.** What matters is not wasting practice time.
* **The fancy loss variants are tweaks of how to average the learning signal;** with one update per batch they do almost the same thing.
* **RL doesn't make the model smarter on problems it can't do** (AIME unchanged, pass@8 +5). It makes it **more consistent** on what it could already do (pass@1 +11).
* **Our difficulty gauge is fuzzy** (8 attempts is too few), so the curriculum can likely be improved with better labels.

---

## 8. Suggested slide plan (about 15 to 18 slides, 20 min)

1. Title and question: "What matters when RL-fine-tuning a small model: the loss or the data?"
2. Background timeline: SFT, PPO/RLHF, RLVR, GRPO, Dr.GRPO/DAPO/GSPO
3. Setup in one picture: prompt, 8 samples, check answer, group baseline, update
4. Math 1: objective, policy gradient, group advantage, unbiasedness
5. Math 2: loss variants table; length bias; difficulty bias
6. Math 3: informative-group probability and $p(1-p)$; the figure to draw is P(informative) vs p
7. Hypotheses H1 to H7
8. Dataset and evaluation sets; reward function
9. Experiment design map (screen, 500-step, pass-rate study, re-eval, noise)
10. Result A+B: bar chart `figs/eval_bars.png` and the table of Experiment B
11. Result B2: pass@1 vs pass@8 (sharpening)
12. Result C: pass-rate arms bar chart `figs/prstudy.png`
13. Mechanism: zero-signal 57% to 43%, truncation, informative prompts per step
14. Result D: 4,500-problem confirmation
15. Result E: label noise (8 vs 32 samples)
16. Conclusions (5 bullets)
17. Limitations and ongoing work (cap pilot, lr sweep, 16/32-sample curricula)
18. Intuitive summary and Q&A

### Likely questions and short answers
* *Why is the difference only 1 point?* That is the typical size of selection effects against 1-point seed noise; that is why we used 10 seeds for the main comparison and paired seeds.
* *Why not just run longer?* Compute budget; the curves are still rising at 500 steps, so longer training is a future experiment.
* *Is the filter cheating by using the starting model's rollouts?* No test data are used; labels come from training problems only. The cost is one scoring pass (about 7,498 x 8 samples).
* *Why do the loss variants tie?* Section 2.4: one on-policy update, so clipping is inactive; they only differ in normalisation.
* *Why does the mid band win with only 1,427 problems?* Highest informative signal per prompt ($p(1-p)$ peak), and 5.6 epochs didn't overfit visibly; but with 4 seeds it ties hard and filter.
* *Is the RL gain real learning or just formatting?* Not mainly formatting: on MATH-500 the base already puts a boxed answer in 93% of samples and a Dr.GRPO model (seed 0) in 92%, and mean length is similar (580 vs 602 tokens). RL also raises GSM8K and pass@8. We did not run a dedicated format ablation.

### Figures already generated
`outputs/figs/training_curves.png`, `eval_bars.png` (100-step screen), `round2_curves.png`, `round2_final.png` (500-step), `prstudy.png` (pass-rate study).
