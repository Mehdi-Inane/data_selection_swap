# Gradient Swap vs. Random and GraphCut: second-order swaps, class balance, long tails and label noise

*Research notes · 2026-11-19 · ResNet-18 · 102 selection runs + 4 reference models · repo `Mehdi-Inane/data_selection_swap`, branch `multiple-swap`*

## TL;DR

1. **No swap variant beats Random beyond noise.** Across CIFAR-100 (1–50%), CIFAR-10-LT, CIFAR-100-LT and 20%-noisy CIFAR-10, all swap variants land within about ±1 point of Random. That includes Algorithm 1, the second-order swap with k = 8 and k = 16, and the class-balanced second-order swap. Where three seeds exist, the gaps are within one standard deviation. This holds even though the swaps cut the exact full-network $$G_T$$ at the epoch-1 checkpoint about 5× below Random's.
2. **GraphCut is the only method that clearly beats Random, and only in two regimes.** On CIFAR-100 it beats Random at tiny budgets: 12.4 vs 8.7% at 1% and 15.0 vs 12.7% at 2%. On 20%-label-noise CIFAR-10 at 10% it wins 64.4 vs 56.3%. It loses badly at mid budgets: 41.1 vs 50.7% on CIFAR-100 at 25%. GraphCut picks *easy, low-gradient* examples, so its $$G_T$$ is 10–100× *higher* than Random's. That is the opposite of faithfulness.
3. **Within a budget, epoch-1 $$G_T$$ does not predict accuracy.** The within-budget Spearman correlation between accuracy and $$G_T$$ ranges from −0.5 to +0.7. It is mostly positive, the wrong sign for the theory. The strong pooled correlation (≈ −0.7) is a budget-size effect: bigger subsets have both lower $$G_T$$ and higher accuracy.
4. **Contamination pilot: the swaps do not remove mislabelled examples.**
   - The swaps' subsets keep 19.5–20.1% mislabelled examples, against 19.7% for Random. The second-order swaps add and drop mislabelled examples at the same rate (15–19% of swaps each).
   - GraphCut keeps only 5.0%.
   - Recommendation: **don't build the full Exp B with this objective.**
5. **Which examples the swaps move.**
   - Algorithm 1 only ever trades examples from the top of the gradient-norm distribution: added and dropped examples have a median at the 96–99th percentile. It then stops at its first rejected swap, after 9–200 swaps.
   - The second-order swap starts the same way: its first swaps involve examples around the 95th percentile. It then works through low-norm examples, and its last swaps add examples at the 1st–20th percentile.
   - Neither method shifts the subset's overall gradient-norm distribution; the mean percentile stays about 50.
6. **Why the large $$G_T$$ reductions don't move accuracy (best-supported explanation).** Matching gradients at the epoch-1 checkpoint does not carry over to the rest of training. At epochs 150 and 300 the swapped subsets have the **same** gradient discrepancy as the Random subset they started from. A random subset is already as faithful as its size allows later in training.

---

## 1. Setup

| | |
|---|---|
| Model | ResNet-18 (CIFAR stem), all runs |
| Reference model | One 300-epoch SGD run per dataset on the full pool, seed 42. Every selector reads its **epoch-1 checkpoint** (τ = 1). |
| Retraining | Fresh model from a shared per-(dataset, seed) initialisation, 300 epochs, SGD (Nesterov 0.9), lr 0.01, wd 5e-4, cosine schedule, batch 128, crop + flip |
| Metric | Final-epoch top-1 accuracy on the clean, balanced test set (10,000 images) |
| Scoring view | Clean images, BatchNorm in eval mode |
| Hardware | 4× A100-80GB on one machine, one task per GPU |

**Methods**

| Name in tables | What it does |
|---|---|
| Random | Uniform draw of m pool examples (seeded) |
| GraphCut | apricot GraphCut (λ = 1) on penultimate features of the epoch-1 checkpoint, with per-class quotas |
| Single swap (Alg. 1) | Paper's Algorithm 1 on exact full-network gradients: swap the top linear-ranked pair; stop at the first rejection or after 200 swaps |
| Second-order k | Shortlist the k best drops and k best adds by their exact single-example effect, take the best of the k × k exact swaps, and accept only if $$G_T$$ falls. Runs on a 1024-dim count sketch of the gradients; at most 1500 swaps |
| Second-order k=16, class-bal. | Same, but the start is stratified at quotas proportional to the pool's class counts, and every swap stays inside one class (16 × 16 per class) |

All swap runs except the class-balanced one start from **exactly the Random subset** (same seeded draw), so "swap vs Random" is a paired comparison. The class-balanced run starts from its own stratified draw.

**Datasets and budgets**

| Dataset | Pool | Budgets | Seeds | Methods |
|---|---|---|---|---|
| CIFAR-100 | 50,000 | 5, 10, 25, 50% | 42 | all 6 |
| CIFAR-100, low budget | 50,000 | 1, 2% | 42, 562, 777 | Random, GraphCut, SO k=16, SO class-bal. |
| CIFAR-10-LT (ratio 50) | 13,996 (5,000 → 100 per class) | 5, 10, 25, 50% | 42 | all 6 |
| CIFAR-100-LT (ratio 50) | 12,608 (500 → 10 per class) | 5, 10% | 42, 562, 777 | Random, GraphCut, SO k=16, SO class-bal. |
| CIFAR-10 + 20% label noise | 50,000 (10,000 wrong labels) | 10% | 42 | all 6 |

The full-data reference models reach 75.11% (CIFAR-100) and 73.81% (CIFAR-10-LT) test accuracy.

---

## 2. Results

### 2.1 CIFAR-100: swaps match Random; GraphCut collapses at 25%

Test accuracy (%), seed 42. **Bold** = best in column.

| Method | 5% | 10% | 25% | 50% |
|---|---|---|---|---|
| Random | 21.02 | **32.76** | 50.74 | 65.16 |
| GraphCut | 20.83 | 29.25 | 41.10 | 63.45 |
| Single swap (Alg. 1) | 20.99 | 32.69 | 50.41 | 65.98 |
| Second-order k=8 | **21.67** | 31.99 | **50.83** | 64.79 |
| Second-order k=16 | 20.93 | 32.46 | 50.48 | **66.07** |
| Second-order k=16, class-bal. | 20.69 | 32.21 | 50.15 | 64.54 |

Every swap variant is within 1 point of Random at every budget. The paper's 5-seed standard deviations are 0.3–1.1 points, so none of these gaps is distinguishable from retraining noise.

### 2.2 CIFAR-100 at 1% and 2%: GraphCut wins, the swaps don't

Mean ± SD over 3 seeds.

| Method | 1% (≈5 / class) | 2% (≈10 / class) |
|---|---|---|
| Random | 8.69 ± 0.94 | 12.71 ± 0.70 |
| GraphCut | **12.43 ± 0.28** | **15.04 ± 0.23** |
| Second-order k=16 | 8.42 ± 0.33 | 12.99 ± 0.15 |
| Second-order k=16, class-bal. | 9.06 ± 0.10 | 13.17 ± 0.65 |

GraphCut is deterministic: it picks the same subset for every seed, so its SD is retraining noise only. Even so, its +3.7 and +2.3 point margins are 4–5× Random's SD.

### 2.3 CIFAR-10-LT: everyone ties; the class-balanced swap is best at 25–50% by ~0.7 points

Test accuracy (%), seed 42.

| Method | 5% | 10% | 25% | 50% |
|---|---|---|---|---|
| Random | 35.35 | **40.22** | **51.09** | 62.13 |
| GraphCut | 35.17 | 39.07 | 46.73 | 60.95 |
| Single swap (Alg. 1) | 33.75 | 39.96 | 49.48 | 62.36 |
| Second-order k=8 | 35.09 | 38.92 | 49.38 | 61.66 |
| Second-order k=16 | **35.80** | 39.36 | 49.48 | 62.19 |
| Second-order k=16, class-bal. | 32.94 | 38.73 | 50.79 | **62.79** |

### 2.4 CIFAR-100-LT: everyone ties

Mean ± SD over 3 seeds.

| Method | 5% | 10% |
|---|---|---|
| Random | 9.57 ± 0.44 | 13.37 ± 0.29 |
| GraphCut | **9.94 ± 0.08** | 12.88 ± 0.23 |
| Second-order k=16 | 9.53 ± 0.18 | **13.56 ± 0.41** |
| Second-order k=16, class-bal. | 9.49 ± 0.19 | 12.75 ± 0.17 |

### 2.5 Label-noise pilot (CIFAR-10, 20% wrong labels, 10% budget): GraphCut avoids the noise, the swaps keep it

| Method | Test acc. (%) | Mislabelled in subset | Swaps | Mislabelled added / dropped by swaps |
|---|---|---|---|---|
| Random | 56.28 | 19.7% | – | – |
| GraphCut | **64.41** | **5.0%** | – | – |
| Single swap (Alg. 1) | 55.45 | 19.5% | 71 | 25 / 35 |
| Second-order k=8 | 56.45 | 19.7% | 669 | 117 / 116 |
| Second-order k=16 | 56.02 | 19.8% | 615 | 114 / 107 |
| Second-order k=16, class-bal. | 53.55 | 20.1% | 799 | 121 / 118 |

**Pilot verdict:** none of the swaps moves the mislabelled share away from the pool's 20%. Exp B (SVHN or Gaussian contamination, all architectures) is not worth building for the current objective.

---

## 3. $$G_T$$ versus accuracy

To compare every method on one scale, I recomputed $$G_T$$ for all 102 subsets on the exact, full-network gradients (11.2M parameters, no sketch). It is computed separately at the reference checkpoints of epochs 1, 150 and 300:

$$G^{(k)}(I)=\Big\|\tfrac{1}{m}\textstyle\sum_{i\in I}\nabla\ell(\theta_k,z_i)-\nabla L_S(\theta_k)\Big\|^2 .$$

$$G^{(1)}$$ is exactly what the swaps minimise, up to the sketch.

### 3.1 The swaps lower epoch-1 $$G_T$$ about 5×, not the ~100× they report

CIFAR-100, exact $$G^{(1)}$$; lower is "more faithful".

| Method | 5% | 10% | 25% | 50% |
|---|---|---|---|---|
| Random | 2.255 | 0.961 | 0.365 | 0.131 |
| GraphCut | 37.3 | 31.4 | 20.2 | 1.78 |
| Single swap (Alg. 1) | 0.766 | 0.267 | 0.083 | 0.025 |
| Second-order k=8 | 0.425 | 0.204 | 0.069 | 0.022 |
| Second-order k=16 | 0.436 | 0.201 | 0.068 | 0.023 |
| Second-order k=16, class-bal. | **0.405** | **0.185** | **0.059** | **0.021** |

- **Sketch versus exact.** The second-order swap's own log says $$G_T$$ fell 2.29 → 0.019 (−99%) at 5%. On exact gradients the drop is 2.26 → 0.44 (−81%). The swap has largely optimised the 1024-dim sketch rather than the full gradient. Still, at every budget it ends at about 0.2× Random's $$G^{(1)}$$, versus 0.2–0.34× for Algorithm 1.
- **Class balance alone helps.** The stratified start has a lower $$G^{(1)}$$ than the random start before any swap (for example 0.063 vs 0.135 sketched at CIFAR-100 50%).

### 3.2 The epoch-1 improvement is gone by epoch 150

CIFAR-100, exact $$G$$ at later reference checkpoints.

| Method | 5%, ep 150 | 5%, ep 300 | 50%, ep 150 | 50%, ep 300 |
|---|---|---|---|---|
| Random | 0.0109 | 0.00583 | 0.00065 | 0.00016 |
| Single swap | 0.0109 | 0.00580 | 0.00065 | 0.00016 |
| Second-order k=8 | 0.0104 | 0.00558 | 0.00064 | 0.00015 |
| Second-order k=16 | 0.0106 | 0.00514 | 0.00064 | 0.00015 |
| GraphCut | **0.0068** | **0.00227** | **0.00061** | 0.00020 |

The swaps' 3–5× advantage at epoch 1 shrinks to 0–12% at epochs 150 and 300. The theory's bound is a sum over the *whole* trajectory, so only the epoch-1 term was improved. Oddly, GraphCut, which is the least faithful at epoch 1, is the most faithful later at small budgets. This is the same pattern at 1–2%:

| 1% budget | ep 1 | ep 150 |
|---|---|---|
| Random | 9.93 | 0.049 |
| GraphCut | 44.1 | 0.014 |

### 3.3 Within a budget, epoch-1 $$G_T$$ does not rank subsets by accuracy

Spearman correlation between accuracy and exact $$G^{(1)}$$ within one dataset and budget. The theory predicts a **negative** sign (lower $$G$$, higher accuracy). n is the number of subsets.

| Dataset | Budget | All methods | Without GraphCut |
|---|---|---|---|
| CIFAR-100 | 1% | +0.46 (n=12) | −0.23 (n=9) |
| CIFAR-100 | 2% | +0.34 (12) | −0.51 (9) |
| CIFAR-100 | 5% | +0.09 (6) | +0.40 (5) |
| CIFAR-100 | 10% | −0.03 (6) | +0.70 (5) |
| CIFAR-100 | 25% | −0.14 (6) | +0.50 (5) |
| CIFAR-100 | 50% | −0.09 (6) | +0.60 (5) |
| CIFAR-10-LT | 5 / 10 / 25 / 50% | +0.31 / +0.43 / −0.06 / −0.37 | +0.30 / +0.70 / +0.67 / +0.10 |
| CIFAR-100-LT | 5 / 10% | +0.49 / +0.13 (12) | +0.03 / +0.67 (9) |
| CIFAR-10 + noise | 10% | +0.37 (6) | −0.10 (5) |

**Pooled across budgets** (accuracy vs log₁₀ $$G^{(1)}$$): −0.69 on CIFAR-100 (n = 48), −0.67 on CIFAR-10-LT (24), −0.17 on CIFAR-100-LT (24).

Reading: the pooled correlation is real but comes from subset size, not from which examples are chosen. Within a budget, the sign is inconsistent and usually positive. With n = 5–12 and one seed for most cells these are rough signals. Still, nothing here supports "lower epoch-1 $$G_T$$ → higher accuracy" among reasonable subsets. This doesn't contradict the paper's Figure 1 (r ≈ 0.99): that correlation is driven by catastrophic subsets (EL2N) with very large $$G_T$$, and none of our methods produced one.

---

## 4. What do the swaps actually swap?

For every pool example I computed, at the epoch-1 checkpoint, its exact gradient norm ‖∇ℓ‖, its loss, and whether the model classifies it correctly. The table places the examples each swap run added and dropped in the pool's gradient-norm distribution (percentile 0 = smallest).

**CIFAR-100, 10% budget** (other budgets and datasets look the same):

| Method | Swaps | Added: median pct. | Added in top 10% | Dropped: median pct. | First 10% of adds | Last 10% of adds | Subset mean pct. before → after |
|---|---|---|---|---|---|---|---|
| Single swap (Alg. 1) | 83 | 98.3 | 86% | 96.9 | 99.8 | 99.0 | 49.4 → 49.5 |
| Second-order k=16 | 582 | 35.4 | 12% | 30.2 | 97.6 | 1.3 | 49.4 → 49.9 |
| Second-order k=16, class-bal. | 787 | 30.7 | 10% | 25.9 | 95.5 | 2.6 | 49.6 → 50.3 |

- **Algorithm 1 only trades high-gradient examples.** At every budget and on every dataset, its added and dropped examples have median percentiles of 94–99. Almost all are misclassified at epoch 1. It ranks by the linear score $$s_r \propto \langle g_r, \text{error}\rangle$$, which grows with ‖g_r‖. That pulls the largest-norm examples to the top, and their own curvature ‖g_j − g_i‖²/m² soon makes the top pair a losing swap, so it stops. On CIFAR-10-LT it stops after only 9–22 swaps.
- **The second-order swap is not biased toward high-gradient data.** Its first swaps are large-norm trades, which fix the biggest pieces of the mismatch. It then fine-tunes with small-norm examples, which shift $$G_T$$ by small, precise amounts. Over a whole run, 54–77% of its added examples come from the *bottom half* of the pool by gradient norm.
- **Neither method changes what kind of data the subset holds.** The subset's mean gradient-norm percentile, loss and misclassification rate stay at the pool average (≈ 50th percentile). The swaps replace examples with others of the same kind; they don't make the subset harder or easier.
- **Non-balanced swaps cross classes.** Only 0–7% of CIFAR-100 swaps replace an example with one of the same class (26–43% for the second-order swap on CIFAR-10-LT). As a by-product the class counts move toward the pool's proportions: at 5% the worst class's deviation from its expected count drops from 48% (Random) to 28% (second-order k=8).
- **Few examples change at large budgets.** The swapped subsets share this fraction of their examples with the Random start:

  | Budget | Second-order | Algorithm 1 |
  |---|---|---|
  | 1% | 52% | – |
  | 5% | 70% | 97.5% |
  | 50% | 92% | 98.4% |

**GraphCut is the opposite.** At budgets up to 25%, its subsets sit at the 20–30th percentile of gradient norm (median), with 0–3% in the top 10%. Its examples have a lower epoch-1 loss (2.3–2.7 vs 3.2 for Random on CIFAR-100) and a lower misclassification rate (51–67% vs 80%). At 50% it is closer to the pool (41st percentile). It picks prototypical, easy examples. This explains all three of its results: it keeps few mislabelled examples, it wins at tiny budgets, and it loses at 25%.

---

## 5. Class balance and the long tail

**What "class-balanced" means here.** The quotas are *proportional to the pool* (`subsets.per_class_budget`). On CIFAR-10-LT at 5% that is 250, 161, 105, 68, 44, 28, 18, 12, 8, 5 images. The method preserves the long tail; it does not rebalance it.

| Observation | Evidence |
|---|---|
| Class balance doesn't help accuracy | −2.9 to +1.3 points against the plain second-order swap (k=16); better in 4 of 14 cells, worse in 10 |
| Fixed proportional quotas can starve the tail | CIFAR-10-LT 5%: class-balanced gives the 3 rarest classes 25 images (Random happened to give 33), and their test accuracy drops: 6.0% vs 9.9% |
| Covering more classes reduces "dead" classes but not mean accuracy | CIFAR-100-LT 5%: classes with ≥ 1 training image are Random 85, second-order 90, class-balanced 97, GraphCut 97. Classes at 0% test accuracy: 30.0, 25.0, 17.7, 16.7. Overall accuracy: 9.57, 9.53, 9.49, 9.94 |
| On CIFAR-10-LT the tail is never empty | The rarest class has 100 images, so every method gives it at least 5; none had a class at 0% accuracy |

**Per-class-group test accuracy, CIFAR-10-LT** (head = classes 0–2, mid = 3–6, tail = 7–9):

| Budget | Method | Head | Mid | Tail |
|---|---|---|---|---|
| 5% | Random | 70.9 | 27.7 | **9.9** |
| 5% | GraphCut | 63.7 | **32.8** | 9.8 |
| 5% | Second-order k=16 | 70.0 | 29.6 | 9.8 |
| 5% | Second-order k=16, class-bal. | **72.6** | 23.4 | 6.0 |
| 50% | Random | 88.6 | 62.4 | 35.4 |
| 50% | GraphCut | 86.6 | 60.2 | 36.3 |
| 50% | Second-order k=16 | **89.1** | 61.7 | 35.9 |
| 50% | Second-order k=16, class-bal. | 88.5 | **62.6** | **37.4** |

---

## 6. Why: hypotheses, ranked by how well the data supports them

1. **Faithfulness at epoch 1 is not faithfulness over training.** *(Strongly supported, §3.2.)* The selection matches gradients only at θ₁. After a few dozen epochs a subset model has fit its own m examples, and from then on its gradients differ from the full data's whatever was chosen. Measured on the reference trajectory, the swapped subsets have the same $$G$$ as Random at epochs 150 and 300. The KL bound sums over every step, so the improvement covers a small part of it.
2. **A random subset is already near the floor that matters.** *(Supported, §3.1.)* Random's $$G^{(1)}$$ follows the paper's Theorem 4.6(ii), $$\mathbb E\,G \propto (n-m)/(m(n-1))$$. Measured: 2.26 at 5% vs 0.13 at 50%, a 17× ratio against a predicted 19×. That is pure sampling noise, and accuracy at a given m is insensitive to it in the "reasonable subset" regime. The paper's Appendix I reaches the same conclusion: once a subset is inside the low-$$G_T$$ envelope, further reductions don't raise accuracy.
3. **The sketch overstates progress.** *(Supported, §3.1.)* With m ≥ 500 free choices and a 1024-dim target, the second-order swap can drive the sketched objective almost to zero. Its exact $$G^{(1)}$$ plateaus at about 0.2× Random's.
4. **Unfaithful selection can be better for accuracy.** *(Supported in two regimes.)* The faithfulness objective aims to reproduce full-data training, and reproduces whatever that training contains, including mislabelled examples. GraphCut is deliberately unfaithful: it picks prototypes and drops atypical (often mislabelled) examples. That helps when data is very scarce (1–2%) or noisy. It hurts at mid budgets, where hard examples carry the signal. This is consistent with the "keep easy examples when data is scarce, hard ones when it is plentiful" result of Sorscher et al. (2022). The paper's guarantee is one-sided: a small $$G_T$$ bounds the gap to *full-data training*, and says nothing about beating it.
5. **Single-seed noise.** *(Certain, but not the whole story.)* Retraining noise (SD 0.1–0.9 across 3 seeds here) is as large as most gaps in §2.1 and §2.3. It cannot explain GraphCut's margins at 1–2% or under label noise.

---

## 7. Where a swap could still win, and what to run next

1. **Optimise $$G_T$$ over the trajectory, not just epoch 1.** Select with several spread checkpoints (τ = 4–10, including late ones) and check whether $$G^{(150)}$$ and $$G^{(300)}$$ actually fall. If they do and accuracy still doesn't move, hypothesis 1 is ruled out as the cause.
2. **Measure faithfulness directly, not only accuracy.** The paper claims "faithful to full-data training". Test agreement with the full model's predictions, per-class gaps, and spread across seeds. Swaps may win there even when they tie on accuracy. The retrained models are gone (see §8), but the indices are kept, so this needs only retraining.
3. **Warm-start the swap from GraphCut at 1–2% and under label noise.** Does lowering $$G_T$$ from a prototype subset keep GraphCut's gains, or erase them by pulling high-norm and mislabelled examples back in? Note that §4 predicts the second-order swap's late swaps add low-norm examples.
4. **A clean target gradient for noisy data.** Replace $$\bar g$$ (the pool's mean gradient) by the mean gradient of a small clean validation set. That changes the objective, but it is the version that could drop mislabelled examples.
5. **Equal-per-class quotas** on CIFAR-100-LT. The current class balance is proportional and never adds tail examples.
6. **More seeds for the main grid** (3–5). Every single-seed cell in §2.1 and §2.3 is within noise.

---

## 8. Caveats

- Single seed (42) for the CIFAR-100 5–50% grid, CIFAR-10-LT and the noise pilot; three seeds only for CIFAR-100 1–2% and CIFAR-100-LT.
- ResNet-18 only; VGG-16 was dropped mid-run to save compute.
- Every seed uses the same seed-42 reference trajectory, as in Appendix G.1.
- $$G_T$$ uses eval-mode BatchNorm on clean images, a reproducible stand-in for the train-mode gradient field the reference was trained with.
- Retraining is SGD, not Langevin; the KL bound is not claimed to hold here.
- Selection times were measured with 4 jobs sharing one machine (16 vCPUs). Approximate costs:

  | Method | Selection time (CIFAR-100) | Peak GPU memory |
  |---|---|---|
  | Second-order | ~90 s (88 s of it caching per-example gradients) | 6.3 GB |
  | Class-balanced second-order | 130–170 s | – |
  | Algorithm 1 | 216–1,202 s | 2.2 GB |
  | GraphCut | 5–21 s | 0.2 GB |

- The GPU machine has been released. **Retrained model weights were not kept.** Indices, swap histories, metrics, timings and the analysis JSON were saved (§9).

## 9. Reproducibility

| Item | Where |
|---|---|
| Configs | `configs/revision/cifar100_second_order.yaml`, `cifar10_lt_second_order.yaml`, `cifar100_low_budget.yaml`, `cifar100_lt_pilot.yaml`, `cifar10_noisy_pilot.yaml` |
| Code | `a0942ef` (pushed); `be7a80c` (class-balanced stage) and `6e9fc31` (`cifar10_noisy20` dataset + pilot configs), committed but **not yet pushed** |
| Run one task | `python scripts/experiments/run_config.py run-task CONFIG STAGE TASK --data-root DATA` |
| Raw outputs | `~/report/arts.tgz` on the Lor Computer sandbox (4.6 MB): every `*_indices.pt`, `*_init_indices.pt`, `*_objective.json` (swap histories), `*_metrics.json`, `*_timing.json`, the reference manifests, `report_data_<dataset>.json` (exact $$G$$, per-class accuracy, class counts, overlaps) and `swap_stats*_<dataset>.json` (per-swap example statistics) |
| Compute | 4× A100-80GB for about 4.7 h, roughly $55 (estimate from list prices) |
