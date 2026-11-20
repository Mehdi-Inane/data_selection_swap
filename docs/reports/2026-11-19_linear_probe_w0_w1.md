# Gradient Swaps on a Linear Probe: Selection at Initialization and After One Epoch

*Research note · 2026-11-19 · branch `multiple-swap` (commit `1846086`) · config `configs/revision/cifar100_linear_probe_w0_w1.yaml`*

## 1 Experimental setup

We study gradient-swap selection in the simplest setting where training does not move the features: a linear classifier on top of a frozen, pretrained network. We organise the experiment around three questions: (i) does minimizing the gradient discrepancy $G$ at a cheap selection point improve on random selection when the learner is a linear probe? (ii) is the improvement visible in faithfulness to full-data training, and not only in accuracy? and (iii) which selection point, initialization or one epoch of training, gives the better target, and at what computational cost?

**Datasets and features.** We use CIFAR-100 (Krizhevsky, 2009). The 50,000 training images, 500 per class, form the selection pool $S$, with $n = 50{,}000$. The 10,000 test images are used only for evaluation. Every image is embedded once by the ImageNet-1k ResNet-18 of torchvision (He et al., 2016), which is kept frozen. Images are ImageNet-normalised and bilinearly upsampled from 32 to 224 pixels inside the network, and no augmentation is used. Each image $Z_i = (x_i, y_i)$ is represented by its 512 penultimate features and a constant coordinate for the bias, $\phi_i \in \mathbb{R}^{513}$.

**Learner.** The learner $\mathcal{A}$ fits a multinomial logistic regression $W \in \mathbb{R}^{100 \times 513}$ by minimizing

$$
L_I(W) \;=\; \frac{1}{m}\sum_{i \in I} \ell(W; Z_i) \;+\; \frac{\lambda}{2}\,\|W\|^2, \qquad \ell(W; Z_i) = -\log \text{softmax}(W\phi_i)_{y_i},
$$

with full-batch L-BFGS from $W = 0$, run to convergence: the largest final gradient norm over all fits is $1.4 \times 10^{-6}$. The objective is $\lambda$-strongly convex and the fit is deterministic, so the model trained on a subset $I$ depends only on $I$. We write $W^\star$ for the model trained on all of $S$.

**Selection points.** For a point $W$ and a subset $I \subseteq [n]$ of size $m$, the gradient discrepancy at $W$ is

$$
G_W(I) \;=\; \Big\| \frac{1}{m}\sum_{i \in I} g_i(W) \;-\; \bar g(W) \Big\|^2, \qquad g_i(W) = \nabla_W \ell(W; Z_i), \quad \bar g(W) = \frac{1}{n}\sum_{i=1}^{n} g_i(W), \qquad\qquad (1)
$$

which is the single-iterate discrepancy of Remark 4.3 in the manuscript. Every gradient-based method in this note selects at one of two points: initialization, $W_0 := 0$, and $W_1$, the model after one epoch of SGD on $S$ from $W_0$ (Nesterov momentum 0.9, learning rate 0.01, batch size 128, fixed shuffle). $W_1$ is computed once for each $\lambda$ and shared by all methods and seeds. The full-data model $W^\star$ is used only for evaluation, never for selection.

**Evaluation protocol.** Each method selects a subset at five retention fractions of the pool: 5%, 7%, 10%, 12% and 15%, that is $m \in \{2{,}500,\ 3{,}500,\ 5{,}000,\ 6{,}000,\ 7{,}500\}$, or 25 to 75 images per class. Every subset is used to train $\mathcal{A}$ for three weight decays, $\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}\}$. Seed-dependent methods use seeds 0, 1 and 2. We report three quantities:

- the test accuracy of $\mathcal{A}(S_I)$;
- its *agreement* with $\mathcal{A}(S)$, the fraction of test images on which the two models predict the same class, regardless of correctness;
- the discrepancies $G_{W_0}(I)$ and $G_{W_1}(I)$, and, for evaluation only, $G_{W^\star}(I)$.

Agreement measures faithfulness to full-data training directly. Table 1 gives the reference accuracies.

**Table 1:** Test accuracy (%) of the probe at the full-data solution $W^\star$ and at the two selection points.

| $\lambda$ | Full-data probe $W^\star$ | $W_1$ (one SGD epoch) | $W_0 = 0$ |
|:---:|:---:|:---:|:---:|
| $10^{-4}$ | 67.03 | 61.98 | 1.00 |
| $10^{-3}$ | 69.27 | 61.94 | 1.00 |
| $10^{-2}$ | 66.22 | 61.24 | 1.00 |

**Compute.** All runs used two AWS g5.xlarge instances (one NVIDIA A10G GPU and 4 vCPUs each), with three tasks per instance at a time. The full grid took about three hours of wall-clock time.

## 2 Methods

**Baselines.** *Random* draws $m$ indices uniformly from $[n]$. *Stratified random* draws uniformly within each class, with per-class quotas proportional to the pool's class counts; these quotas are the ones used by every class-balanced method below. *GraphCut* (Iyer et al., 2021) runs per class at the same quotas and maximises the graph-cut function with the greedy algorithm of apricot ($\lambda_{GC} = 1$). This rewards images that are similar to the rest of their class and penalises redundancy among the chosen ones. To give GraphCut the same information as the swaps, we build its per-class similarity from the gradient kernel at the selection point, $K_{ij}(W) = \langle g_i(W), g_j(W) \rangle$, and run it at $W_0$ and at $W_1$. GraphCut is deterministic, so each configuration yields a single subset.

**Second-Order Swap (SO-Swap).** Write $\zeta_i = g_i(W) - \bar g(W)$ for the centred gradients at the selection point, $C_{ij} = \langle \zeta_i, \zeta_j \rangle$, and $s = \frac{2}{m^2} C \mathbf{1}_I$. Exchanging $i \in I$ for $j \notin I$ changes the objective by exactly

$$
G_W(I') - G_W(I) \;=\; \underbrace{-s_i + \frac{C_{ii}}{m^2}}_{d_i} \;+\; \underbrace{s_j + \frac{C_{jj}}{m^2}}_{a_j} \;-\; \frac{2C_{ij}}{m^2}, \qquad\qquad (2)
$$

which is Equation (9) of the manuscript with the diagonal terms grouped by example. Algorithm 1 of the manuscript ranks exchanges by the first-order term $s_j - s_i$ alone. That term grows with $\|\zeta\|$, so the top-ranked pair consists of high-norm examples whose curvature term outweighs the first-order gain, and the search stops at its first rejection. SO-Swap instead uses the exact single-example effects:

1. compute $d_i$ for every $i \in I$ and $a_j$ for every $j \notin I$;
2. shortlist the $k = 16$ smallest $d_i$ and the $k = 16$ smallest $a_j$;
3. evaluate (2) exactly, cross term included, for all $k \times k$ pairs;
4. apply the best exchange if it decreases $G_W$ by more than $10^{-12}$, and otherwise stop.

As in Theorem 4.6(i), every accepted exchange strictly decreases $G_W$, so the search terminates. SO-Swap starts from the Random subset of the same seed, and the search is capped at 5,000 exchanges.

**Class-Balanced Swap (CB-Swap).** CB-Swap minimizes the same objective, starting from the Stratified random subset of the same seed, and only exchanges examples within a class, so every class keeps its quota. At every step each class shortlists $k_c = 16$ removals and $k_c = 16$ additions among its own examples, and the best exchange over all classes is applied.

**Exact gradients for the probe.** The per-example gradient factors as $g_i(W) = r_i(W)\,\phi_i^\top$, with $r_i(W) = \text{softmax}(W\phi_i) - e_{y_i} \in \mathbb{R}^{100}$, so that $\langle g_i, g_j \rangle = \langle r_i, r_j \rangle \langle \phi_i, \phi_j \rangle$. All quantities in (2) are therefore computed from $r$ and $\phi$, and the 51,300-dimensional gradients are never formed. Unlike the CIFAR experiments of the manuscript, the swaps here use exact gradients, with no count sketch.

**Remark 2.1** (GraphCut at initialization). At $W_0$ every image of class $c$ has the same residual $r_i = \frac{1}{100}\mathbf{1} - e_c$. Within a class, the gradient kernel is therefore a constant multiple of the feature kernel $\langle \phi_i, \phi_j \rangle$, and GraphCut($W_0$) coincides with the standard feature-based GraphCut. We verified this numerically.

## 3 Results

Tables 2 and 3 report test accuracy, Table 4 agreement with the full-data model, Table 5 the discrepancies at the two selection points, and Table 6 the correlation between the discrepancy at $W^\star$ and accuracy.

**Every swap variant improves on its random starting subset.** Table 3 compares each swap with the subset it started from, paired by seed: SO-Swap with Random and CB-Swap with Stratified random. The swap wins in all 180 comparisons (4 variants × 15 cells × 3 seeds). The gain is between +3.7 and +5.9 points at 5% retention and between +1.9 and +3.1 points at 15%. The standard deviation across seeds is between 0.02 and 0.70 points (Table 2), so the improvement is several times the seed-to-seed variation. The gain decreases with the retention fraction; for CB-Swap($W_0$) at $\lambda = 10^{-3}$ it is +5.6, +4.4, +3.8, +3.3 and +3.0 points from 5% to 15%. This is the behaviour expected if the swaps remove sampling noise, since the expected discrepancy of a uniform subset decays as $\frac{n-m}{m(n-1)}$ (Theorem 4.6(ii)).

**Table 2:** Test accuracy (%) of the probe trained on the selected subset, as a function of the retention fraction. Seed-dependent methods report the mean ± std over three seeds. The best method in each column is in bold.

*(a) $\lambda = 10^{-4}$ (full-data probe: 67.03%)*

| Method | 5% | 7% | 10% | 12% | 15% |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random | 53.50 ± 0.51 | 55.58 ± 0.64 | 57.55 ± 0.29 | 58.02 ± 0.12 | 59.13 ± 0.35 |
| Stratified random | 54.15 ± 0.20 | 55.87 ± 0.18 | 57.60 ± 0.41 | 58.69 ± 0.26 | 59.44 ± 0.45 |
| GraphCut ($W_0$) | 52.91 | 54.11 | 55.81 | 56.68 | 57.70 |
| GraphCut ($W_1$) | 57.15 | 57.99 | 59.53 | 59.80 | 60.66 |
| SO-Swap ($W_0$) | 57.24 ± 0.17 | 58.64 ± 0.02 | 59.65 ± 0.43 | 60.19 ± 0.14 | 61.01 ± 0.20 |
| SO-Swap ($W_1$) | 58.19 ± 0.12 | 59.63 ± 0.12 | 60.62 ± 0.21 | 61.03 ± 0.40 | 61.54 ± 0.11 |
| CB-Swap ($W_0$) | **59.42 ± 0.31** | **60.32 ± 0.11** | **61.39 ± 0.05** | **61.74 ± 0.17** | 62.21 ± 0.39 |
| CB-Swap ($W_1$) | 59.12 ± 0.22 | 59.91 ± 0.39 | 61.18 ± 0.07 | 61.73 ± 0.50 | **62.27 ± 0.18** |

*(b) $\lambda = 10^{-3}$ (full-data probe: 69.27%)*

| Method | 5% | 7% | 10% | 12% | 15% |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random | 54.03 ± 0.59 | 56.25 ± 0.67 | 58.57 ± 0.26 | 59.31 ± 0.19 | 60.75 ± 0.13 |
| Stratified random | 54.71 ± 0.13 | 56.74 ± 0.02 | 58.69 ± 0.37 | 59.84 ± 0.25 | 60.91 ± 0.34 |
| GraphCut ($W_0$) | 53.27 | 54.58 | 56.17 | 57.28 | 58.28 |
| GraphCut ($W_1$) | 56.71 | 57.72 | 59.31 | 59.69 | 60.37 |
| SO-Swap ($W_0$) | 57.96 ± 0.25 | 59.51 ± 0.12 | 60.86 ± 0.15 | 61.76 ± 0.15 | 62.91 ± 0.42 |
| SO-Swap ($W_1$) | 59.05 ± 0.70 | 60.58 ± 0.17 | 61.92 ± 0.17 | 62.46 ± 0.55 | 63.31 ± 0.18 |
| CB-Swap ($W_0$) | **60.29 ± 0.29** | **61.18 ± 0.14** | **62.49 ± 0.08** | **63.15 ± 0.22** | 63.88 ± 0.30 |
| CB-Swap ($W_1$) | 59.85 ± 0.30 | 60.77 ± 0.28 | 62.33 ± 0.07 | 62.96 ± 0.31 | **63.98 ± 0.30** |

*(c) $\lambda = 10^{-2}$ (full-data probe: 66.22%)*

| Method | 5% | 7% | 10% | 12% | 15% |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random | 54.95 ± 0.56 | 57.67 ± 0.66 | 60.12 ± 0.27 | 60.90 ± 0.42 | 62.00 ± 0.36 |
| Stratified random | 55.74 ± 0.37 | 57.98 ± 0.17 | 60.24 ± 0.26 | 61.11 ± 0.17 | 62.02 ± 0.29 |
| GraphCut ($W_0$) | 53.40 | 54.87 | 56.32 | 56.84 | 58.16 |
| GraphCut ($W_1$) | 56.01 | 56.91 | 58.05 | 58.45 | 59.18 |
| SO-Swap ($W_0$) | 59.42 ± 0.41 | 61.51 ± 0.29 | 62.78 ± 0.05 | 63.39 ± 0.22 | 64.06 ± 0.27 |
| SO-Swap ($W_1$) | 60.32 ± 0.32 | 61.76 ± 0.21 | 63.17 ± 0.33 | 63.94 ± 0.09 | 64.53 ± 0.13 |
| CB-Swap ($W_0$) | **61.59 ± 0.17** | **62.60 ± 0.12** | **63.59 ± 0.13** | 64.12 ± 0.31 | **64.73 ± 0.26** |
| CB-Swap ($W_1$) | 60.90 ± 0.18 | 62.48 ± 0.21 | 63.42 ± 0.14 | **64.13 ± 0.14** | 64.67 ± 0.10 |

**Table 3:** Paired gain in test accuracy (points) of each swap over its starting subset at the same seed (SO-Swap over Random, CB-Swap over Stratified random), averaged over three seeds. Every individual comparison is positive.

| $\lambda$ | Method | 5% | 7% | 10% | 12% | 15% |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| $10^{-4}$ | SO-Swap ($W_0$) | +3.7 | +3.1 | +2.1 | +2.2 | +1.9 |
|  | SO-Swap ($W_1$) | +4.7 | +4.1 | +3.1 | +3.0 | +2.4 |
|  | CB-Swap ($W_0$) | +5.3 | +4.4 | +3.8 | +3.0 | +2.8 |
|  | CB-Swap ($W_1$) | +5.0 | +4.0 | +3.6 | +3.0 | +2.8 |
| $10^{-3}$ | SO-Swap ($W_0$) | +3.9 | +3.3 | +2.3 | +2.5 | +2.2 |
|  | SO-Swap ($W_1$) | +5.0 | +4.3 | +3.4 | +3.2 | +2.6 |
|  | CB-Swap ($W_0$) | +5.6 | +4.4 | +3.8 | +3.3 | +3.0 |
|  | CB-Swap ($W_1$) | +5.1 | +4.0 | +3.6 | +3.1 | +3.1 |
| $10^{-2}$ | SO-Swap ($W_0$) | +4.5 | +3.8 | +2.7 | +2.5 | +2.1 |
|  | SO-Swap ($W_1$) | +5.4 | +4.1 | +3.1 | +3.0 | +2.5 |
|  | CB-Swap ($W_0$) | +5.9 | +4.6 | +3.3 | +3.0 | +2.7 |
|  | CB-Swap ($W_1$) | +5.2 | +4.5 | +3.2 | +3.0 | +2.7 |

**CB-Swap is the strongest selector in every configuration.** CB-Swap is the most accurate method in all 15 cells of Table 2 and the most faithful in all 15 cells of Table 4. The selection point matters little for it: CB-Swap($W_0$) is best in 12 cells and CB-Swap($W_1$) in 3, and the two differ by at most 0.7 points.

**Matched subsets are markedly more faithful to full-data training.** The gains in agreement exceed the gains in accuracy (Table 4). Relative to Stratified random, CB-Swap($W_0$) raises agreement by 4.3–6.9 points at $\lambda = 10^{-4}$, by 6.1–8.5 points at $\lambda = 10^{-3}$ and by 8.3–12.0 points at $\lambda = 10^{-2}$. In accuracy, the best CB-Swap closes 38–44% of the gap between Random and the full-data probe for $\lambda \le 10^{-3}$, and 59–65% for $\lambda = 10^{-2}$, at both 5% and 15% retention.

**Table 4:** Agreement (%) between the probe trained on the selected subset and the full-data probe $W^\star$: the fraction of test images on which they predict the same class. Seed-dependent methods report the mean over three seeds. The best method in each column is in bold.

*(a) $\lambda = 10^{-4}$*

| Method | 5% | 7% | 10% | 12% | 15% |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random | 58.0 | 60.7 | 63.5 | 64.5 | 66.0 |
| Stratified random | 58.6 | 61.3 | 63.9 | 65.6 | 66.9 |
| GraphCut ($W_0$) | 56.9 | 58.5 | 60.7 | 62.3 | 64.1 |
| GraphCut ($W_1$) | 61.8 | 62.9 | 64.3 | 65.2 | 66.0 |
| SO-Swap ($W_0$) | 62.6 | 64.8 | 66.6 | 67.7 | 69.1 |
| SO-Swap ($W_1$) | 63.9 | 66.0 | 67.8 | 69.1 | 70.1 |
| CB-Swap ($W_0$) | **65.5** | **67.4** | 69.1 | **69.9** | **71.2** |
| CB-Swap ($W_1$) | 65.1 | 66.7 | **69.2** | 69.9 | 71.2 |

*(b) $\lambda = 10^{-3}$*

| Method | 5% | 7% | 10% | 12% | 15% |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random | 62.1 | 65.4 | 68.9 | 70.1 | 72.3 |
| Stratified random | 63.1 | 65.9 | 69.3 | 70.9 | 72.9 |
| GraphCut ($W_0$) | 61.4 | 63.2 | 65.9 | 67.5 | 69.3 |
| GraphCut ($W_1$) | 66.9 | 68.3 | 70.1 | 71.3 | 72.4 |
| SO-Swap ($W_0$) | 68.1 | 70.5 | 73.1 | 74.4 | 76.1 |
| SO-Swap ($W_1$) | 69.9 | 72.3 | 74.5 | 75.6 | 77.4 |
| CB-Swap ($W_0$) | **71.7** | **73.6** | **75.9** | **77.3** | **78.9** |
| CB-Swap ($W_1$) | 71.4 | 73.0 | 75.7 | 76.9 | 78.5 |

*(c) $\lambda = 10^{-2}$*

| Method | 5% | 7% | 10% | 12% | 15% |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random | 66.8 | 71.4 | 75.8 | 77.6 | 80.0 |
| Stratified random | 68.6 | 72.4 | 76.6 | 78.4 | 80.7 |
| GraphCut ($W_0$) | 67.0 | 69.8 | 72.5 | 73.8 | 76.0 |
| GraphCut ($W_1$) | 73.3 | 75.1 | 77.2 | 78.1 | 79.6 |
| SO-Swap ($W_0$) | 75.3 | 79.0 | 82.5 | 84.2 | 85.8 |
| SO-Swap ($W_1$) | 77.0 | 80.6 | 83.7 | 85.8 | 87.6 |
| CB-Swap ($W_0$) | **80.6** | **83.2** | **86.1** | **87.5** | **89.0** |
| CB-Swap ($W_1$) | 79.3 | 82.5 | 85.8 | 87.1 | 88.9 |

**The class-mean component of the gradient drives the ranking.** Averaged over a subset, row $c$ of the gradient at $W$ splits into two parts:

$$
\frac{1}{m}\sum_{i\in I} g_i(W)_{c} \;=\; \underbrace{\frac{1}{m}\sum_{i \in I} p_{ic}(W)\,\phi_i^\top}_{\text{depends on } W} \;-\; \underbrace{\frac{m_c}{m}\,\bar\phi_{I,c}^\top}_{\text{independent of } W},
$$

where $p_{ic}(W)$ is the predicted probability of class $c$, $m_c$ the number of class-$c$ examples in $I$ and $\bar\phi_{I,c}$ their mean feature. At $W_0$ the predictions are uniform, so $G_{W_0}$ essentially measures the second part: the class proportions and class-mean features of the subset. That part does not depend on $W$, so a subset that matches it at $W_0$ keeps matching it along the whole fit of $\mathcal{A}$.

Table 5 is consistent with this reading:

- CB-Swap($W_0$) has the smallest $G_{W_0}$ (0.019 at 5%, 0.004 at 15%) and the best accuracy.
- SO-Swap($W_1$) attains a small $G_{W_1}$ (0.021 at 5%), but its $G_{W_0}$ returns to 0.205, close to the 0.301 of Random, and it is less accurate than both CB-Swaps.
- CB-Swap($W_1$) attains the smallest $G_{W_1}$ while keeping $G_{W_0}$ small (0.047 at 5%), because its class proportions are fixed.

**Selecting at $W_1$ helps only the unconstrained swap.** Moving the selection point from $W_0$ to $W_1$ raises the accuracy of SO-Swap by 0.3–1.1 points, but changes that of CB-Swap by −0.7 to +0.1 points. One epoch of training adds information about which examples the model gets wrong. For CB-Swap, which already matches the class-mean component through its quotas, that information does not translate into accuracy.

**Table 5:** Gradient discrepancy (1) of the selected subsets at the two selection points, for $\lambda = 10^{-3}$. Selections at $W_0$ do not depend on $\lambda$. Those at $W_1$ depend on it only through $W_1$, and the other weight decays give nearly identical values. Seed-dependent methods report the mean over three seeds.

*(a) $G_{W_0}(I)$*

| Method | 5% | 7% | 10% | 12% | 15% |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random | 0.3013 | 0.2083 | 0.1304 | 0.1087 | 0.0868 |
| Stratified random | 0.0748 | 0.0522 | 0.0354 | 0.0291 | 0.0224 |
| GraphCut ($W_0$) | 0.0910 | 0.0727 | 0.0573 | 0.0504 | 0.0413 |
| GraphCut ($W_1$) | 0.0948 | 0.0767 | 0.0600 | 0.0512 | 0.0426 |
| SO-Swap ($W_0$) | 0.0380 | 0.0249 | 0.0154 | 0.0121 | 0.0090 |
| SO-Swap ($W_1$) | 0.2050 | 0.1376 | 0.0943 | 0.0757 | 0.0610 |
| CB-Swap ($W_0$) | 0.0193 | 0.0123 | 0.0074 | 0.0058 | 0.0042 |
| CB-Swap ($W_1$) | 0.0468 | 0.0320 | 0.0209 | 0.0170 | 0.0133 |

*(b) $G_{W_1}(I)$*

| Method | 5% | 7% | 10% | 12% | 15% |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random | 0.1464 | 0.0964 | 0.0583 | 0.0502 | 0.0386 |
| Stratified random | 0.0619 | 0.0431 | 0.0306 | 0.0253 | 0.0196 |
| GraphCut ($W_0$) | 0.1273 | 0.1119 | 0.0967 | 0.0898 | 0.0777 |
| GraphCut ($W_1$) | 0.0716 | 0.0651 | 0.0608 | 0.0579 | 0.0530 |
| SO-Swap ($W_0$) | 0.0405 | 0.0272 | 0.0167 | 0.0131 | 0.0099 |
| SO-Swap ($W_1$) | 0.0207 | 0.0131 | 0.0083 | 0.0064 | 0.0047 |
| CB-Swap ($W_0$) | 0.0310 | 0.0223 | 0.0135 | 0.0102 | 0.0074 |
| CB-Swap ($W_1$) | 0.0165 | 0.0108 | 0.0069 | 0.0054 | 0.0040 |

**The discrepancy at the full-data solution tracks the accuracy gap across methods.** We also evaluate every selected subset at $W^\star$, the point at which the convexity bound of Section 5 holds, and correlate $\sqrt{G_{W^\star}(I)/2}$ with the accuracy gap $\text{acc}(\mathcal{A}(S)) - \text{acc}(\mathcal{A}(S_I))$, the coordinates of Figure 1 of the manuscript. Table 6 reports the Pearson correlation over the 20 subsets of each cell. It lies between +0.84 and +0.95, with median +0.90, and it does not rely on the Random subsets: without them, the median is +0.92 (range +0.77 to +0.98). Over the eight method means of each cell, the median is +0.92 (range +0.79 to +0.97). The discrepancy at $W_1$ correlates as strongly (median +0.90, range +0.85 to +0.94), and the one at $W_0$ less (median +0.53, range +0.48 to +0.60). $G_{W_0}$ measures only the class-mean component of the gradient, so it does not register the improvement that SO-Swap($W_1$) makes in the component that depends on the predictions (Table 5(a)). No method selects with $W^\star$; it is used here only to evaluate the subsets.

**Table 6:** Pearson correlation between $\sqrt{G_{W^\star}(I)/2}$ and the accuracy gap to the full-data probe (points), over the 20 subsets of each cell: Random, Stratified random, SO-Swap and CB-Swap at both selection points (three seeds each), and GraphCut at both selection points (one subset each).

| $\lambda$ | 5% | 7% | 10% | 12% | 15% |
|:---|:---:|:---:|:---:|:---:|:---:|
| $10^{-4}$ | +0.84 | +0.88 | +0.92 | +0.92 | +0.89 |
| $10^{-3}$ | +0.87 | +0.91 | +0.94 | +0.93 | +0.95 |
| $10^{-2}$ | +0.87 | +0.87 | +0.90 | +0.90 | +0.91 |

**Class balance alone does not explain the gains.** Stratified random is only 0.0–0.8 points more accurate than Random (Table 2), while CB-Swap adds a further 2.7–5.9 points on top of Stratified random (Table 3).

**GraphCut does not exploit the discrepancy.** GraphCut($W_0$) is less accurate than Random in all 15 cells. GraphCut($W_1$) is more accurate than Random in 10 of the 15 cells: every retention fraction at $\lambda = 10^{-4}$, all but 15% at $\lambda = 10^{-3}$, and only 5% at $\lambda = 10^{-2}$. Its $G_{W_1}$ is larger than that of Stratified random in every cell. GraphCut favours central, non-redundant examples, which is not the same as matching the subset's mean gradient to that of the pool.

## 4 Computational cost

**Wall-time.** Table 7 reports the selection time of each run: the swap search, or GraphCut's greedy maximisation. It excludes the steps shared by all methods: feature extraction (about two minutes per instance), the computation of the residuals at the selection point, the fit of $W^\star$ and the one SGD epoch for $W_1$, each of which takes seconds.

SO-Swap takes 2.8–5.6 s at $W_0$ and 5.0–12.1 s at $W_1$. Each step costs one product between the pool and the current aggregate, plus one $16 \times 16$ block. CB-Swap takes 5.7–10.8 min at $W_0$ and 4.7–8.9 min at $W_1$, which is 45–120 times slower. The difference comes from our implementation, which loops over the 100 classes in Python at every step and scores a separate $16 \times 16$ block per class. The step cost is therefore dominated by per-class overhead, not arithmetic. Times were measured while three tasks shared a 4-vCPU instance, which explains part of their spread.

**Table 7:** Selection wall-time per run, in the unit given with each method: mean over nine runs (three weight decays × three seeds), with the range in parentheses. GraphCut is over three runs.

| Method, unit | 5% | 7% | 10% | 12% | 15% |
|:---|:---:|:---:|:---:|:---:|:---:|
| SO-Swap ($W_0$), s | 2.8 (2.2–4.0) | 3.3 (2.9–3.8) | 4.4 (4.0–4.7) | 4.9 (3.6–5.7) | 5.6 (4.2–6.1) |
| SO-Swap ($W_1$), s | 5.0 (4.5–5.7) | 5.9 (4.9–7.7) | 7.9 (6.4–9.7) | 10.4 (8.3–11.9) | 12.1 (10.0–14.0) |
| CB-Swap ($W_0$), min | 5.7 (5.6–5.8) | 6.8 (6.3–7.1) | 8.1 (5.8–8.8) | 9.7 (9.5–10.0) | 10.8 (10.4–11.4) |
| CB-Swap ($W_1$), min | 4.7 (4.5–5.0) | 5.3 (4.0–6.3) | 6.4 (3.0–8.0) | 8.3 (5.7–9.3) | 8.9 (3.8–10.5) |
| GraphCut ($W_0$), s | 13.3 (13.0–13.7) | 14.8 (14.2–15.8) | 15.6 (14.8–16.9) | 15.5 (13.8–17.4) | 16.1 (15.3–17.2) |
| GraphCut ($W_1$), s | 10.2 (9.9–10.5) | 10.2 (9.3–10.8) | 11.6 (10.4–12.3) | 13.7 (12.6–14.5) | 14.6 (13.7–15.2) |

**Number of swaps.** Table 8 reports the number of accepted exchanges. All 180 swap runs stopped because no shortlisted exchange decreased $G_W$ any further; none reached the cap of 5,000. The number of exchanges grows with $m$ but more slowly than $m$, between 0.11 and 0.52 exchanges per selected example. CB-Swap performs 3.0–3.3 times as many exchanges as SO-Swap at $W_0$, and 1.3–1.5 times as many at $W_1$: searching every class at each step keeps it finding improving exchanges for longer. At $W_1$, SO-Swap performs fewer exchanges for larger $\lambda$; at 15% it makes 1,892, 1,857 and 1,596 exchanges for $\lambda = 10^{-4}$, $10^{-3}$ and $10^{-2}$. Since an example can leave and later re-enter the subset, these counts are not the number of examples replaced. We did not record that number.

**Table 8:** Accepted exchanges per run: mean over nine runs, with the range in parentheses.

| Method | 5% | 7% | 10% | 12% | 15% |
|:---|:---:|:---:|:---:|:---:|:---:|
| SO-Swap ($W_0$) | 393 (380–404) | 483 (452–521) | 636 (629–645) | 715 (659–759) | 822 (789–848) |
| SO-Swap ($W_1$) | 739 (651–810) | 989 (863–1065) | 1299 (1157–1387) | 1507 (1299–1635) | 1782 (1572–1933) |
| CB-Swap ($W_0$) | 1294 (1278–1303) | 1575 (1535–1628) | 2013 (1997–2029) | 2210 (2197–2218) | 2487 (2434–2554) |
| CB-Swap ($W_1$) | 1078 (1037–1114) | 1385 (1254–1437) | 1787 (1718–1845) | 2033 (1928–2143) | 2338 (2236–2399) |

## 5 Discussion and Limitations

On a linear probe over frozen features, minimizing the gradient discrepancy at a cheap selection point improves substantially on random selection. Both swap variants beat their random starting subsets in every configuration, with gains of 2–6 points in accuracy and up to 12 points in agreement with the full-data model, and CB-Swap is the strongest selector throughout. These gains are much larger than those of the CIFAR experiments of the manuscript, where models are trained from scratch. Three properties of the probe explain the difference:

- *the features are fixed and the fit is deterministic*, so the trained model is a function of the selected subset alone;
- *the per-class budget is small* (25–75 images), so the sampling noise of a random subset is large;
- *the component of the gradient determined by the class proportions and class-mean features is independent of $W$*, so matching it at $W_0$ controls the gradient at every point the fit visits.

We do not read these results as evidence that the swaps are the best selectors in general. The setting is a convex problem on fixed features. Our convexity bound, $\|\mathcal{A}(S_I) - W^\star\| \le \sqrt{G_{W^\star}(I)}/\lambda$, holds only for the discrepancy at $W^\star$, so the gains reported here at $W_0$ and $W_1$ are empirical rather than guaranteed. Across methods, the discrepancy at $W^\star$ is strongly correlated with the accuracy gap (Table 6), but this relation is also empirical: the bound concerns the distance in parameter space, not accuracy. The results also come from a single dataset and backbone, with three seeds per configuration. The paired gains are consistent across all 180 comparisons, but each standard deviation is estimated from three runs. Finally, the wall-times of CB-Swap reflect a per-class loop in our implementation rather than the cost of the method itself, and were measured under CPU contention.

**Implementation details.** The probe and both swaps are implemented in `src/data_selection_swap/selection/linear_probe.py`, which reuses the swap rule of `selection/second_order.py`. A single cell is run with

```
python scripts/experiments/run_config.py run-task \
    configs/revision/cifar100_linear_probe_w0_w1.yaml probe TASK --data-root DATA
```

Tasks 0–14 enumerate the five retention fractions for each of the three weight decays. Each cell writes one `results_f<fraction>_wd<λ>.json`, with one row per method and seed (accuracy, agreement, distance to $W^\star$, discrepancies at $W_0$, $W_1$ and $W^\star$, selection time, number of exchanges and stopping reason), together with the indices of every selected subset.
