# cae-rag-router
computational attention engine polymorphic scale free router



A second robustness term reduces presentation-order dependence: measure disagreement against node’s own prototype:
\[
\gamma = s(\hat{x},\hat{p}_u)
\]
and use it to modulate confidence:
\[
\textbf{conf}'(u,x)=\textbf{conf}(u,x)\cdot \sigma(\lambda(\gamma-\mu_u))
\]
where \(\sigma\) is sigmoid and \(\mu_u\) is a rolling average similarity at node \(u\).

### 2) Cohesion and cohesion gain (for splitting)
Define a node’s **internal cohesion** as an average similarity of assigned items to its prototype and/or exemplars:
\[
\text{coh}(u) = \mathbb{E}_{x\sim \mathcal{A}_u}\left[s(\hat{x},\hat{p}_u)\right]
\]
In streaming form, maintain running estimates:
- \(M_u\): effective mass (see data model)
- \(S_u\): accumulated similarity sum
Then:
\[
\text{coh}(u) \approx \frac{S_u}{M_u}.
\]

When you consider splitting \(u\), you need a candidate partition of its internal evidence into two groups (e.g., via a fast online axis method, or via exemplar-based partitioning). Suppose you form two child proto candidates \(p_{u_0}, p_{u_1}\) and estimated assigned subsets with proportions \(\pi_0,\pi_1\). Then define:
\[
\text{gain}(u) = \Big(\pi_0\,\text{coh}(u_0) + \pi_1\,\text{coh}(u_1)\Big) - \text{coh}(u).
\]

To prevent trivial splits, normalize by uncertainty/variance:
\[
\text{gain}_{\text{norm}}(u) = \frac{\text{gain}(u)}{\epsilon+\sqrt{\text{VarSim}(u)}}.
\]

### 3) Split threshold (depth-dependent, stability-first)
Let node age \(a_u\) (time since creation) and mass \(m_u\) drive threshold. A typical stable rule:
\[
\textbf{split}(u) \;\;\text{iff}\;\; 
\text{gain}_{\text{norm}}(u) \ge \theta_{\text{split}}(h,u)
\]
with:
\[
\theta_{\text{split}}(h,u)=\theta_{\text{split}}^{0}\; \cdot e^{-\beta a_u}\; \cdot e^{\eta h_{\text{top}}(h)}
\]
and/or enforce minimum evidence:
\[
m_u \ge m_{\min}(h),\quad a_u \ge a_{\min}.
\]
You said “prevent top-level instability”: make \(m_{\min}\) and/or \(\theta_{\text{split}}\) higher for small \(h\).

Also require routing consistency: splits should align with routing confidence statistics of items routed through \(u\). Use:
\[
\overline{\Delta_q}(u) \ge \theta_{\text{margin}}
\]
computed over items that arrived at \(u\).

### 4) Collapse thresholds (soft collapse, no identity deletion)
Collapse means “children don’t earn their keep,” but the subtree identity remains.

Define a child usefulness score:
\[
U_c(u)=\mathbb{E}_{x\sim \mathcal{A}_c}\left[\textbf{conf}(u,x)\right] \cdot \text{coh}(c)
\]
and a compare-to-parent baseline:
\[
R_c = U_c(u) - \text{coh}(u)
\]
Then a soft collapse happens when the children collectively fail:
\[
\sum_{c\in \text{active children}} \pi_c\,R_c \le \theta_{\text{collapse}}(h,u).
\]

Additionally: collapse should trigger when evidence is mostly “absorbed” by one child or by the parent prototype without improving overall cohesion. Equivalent:
\[
\textbf{gain}(u) \le \theta_{\text{collapse\_gain}}(h).
\]

Use hysteresis:
- collapse threshold higher for shallow depth (more stable),
- and require that the condition holds for a duration \(\Delta t\) or for \(N\) new arrivals.

### 5) Reactivation thresholds (lower than split)
Reactivation means: a collapsed child/subtree gets evidence again, so we let it regain “active” status (latent → structural).

If a subtree \(v\) is collapsed under parent \(u\), maintain a reactivation similarity stream:
\[
\rho_v(u,x)=s(\hat{x},\hat{p}_v)
\]
and optionally a confidence margin relative to sibling active ones.

Define:
\[
\overline{\rho}_v \ge \theta_{\text{react}}(h)
\]
but make \(\theta_{\text{react}}\) **lower** than \(\theta_{\text{split}}\) because reactivation does not require discovering a new partition—its identity already exists (prototypes/exemplars/subspace hints remain). In practice:
\[
\theta_{\text{react}}(h) \approx 0.5\,\theta_{\text{split}}(h)
\]
(or tune empirically), plus a minimum count:
\[
m_v \ge m_{\text{react\_min}}.
\]

Also avoid thrash: require reactivation score to exceed collapse score by margin:
\[
\overline{\rho}_v - \overline{\rho}_{\text{parent}} \ge \theta_{\text{react\_margin}}.
\]

---
