## Node data model

Conceptually, your tree becomes a **dendrogram with lifecycle states**. Here’s a pragmatic model that supports latent children + exemplar/aging prototypes.

### Node fields
- **Identity / structure**
  - `node_id`
  - `parent_id`
  - `children_active: list[node_id]`
  - `children_latent: list[node_id]` (collapsed but retained)
  - `depth`

- **Lifecycle**
  - `state ∈ {PROVISIONAL, SUPPORTED, STRUCTURAL, DORMANT}`
  - `state_since` (timestamp / step index)
  - `collapse_reason` (optional string/enum)
  - `active_mass` vs `dormant_mass` (or track all with flags)

- **Evidence accumulation**
  - `mass m` (effective count with aging/decay)
  - `last_update_t`
  - `age a` (derived)
  - `num_routed_here`
  - `num_forced_here` (if any backoff routing)

- **Prototypes**
  - `prototype p` (aging prototype)
  - `prototype_cov` or `sim_var` (optional, for gain normalization)
  - `exemplars E` (small fixed budget, with replacement policy)

- **Routing statistics**
  - `sim_sum S` and `sim_sq_sum` for cohesion estimate
  - `routing_conf_margin_hist` (optional sketch; or rolling stats)
  - `uncertainty` estimate (entropy of soft routing distribution)

- **Lifecycle gate parameters (optional cached)**
  - `provisional_thresholds`
  - `split_candidate_score`
  - `collapse_candidate_score`

- **Soft collapse / reactivation support**
  - `collapsed_children_meta`: per child
    - `last_active_t`
    - `reactivation_score_running`
    - `decayed_mass`
    - `cooldown_until`

### Lifecycle transition sketch
- PROVISIONAL → SUPPORTED: enough consistent routing evidence + cohesion above baseline
- SUPPORTED → STRUCTURAL: cohesion gain consistently positive for \(k\) windows
- STRUCTURAL → DORMANT: children collectively don’t improve gain beyond collapse threshold
- DORMANT → SUPPORTED/STRUCTURAL: reactivation threshold met (low threshold)
- Splitting: only from STRUCTURAL or high-mass SUPPORTED nodes (your call), to avoid churn.

---
