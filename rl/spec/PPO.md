# twisterl 0.5.1 BasicPolicy + PPO: exact specification for the Mojo port

Sources read (all as data, nothing from them was built or run except the pinned wheels from requirements.txt):

* `twisterl` 0.5.1 Python layer: `venv/.../site-packages/twisterl/{nn/policy.py,nn/utils.py,rl/algorithm.py,rl/ppo.py,rl/az.py,defaults.py,utils.py}`.
* `twisterl` 0.5.1 Rust crate: GitHub tag `0.5.1` of IBM/twisterl, `rust/Cargo.toml` says `version = "0.5.1"`, deps `nalgebra 0.33`, `rand 0.8`, `rayon 1.1`. Files: `rust/src/{nn/{layers,modules,policy}.rs, collector/{collector,ppo,az}.rs, rl/{solve,evaluate,search,tree,env}.rs, python_interface/*.rs}`. (PyPI has wheels only, no sdist; the compiled `.so` in the wheel was cross-checked numerically against this source, see "Golden data".)
* `qiskit_gym` 0.4.1 Python RL layer: `rl/configs.py` (PPOConfig, EvalConfig, BasicPolicyConfig), `rl/synthesis.py` (RLSynthesis), `envs/adapters.py`. Env logic itself (reset/step/reward/twists) is the "envs" agent's job; section 11 lists the contract PPO needs from it.
* QUMO driver: `training/ai_transpiler/synth/train.py` (the only user of twisterl PPO), `synth/export_model.py`, `common/export.py`. QUMO's own router PPO (`route/train_ppo.py`) is covered in section 12.

Everything below is verified: `ppo_golden/ref_ppo.py` is an independent numpy implementation of sections 2-7 that reproduces torch 2.14.1 / twisterl 0.5.1 (forward 1e-5, loss 1e-5, gradients 2e-6, weights after 10 Adam epochs 2e-6) and the compiled Rust collector (GAE, evaluate).

---------------------------------------------------------------------------------------------------

## 0. Pipeline in one paragraph

`synth/train.py` builds a qiskit-gym env (Rust), a `BasicPolicy` (torch), a `PPO` object (`twisterl.rl.PPO`), then loops `algo.learn_step()` until a wall-clock budget expires. One `learn_step` = (a) copy the torch weights into an immutable Rust policy, (b) evaluate that policy at the current difficulty (2 evals, 200 episodes each), (c) collect `num_episodes` (1024) full on-policy episodes in parallel Rust threads, (d) convert to tensors, (e) run 10 full-batch Adam steps on the PPO loss. After each `learn_step` the driver raises the env difficulty by 1 if the deterministic-eval success was >= 0.85. Checkpoint = torch `state_dict` as safetensors (f32). There is no LR schedule, no minibatching, no gradient clipping, no weight decay, no target network, no replay.

## 1. Constants actually in effect for QUMO (`train.py` + `make_config`)

`PPOConfig(num_cores=a.cores, num_episodes=a.episodes, evals=evals, diff_max=a.diff_max)`; everything else is the dataclass default. `RLSynthesis.init_algorithm` passes `rl_config.to_json()` as the algorithm config and `make_config("PPO", cfg)` merges it over `twisterl.defaults` (the nested dicts `collecting/training/optimizer/learning/logging/evals` of the input replace the defaults wholesale; each eval dict is then overlaid on `BASE_EVAL_CONFIG`, adding `seed=0, max_expand_depth=1`).

| constant | value | where |
|---|---|---|
| num_cores (rollouts, evals, torch threads) | `--cores`, default 8, hard cap 8 | train.py |
| episodes per iteration | `--episodes`, default 1024 | train.py |
| gamma | 0.995 | PPOConfig |
| lambda (gae_lambda) | 0.995 | PPOConfig |
| epochs per iteration | 10, all full batch (batch = every recorded state of the iteration) | PPOConfig |
| clip_ratio | 0.1 | PPOConfig |
| vf_coef | 0.8 | PPOConfig |
| ent_coef | 0.01 | PPOConfig |
| normalize_advantage | False | PPOConfig |
| optimizer | torch.optim.Adam, lr 3e-4, betas (0.9, 0.999), eps 1e-8, weight_decay 0, amsgrad False | PPOConfig + torch default |
| LR schedule | none (constant) | |
| grad clipping | none | |
| diff_threshold | 0.85 | PPOConfig |
| diff_metric | `"ppo_deterministic"` | PPOConfig |
| diff_max | `--diff-max`, default 256 | train.py |
| initial difficulty | 1 (or the sibling run's `final_difficulty` for a `__cont` run) | train.py |
| eval `ppo_deterministic` | num_episodes 200, deterministic (argmax), num_searches 1, num_mcts_searches 0, num_cores = cores | train.py |
| eval `ppo_10` | num_episodes 200, sampled (deterministic=False), num_searches 10, num_mcts_searches 0, num_cores = cores | train.py |
| BasicPolicy | `embedding_size` = `--emb` (default 512), `common_layers` = `[int(x) for x in --common.split(",")]` (default `[256]`), `policy_layers=[]`, `value_layers=[]` | train.py |
| device | cuda if available else cpu (Apple: cpu, torch float32) | defaults.make_config |
| env | `max_depth` 128, `depth_slope` 2, `add_inverts` True, `add_perms` True (qiskit-gym defaults, `from_coupling_map`) | envs/synthesis.py |
| evals' `C` / MCTS | 1.41 / unused (num_mcts_searches 0) | |

Only these two evals exist in QUMO's config (it replaces the default dict of 4). `ppo_10` is computed and logged but does not influence anything.

## 2. BasicPolicy architecture

`BasicPolicy(obs_shape, num_actions, embedding_size E, common_layers=(C1,...,Ck), policy_layers=(), value_layers=(), obs_perms, act_perms)`

```
OBS   = prod(obs_shape)                    # e.g. n*n (perm, linear function), 4n^2 (Clifford), 2n(2n+5) (Pauli)
A     = env.num_actions()
embeddings = Linear(OBS -> E)              # weight [E][OBS], bias [E]
common     = Sequential(Linear(E->C1), ReLU, Linear(C1->C2), ReLU, ...)   # empty if common_layers == []
action     = Sequential(Linear(Ck -> A))   # policy_layers=() => a single Linear, NO activation after it
value      = Sequential(Linear(Ck -> 1))
```

QUMO default: E=512, common=[256]: `embeddings.{weight[512][OBS],bias[512]}`, `common.0.{weight[256][512],bias[256]}`, `action.0.{weight[A][256],bias[A]}`, `value.0.{weight[1][256],bias[1]}`. Parameter count = `OBS*E + E + E*C + C + C*A + A + C + 1` (e.g. OBS=64,A=7: 166 664; OBS=36,A=10: 153 099).

State-dict key names and order of `state_dict()` (module registration order): `embeddings.weight, embeddings.bias, common.0.weight, common.0.bias, [common.2.weight, common.2.bias, ...], action.0.weight, action.0.bias, value.0.weight, value.0.bias`. (The `_obs_perm*`/`_act_perm*` buffers are `persistent=False`, not saved.) `common_layers=[256]` => indices `0` only (ReLU modules sit at odd indices); with two layers the second Linear is `common.2`.

### 2.1 Weight initialisation (PyTorch default, nothing custom)

Created in the order embeddings, common.*, action.0, value.0. For every `Linear(fan_in -> out)`:

```
bound = 1 / sqrt(fan_in)          # = kaiming_uniform(a=sqrt(5)): gain=sqrt(2/(1+5)), bound = gain*sqrt(3/fan_in)
weight[o][i] ~ U(-bound, bound)   # fan_in = in_features
bias[o]      ~ U(-bound, bound)   # same bound, fan_in of the weight
```
fan_in: embeddings = OBS (so the bound is 1/sqrt(OBS), e.g. 1/8 for OBS=64), common.k = previous width, action/value = last common width (256 -> 0.0625). No zero-init of heads, no orthogonal init. The RNG stream need not match torch (it cannot be bit-reproduced anyway; rollouts are unseeded in the original). `golden/policy_forward.json["init_check_seed1234_before_x2_scaling"]` records the bound check.

### 2.2 Forward pass (per sample; dense multi-hot x in {0,1}^OBS equals the sparse bag)

```
forward(active_indices S, twist p or none):
    if p: S = { obs_perm[p][v] for v in S }            # replace index v by obs_perm[p][v]
    h = bias0 + sum_{i in S} W0[:, i]                  # sparse embedding bag (torch: dense x @ W0^T + b0); duplicates would add twice
    h = relu(h)
    for (W, b) in common: h = relu(W h + b)
    raw   = Wa h + ba                                  # [A]
    value = Wv h + bv                                  # scalar (value.0 output [1])
    if p: logits[i] = raw[act_perm[p][i]]  for i in 0..A-1
    else: logits = raw
    return logits, value
```
Rust (`Policy::_raw_predict`) does exactly this with f32 `nalgebra` matrices; torch (`_forward_with_indices`) gathers the dense input with the inverse obs perm, `x_perm[j] = x[perm_inv[j]]`, which is the same thing as `x'[obs_perm[v]] = x[v]`, and then `logits.index_select(1, act_perm)`.

Masks: `Policy::forward_with_perm` returns `masked[i] = logits[i] if mask[i] else -1e10`. Probabilities for acting (`predict`): `e[i] = exp(masked-or-0)`: `e[i] = exp(logits[i])` if legal else 0 ; `p[i] = e[i] / (sum(e) + 1e-6)` (no max-subtraction, f32).

### 2.3 Symmetry (twist) augmentation

`obs_perms, act_perms = env.twists()` (lists of equal length T, each `obs_perms[t]` a permutation of `0..OBS-1`, `act_perms[t]` a permutation of `0..A-1`; both empty if `add_perms=False` or no symmetry). In QUMO's models these are the coupling-graph automorphism tables (see common/symmetry.py).

* Rollouts and evaluation: every single forward call (`forward_with_perm`, hence every collector step and every `predict` in `solve`) draws a fresh uniform `t ~ U{0..T-1}` from `rand::thread_rng()` and uses twist `t`. The drawn index is stored per recorded state in `data.perms` (-1 if no twists).
* Training: `policy(x, perm_indices=perms)` applies the recorded twist of each row (`>=0`) so the new logits correspond exactly to the stored behaviour logits. Row with `-1` uses no twist. If `perm_indices is None` torch draws random twists (never the case in PPO.train_step).
* `full_predict` (used only by MCTS): average raw logits (post act-perm, pre mask) and values over ALL twists, then mask / exp / normalise with `+1e-6`. QUMO's PPO never calls it (num_mcts_searches = 0).
* Inference note: twisterl's `predict` uses ONE random twist, whereas QUMO's Mojo inference uses the ensemble; that is existing behaviour and unchanged.

## 3. Observation to tensor

Env gives `observe() -> list[usize]`, the indices of the 1-bits of a flat binary vector of size OBS (row-major over `obs_shape`; qiskit-gym guarantees distinct indices). Training (`PPO.data_to_torch`): `np_obs[i, idx] = 1.0` for each recorded state (dense float32 `[N][OBS]` in the original; the Mojo port should keep the sparse index lists and use the embedding-bag forward/backward, see 7.4). Duplicated indices in one observation would count once in torch (assignment) but twice in Rust: irrelevant for the qiskit-gym envs.

## 4. Rollout collection (`PPOCollector`, Rust)

```
collect(env, rs_policy, num_episodes=1024, gamma, lam, num_cores):
    chunks = parallel_map(i in 0..num_episodes, single_collect)   # rayon pool of num_cores threads, rebuilt per call (num_cores==1: sequential)
    return merge(chunks)

single_collect(env, policy):
    env = env.clone(); env.reset()                 # reset() uses the env's CURRENT difficulty (cloned along with the env)
    loop:
        obs   = env.observe()
        mask  = env.masks()                        # qiskit-gym: [!success]*A  => all-False in a solved state
        r     = env.reward()                       # reward of the CURRENT state, read before acting
        (logits_masked, v, perm) = policy.forward_with_perm(obs, mask)   # fresh random twist each call
        a = argmax_i( logits_masked[i] - ln(|ln(u_i)|) ),  u_i ~ U(0,1) i.i.d.     # Gumbel-max == sample from softmax(logits_masked)
        record (obs, logits_masked, v, r, a, perm)
        if env.is_final(): break                   # the final state IS recorded; its action is never executed
        env.step(a)
    # GAE-lambda; n = number of recorded states
    advs[n-1] = r[n-1] - v[n-1];     rets[n-1] = r[n-1]
    for t = n-2 .. 0:
        rets[t] = r[t] + gamma * ( v[t+1] + lam * advs[t+1] )
        advs[t] = rets[t] - v[t]
    # NOTE: this is the twisterl form, not textbook. reward is a function of the state s_t (not a transition reward),
    # the bootstrap is v[t+1], the terminal state gets no bootstrap, and advs[t+1] enters multiplied by gamma*lam via rets.
    # Expanding: advs[t] = r_t + gamma*v_{t+1} - v_t + gamma*lam*advs[t+1]  (delta_t + gamma*lam*advs[t+1] with delta_t = r_t + gamma v_{t+1} - v_t).
```
* `values` are the value-head outputs of the (twisted) forward pass of the SAME rust policy snapshot; `rewards` are `env.reward()` (`value()` of the Python env protocol); there is no reward scaling and no discount other than `gamma` above. `truncation` (max depth reached, `depth == 0`) is treated as terminal (`is_final()`), no bootstrap.
* Stored `logits` are the MASKED logits (illegal = -1e10, f32). In a final solved state all are -1e10.
* `merge`: `chunks.pop()` takes the LAST episode as the base, then appends episodes `0..num_episodes-2` in order. Verified on the real library (last episode first, then 0,1,...). Order is irrelevant for a full-batch loss (only float-sum order changes).
* Randomness: `rand::thread_rng()` per thread, unseeded (not reproducible; QUMO's README says so). The port should use one seeded counter-based/xoshiro stream per episode so runs are reproducible for a given seed.
* Threading: episodes are independent rayon tasks; policy and env are shared read-only / cloned per episode; nothing is synchronised during collection. The Python GIL is held only for Python-implemented envs (qiskit-gym envs are pure Rust, no GIL). A fresh rayon pool is created for every `collect` and every `evaluate` call.

## 5. Tensor conversion for the update (`PPO.data_to_torch`)

```
obs_dense[i][idx] = 1.0 for idx in obs[i]
old_logits = stored masked logits (f32)                                   # N x A
acts, rets, advs, perms as stored                                         # perms: -1 where None
if normalize_advantage:  advs = (advs - mean(advs)) / (std_unbiased(advs) + 1e-8)     # OFF in QUMO; torch .std() uses N-1
old_logp[i] = log_softmax(old_logits[i])[acts[i]]                         # Categorical(logits=...).log_prob; normalisation by logsumexp
```
Note `values` and `rewards` of the batch are discarded (only `rets`, `advs`).

## 6. PPO loss and update (`PPO.train` / `train_step`)

```
train(torch_data):   # one call per learn_step
    repeat num_epochs (=10) times:  loss_dict = train_step(torch_data)       # SAME full batch every epoch, no shuffling
    return loss_dict of the LAST epoch (only used for logging)

train_step:
    logits, v = policy(obs_dense, perm_indices=perms)        # NEW logits: NOT masked (!), with the row's recorded twist
    logp_all  = log_softmax(logits);  p_all = exp(logp_all)
    lp        = logp_all[i, acts[i]]
    ratio     = exp(lp - old_logp)
    s1        = ratio * advs
    s2        = clamp(ratio, 1-0.1, 1+0.1) * advs
    policy_loss = -mean_i( min(s1, s2) )
    value_loss  = mean_i( (v[i] - rets[i])^2 )               # F.mse_loss(pred_vals[N,1], rets.unsqueeze(1)), mean over N
    entropy     = mean_i( -sum_a p_all[i,a] * logp_all[i,a] )
    loss = policy_loss + 0.8 * value_loss - 0.01 * entropy
    optimizer.zero_grad(); loss.backward(); optimizer.step()   # Adam, section 7.2
    return {value, policy, entropy, total}
```

Analytical gradients (what `ref_ppo.py` implements and the Mojo backward must match):

```
# w.r.t. new logits z_i (post-twist, unmasked), N rows
g_ratio_i = advs_i                                  if s1_i < s2_i            (unclipped branch active)
          = advs_i * [1-c <= ratio_i <= 1+c]        if s2_i < s1_i            (clipped branch active; zero outside the band)
          = advs_i                                  if s1_i == s2_i           (tie: torch splits grad/2 + grad/2, both equal -> total advs)
dL/dlp_i  = -(1/N) * g_ratio_i * ratio_i
dL/dz_i   = dL/dlp_i * (onehot(act_i) - p_all_i)                              # d log p / d z
          + (-0.01/N) * ( -p_all_i * (logp_all_i + H_i) )                     # entropy bonus, dH/dz_j = -p_j (logp_j + H)
dL/dv_i   = 0.8 * 2 * (v_i - rets_i) / N
# un-twist: raw[act_perm[a]] receives dL/dz[a]   (np.add.at), then back through action/value heads, ReLU masks (x>0), common, embedding bag.
```
Documented quirks that the port must keep (they change learning behaviour and the golden data reflects them):
1. **Masked old, unmasked new.** `old_logp` comes from masked logits, `lp` from unmasked logits. For rows with some illegal actions the ratio starts != 1; in a solved final state (all actions masked) old_logp = -ln(A) (uniform) and the action is a random one. In qiskit-gym envs a non-final state always has all actions legal (`masks = !success`), so only the final solved states are affected. The golden batch contains such rows. Do NOT "fix" this silently; if you improve it, do it behind a flag and document it.
2. Reward/advantage conventions of section 4 (state reward, terminal without bootstrap).
3. Entropy is the unmasked entropy of all `A` actions.
4. No advantage normalisation, no value clipping, no KL early stop, no minibatches, no shuffling, no gradient clipping.
5. Old-policy snapshot: the Rust policy is rebuilt from the torch weights at the START of each `learn_step` (`sync_rs_policy`) and used for the evals and the collection; the 10 epochs then move the torch weights. The stored behaviour logits therefore come from exactly the weights at the beginning of the iteration, in f32 Rust arithmetic (differences vs torch forward ~1e-8 verified).

## 7. Optimiser and numerics

### 7.1 Adam (torch.optim.Adam defaults, single param group, ONE optimiser for the whole run)
```
state per parameter: m=0, v=0, step=0   # created lazily, persists across all learn_steps and epochs (never reset)
each optimizer.step():
    t += 1
    m = 0.9*m + 0.1*g
    v = 0.999*v + 0.001*g*g
    bc1 = 1 - 0.9^t ; bc2 = 1 - 0.999^t
    denom = sqrt(v)/sqrt(bc2) + 1e-8
    w -= (lr/bc1) * m / denom            # lr = 3e-4
```
Parameters are stored and updated in float32 (state in float32 too; the numpy reference uses float64 intermediates and matches to 2e-6).
Total optimizer steps = 10 per iteration. **Use dense Adam on every parameter every step** (not lazy/sparse Adam): never-active embedding columns have g=0 so stay 0, but columns that were active in an earlier step keep moving through the momentum `m` even when their gradient is 0.

### 7.2 Precision
float32 everywhere (weights, activations, gradients, logits; -1e10 stored as f32). The golden tolerances (1e-5 forward/loss, 2e-6 gradients/weights) assume f32 accumulation in the port; accumulate dot products in f32 or higher (f64 is also fine against these tolerances).

### 7.3 Complexity, to size the Mojo kernels
N = number of recorded states per iteration = sum of episode lengths (episode length <= min(2*difficulty, 128) + 1). Typical 1024 episodes at difficulty ~100 => N ~ 1.3e5 rows. Per epoch: embedding bag forward/backward O(N * nnz * E) (nnz = obs bits, ~n for permutation, ~n^2/2 for matrices), dense layers O(N*(E*C + C*A)) forward and ~2x that backward (e.g. 512x256 => ~17 GFLOP fwd + ~35 GFLOP bwd per epoch at N=1.3e5), gradient of the embedding weight is a scatter-add of `dpre0[i]` into column `idx` for each active idx (dense `[E][OBS]` buffer), Adam is O(params).

## 8. Curriculum, evaluation, best-checkpoint logic (QUMO `train.py` + `Algorithm.learn_step`)

```
env.difficulty = start_diff                     # 1, or sibling run's summary.json final_difficulty for "<name>__cont" runs
best = None
while now - t0 < budget_s:                      # checked only between iterations
    (bench, train_loss) = learn_step()          # order inside: sync policy -> evals -> collect -> train (section 0)
    cur = (bench.difficulty, bench.success, bench.reward)      # tuple compare, lexicographic
    # bench.* are from the eval of the policy BEFORE this iteration's update, at the difficulty BEFORE the promotion below
    if best is None or cur >= best:
        best = cur
        save model.safetensors := policy weights AFTER this iteration's update      # note: not the weights that were evaluated
    if bench.success >= 0.85 and env.difficulty < diff_max:
        env.difficulty += 1                     # at most +1 per iteration, never demoted
    log.jsonl <- {it, t, step_s, difficulty, succ: {ppo_deterministic, ppo_10}, rew: {...}, loss}
    every 25 its: print
save model_last.safetensors; summary.json
```
* `bench.success` = `successes["ppo_deterministic"]` = (# successful episodes)/200 computed in **f32** and compared as a Python float: `Float64(Float32(n_ok)/200.0f) >= 0.85`. Because f32(0.85) = 0.85000002384 > 0.85, exactly 170/200 DOES promote. Port with the same f32 quotient.
* `bench.reward` = mean over episodes of the summed `env.reward()` of every visited state including the initial and the final one.
* Two things the port may do differently without changing the algorithm's intent, flagged as deviations to decide in code review: (a) save the evaluated (pre-update) weights as "best"; (b) seed everything. The faithful behaviour is as written above.
* `diff_max` check uses the driver's `a.diff_max` (default 256); depth = `min(depth_slope * difficulty, max_depth)` is in the env.
* The env's `difficulty` is a plain field of the Rust env; `reset()` uses it (`Env::reset(difficulty)` for Python envs); the collector clones the env so it picks up the difficulty set before `collect`.

### 8.1 Evaluation (`evaluate` / `solve`, Rust)
```
evaluate(env, policy, num_episodes, deterministic, num_searches, num_mcts_searches, C, max_expand_depth, num_cores):
    parallel over episodes (rayon pool of num_cores, or sequential for 1):
        e = env.clone(); e.reset()                                   # random start at current difficulty
        best = ((0.0, -inf), [])
        repeat num_searches times:                                   # each from the SAME start state (e cloned)
            c = e.clone(); total = 0
            while not c.is_final():
                total += c.reward()
                probs = policy.predict(c.observe(), c.masks()).probs  # random twist per call; masked/exp/normalise with +1e-6
                a = argmax(probs) if deterministic else WeightedIndex(probs).sample()     # first max wins on ties; a failed WeightedIndex -> action 0
                c.step(a)
            total += c.reward()                                      # final state's reward is included
            cand = (1.0 if c.success() else 0.0, total)
            if cand > best.value (lexicographic: success first, then total reward): best = cand
        return best
    return (sum success / num_episodes, sum total reward / num_episodes)      # both f32
```
`track_solution` envs (all four qiskit-gym envs) return the env-tracked solution instead of the action list; irrelevant for training metrics.

## 9. MCTS / AlphaZero (present in twisterl, NOT used by QUMO)

QUMO sets `num_mcts_searches = 0` everywhere and uses `PPOConfig`, so none of this runs. Spec for completeness / future use:
```
predict_probs_mcts(env, policy, N, C, max_expand_depth):
    (priors, _) = policy.full_predict(env.observe(), env.masks())          # twist-averaged
    root = Node(state=env, prior=0, visits=1, value_sum=0);  expand(root, priors)   # one child per action with prior > 0
    repeat N:
        node = root; while node.children: node = argmax_child( q + C * sqrt(parent.visits)/(child.visits+1) * child.prior ),
             q = child.value_sum/child.visits (0 if unvisited)           # ties: first strictly greater wins
        value = 0; d = 0
        while d < max_expand_depth:
            value = node.state.reward()
            if node.state.is_final(): break
            (priors, v) = policy.full_predict(node.state.observe(), node.state.masks())
            expand(node, priors); node = sample child by prior (WeightedIndex); value = v; d += 1
        backpropagate(node, value)         # +value, +1 visit up to the root (no sign flip, single agent)
    probs[child.action] = child.visits / sum(visits)   # uniform 1/A if no visits
```
AZ training (`AZCollector`, `AZ.train_step`): `policy.set_permutations([], [])` (no twists), episodes of MCTS-guided play (`action ~ mcts_probs`), per state record `obs`, `mcts_probs` and `remaining_value = sum of rewards from this state onwards`... exactly `total_val_after - total_val_before` where `total_val` accumulates `reward()` of states **including the current one at the time it is added** (so `remaining_values[t] = sum_{k>=t} reward_k`); loss = `cross_entropy(logits, mcts_probs) + mse(value, remaining_value)`; same Adam(3e-4), 10 epochs, defaults 512 episodes x 1000 sims, C 1.41, max_expand_depth 1, seed 123 (seed unused). Not needed for the QUMO port; skip unless requested.

## 10. Checkpoint, config and export formats

* `model.safetensors`, `model_last.safetensors` (QUMO) / `checkpoint_best.safetensors`, `checkpoint_last.safetensors` (twisterl `learn`, not used): safetensors v0 of `policy.state_dict()`: 8-byte little-endian `u64` header length N, N bytes of JSON header `{ "name": {"dtype":"F32","shape":[...],"data_offsets":[begin,end]}, ... }` (optionally `"__metadata__"`), then the raw little-endian f32 tensors, row-major, tensors laid out in the header's key order sorted alphabetically (by name, the writer sorts by dtype then name), no padding between tensors for F32 (offsets are contiguous). Tensors are the 8 (or more) keys of section 2. Example header: `ppo_golden/safetensors_header.json`, file `ppo_golden/tiny_policy.safetensors`. Readers must tolerate any key order. Weight layout: `weight[out][in]` row-major (torch).
* `config.json` (written by `RLSynthesis.save` BEFORE training, consumed by `export_model.py`): `{"env_cls": "qiskit_gym.envs.synthesis.<PermutationEnv|LinearFunctionEnv|CliffordEnv|PauliNetworkEnv>", "env": {num_qubits, difficulty, gateset: [[gate, [q...]], ...], depth_slope, max_depth, metrics_weights, add_inverts, add_perms, (max_rotations for Pauli)}, "policy_cls": "twisterl.nn.BasicPolicy", "policy": {embedding_size, common_layers, policy_layers, value_layers}, "algorithm_cls": "twisterl.rl.PPO", "algorithm": PPOConfig.to_json()}`. `export_model.py` reads `env_cls`, `env.num_qubits`, `env.gateset`, `env.max_rotations` and `SYNTH_ENVS[...].from_json(env)` (to get twists) from it.
* `summary.json`: `kind, topology, num_qubits, iterations, wall_s, budget_s, seed, cores, emb, common, best, final_difficulty, params, init, start_difficulty` (export reads `seed, budget_s, wall_s, cores, iterations, final_difficulty, topology, init`).
* `log.jsonl`: one JSON per iteration (see section 8).
* The Mojo trainer can either keep these exact files (then `export_model.py` stays usable) or write the QUMO container directly; the weights it must hand to `basic_mlp_tensors(sd)` are the 8 tensors above in f32.

## 11. What PPO needs from an environment (contract for the envs port)

`num_actions() -> A`, `obs_shape() -> [..]`, `reset()` (random state at current `difficulty`), `step(a)`, `observe() -> sorted-ish list of distinct bit indices`, `masks() -> [bool]*A` (qiskit-gym: all `!success`), `is_final() -> bool` (`depth == 0 or success`), `reward() -> f32` (state reward stored in `reward_value`: `achieved(1.0 if solved else 0.0) - weighted metrics delta of the last step`, plus `pauli_layer_reward * new_rotations` for the Pauli env; right after reset/set_state it is `1.0 if already solved else 0.0`; verified in qiskit-gym 0.4.1 rust/src/envs/*.rs; all four envs use `masks = [!success]*A`, `is_final = depth == 0 || success`), `success()`, `twists() -> (obs_perms, act_perms)`, `clone()`, `difficulty` get/set. The collector clones the env per episode (cheap, plain data in Rust; in Mojo a struct copy). The training loop never calls `set_state`.

## 12. QUMO's own router PPO (`route/train_ppo.py`), for the same port

Not twisterl; torch only, an optional fine-tuning stage that the README says made the policy worse (so `make train-ai-route` does behaviour cloning + DAgger, which also uses torch: `train_route.py`). Constants for a Mojo port of those parts:
* Net (`RouterNet`): `acts = b0 + sign(obs) @ E` (E [128][H], init `randn * 1/sqrt(128)`, b0 zeros), `h = relu(acts)`, `logits = out(h)` (`Linear(H,16)` torch default init), `value = Linear(H,1)(h)`.
* BC (`train_route.py`): Adam lr 2e-3, `CosineAnnealingLR(T_max=epochs)` stepped once per epoch (eta_min 0): `lr_e = 0.5*lr*(1+cos(pi*e/epochs))`, batch 512, epochs 30 (time budget), loss = soft cross-entropy to `softmax(-(Q - minQ)/tau)`, tau 0.5, random slot permutation augmentation, masked logits `-1e9`.
* PPO fine-tune (`train_ppo.py`): GAE with a different recursion (`delta = r + gamma*V_next - V; adv = delta + gamma*lam*adv_next`, gamma 0.99, lam 0.95, `ret = adv + V`), advantages standardised (population std + 1e-8), 4 epochs, shuffled minibatches of 1024 (`randperm`), clip 0.1, vf 0.5, ent 0.005, Adam lr 2e-4, global grad-norm clip 0.5 (`clip_grad_norm_`), masked logits `-1e9`, rollouts in a `fork` Pool of 8 workers with per-episode numpy `default_rng(seed)`. The shared primitives for both are: Adam, cosine schedule, softmax cross-entropy, global-norm clipping, GAE, randperm minibatching: they should live in one Mojo `nn/optim` module together with the twisterl-style full-batch PPO above.

## 13. Pseudocode summary for the Mojo trainer (module level)

```
struct Policy { W0[E][OBS], b0[E], layers: List[(W[Ci][Ci-1], b[Ci])], Wa[A][Ck], ba[A], Wv[Ck], bv, obs_perms, act_perms }
init(seed):           Linear init per 2.1
forward_sparse(S, twist) -> (logits[A], v, caches)        # 2.2
sample_action(masked_logits, rng) -> argmax(l - ln|ln u|) # 4
collect(env, policy, 1024, gamma, lam, cores, seed)       # 4, parallelize over episodes, seeded per-episode RNG
evaluate(env, policy, EvalConfig)                         # 8.1
ppo_update(batch, opt)                                    # 5, 6: loop 10 times: forward all rows (sparse), losses, analytic backward (6), Adam (7.1)
train(kind, topology, budget_s, ...):                     # 8
save_safetensors(policy, path)                            # 10
```
Edge cases: `A = 1` (log_softmax -> 0, entropy 0); episodes of length 1 (already-solved start: reward 1, n = 1, `advs[0] = r - v`, `rets[0] = r`); `common_layers = []` is legal in the config (policy then goes embedding -> heads, but QUMO always has >=1 layer).

## 14. Golden data (`ppo_golden/`)

All produced by `ppo_golden/gen_golden.py` (run as `venv/bin/python -I gen_golden.py OUT`), which asserts equality of torch, the compiled Rust objects and `ref_ppo.py` before writing.

| file | content |
|---|---|
| `policy_forward.json` | tiny policy (obs_shape [3,3] -> OBS 9, A 4, E 8, common [6]) with explicit f32 weights (torch layout, flat row-major), twists `obs_perms=[[1,2,0,4,5,3,7,8,6],[8,7,6,5,4,3,2,1,0]]`, `act_perms=[[1,2,3,0],[3,2,1,0]]`; 15 cases (twist none / 0 / 1 x 5 observations incl. empty obs, 9-bit obs, an all-masked mask): raw logits, value, Rust masked logits (-1e10), Rust `predict` probabilities; plus 5 ensemble cases (`full_predict`: mean logits, mean value, probabilities); plus the init-bound check |
| `gae.json` | 4 `(gamma, lambda)` settings with the rewards, values, advs, rets produced by the compiled `PPOCollector` on an 8-state toy episode |
| `ppo_update.json` | the fixed 14-row batch (obs, masks, twist index per row, masked old logits, actions, advs, rets, old_logp), the PPO config, and for each of 10 epochs the loss dict; for epochs 0, 1, 9 also all gradients and all weights after the Adam step; plus a `normalize_advantage=True` one-step variant. Initial weights = `policy_forward.json` weights; Adam fresh (state zero) at epoch 0; lr 3e-4 |
| `safetensors_header.json`, `tiny_policy.safetensors` | byte-exact checkpoint of the tiny policy to test the writer/reader |
| `ref_ppo.py` | numpy reference for forward, ensemble, GAE, loss+gradients, Adam (the executable spec) |
| `extra_checks.py` | verifies merge order, twist/logit consistency between Rust collection and the torch forward, and `evaluate` reward accounting |

Facts verified against the real library while generating them: logits stored by the collector equal the torch forward with the stored `perms` to 1.5e-8; with 3 episodes of lengths 4,5,6 the merged data order is [ep2, ep0, ep1]; `evaluate` returns `(success_rate, mean summed reward incl. initial and final state)`; after the first Adam step every parameter with a non-negligible gradient moves by almost exactly `lr = 3e-4` (m/sqrt(v) = sign), so golden weights after one epoch mostly test the gradient sign; the 10-epoch and gradient entries test magnitudes.

## 15. Open points that need a decision during the port (not blockers)

1. Keep unseeded vs per-episode seeded rollouts (recommended: seeded, makes `make train-ai` reproducible; numbers will not match the committed models either way).
2. "Best checkpoint" semantics (post-update weights with pre-update metrics): keep for fidelity or save evaluated weights.
3. Masked-old / unmasked-new ratio quirk: keep (affects only solved final states).
4. Sparse vs dense input path in the embedding-gradient: must be numerically a dense multi-hot product, with dense Adam (7.1).
5. The router BC/DAgger and router PPO use plain Adam + cosine LR + minibatching and need their own Mojo implementations (section 12).
