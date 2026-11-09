# torch / numpy / safetensors port specification (QUMO AI-transpiler training -> native Mojo)

Scope: every torch, numpy and safetensors operation used by `training/ai_transpiler/{route,synth,common}`
and by the twisterl 0.5.1 Python layer (`twisterl.nn.BasicPolicy`, `twisterl.rl.PPO`, `twisterl.rl.Algorithm`)
that `synth/train.py` drives. Torch 2.14.1, numpy 2.5.4, safetensors 0.8.0 (scratch venv
`/private/tmp/claude-501/-Users-ismaelsertage-code-QUMO/775d4539-a54f-4085-bfda-d988bb18fbcc/scratchpad/rlport/venv`).
All statements below were checked against the installed sources and against real torch runs; the golden data in
`torch_golden/` was produced by torch 2.14.1 and cross-checked against the numpy reference in `torch_golden/ref_np.py`
(float32, tolerances in section 9) and against finite differences (`gradcheck.py`, float64, relative error < 5e-6).

What is NOT torch and is out of scope here: the Rust PPO rollout collector / GAE / evaluation (twisterl `collector`),
qiskit-gym environments, `route_env.py` (already bit-exact against the Mojo Rng), `circuits.py` (python `random`).

---------------------------------------------------------------------------------------------------------------
## 1. Inventory: who uses what

| File | torch / numpy / safetensors surface |
|---|---|
| `route/train_route.py` | `np.fromfile` structured dtype, `np.sign`, `np.nanmax/nanmin`, `default_rng(seed).permutation`, `torch.manual_seed`, `nn.Parameter(randn)`, `nn.Linear` x2, `relu`, `@`, `masked_fill`, `softmax`, `log_softmax`, `randperm(generator)`, `rand(generator)`, `argsort`, `gather`, `Adam`, `CosineAnnealingLR`, autograd `backward`, `np.savez`, `json` |
| `route/train_ppo.py` | `Adam`, `randperm`, `clamp`, `min`, `exp`, `log_softmax`, `gather`, `masked_fill`, mean-MSE, `clip_grad_norm_(0.5)`, numpy float64 forward + `Generator.choice(p=)` + GAE loop + `std()`, `np.savez` |
| `route/export_npz.py`, `export_route.py` | `np.load(.npz)`, float32->float16 `astype`, `default_rng(0).integers/random`, float64 forward of f16-decoded weights |
| `common/export.py` | `astype("<f2")`, `np.isfinite`, `frombuffer`, `tobytes`, FNV-1a 64, SHA-256, JSON; `basic_mlp_tensors` (transpose of `embeddings.weight`) |
| `synth/train.py` | twisterl `BasicPolicy` (nn.Linear stack), twisterl `PPO.train_step` (Categorical, clamp, min, mse_loss, Adam), `torch.manual_seed`, `safetensors.torch.save_file/load_file`, `load_state_dict` |
| `synth/export_model.py` | `safetensors.numpy.load_file`, `basic_mlp_tensors`, `default_rng(20261014)`, `np.allclose`, float32 `refenv.Policy` forward |
| `synth/eval.py` | only the optional `qiskit_baseline` uses numpy (`default_rng(7)`, `argsort`, `mean`); everything else is subprocess/JSON |

No mixed precision anywhere: no autocast, no GradScaler, no bf16, no `.half()`. Everything trains in float32 on CPU
(`device = "cuda" if torch.cuda.is_available() else "cpu"`; on the M5 it is cpu). The only reduced precision is the
export: float16 weight storage (section 7).

---------------------------------------------------------------------------------------------------------------
## 2. Networks

### 2.1 RouterNet (`route/train_route.py`)  [route model `qumo.rlroute.v0`]
Parameters (state order = `net.parameters()` order, which only matters for the global grad-norm sum):

| name | torch object | shape | init (torch.manual_seed(seed), draws in this order) |
|---|---|---|---|
| emb | `nn.Parameter(randn(128,H) * (1/sqrt(128)))` | [128,H] (feature-major = container layout, NO transpose on export) | N(0,1) * 0.08838835 (float32 randn then multiply) |
| b0 | `nn.Parameter(zeros(H))` | [H] | 0 |
| out.weight / out.bias | `nn.Linear(H,16)` | [16,H], [16] | U(-1/sqrt(H), 1/sqrt(H)) each, weight first then bias |
| value.weight / value.bias | `nn.Linear(H,1)` | [1,H], [1] | U(-1/sqrt(H), 1/sqrt(H)), weight then bias |

`nn.Linear` default init = `kaiming_uniform_(a=sqrt(5))` => bound = sqrt(6/((1+5)*fan_in)) = 1/sqrt(fan_in) for the weight
(fan_in = in_features), and bias bound = 1/sqrt(fan_in); both uniform. Verified: golden `synth_ppo_golden.json:init_bounds`.
Bit-exact reproduction of torch's Mersenne/Philox draws is neither possible nor needed (section 8).

Forward (`forward(s)`, s float32 [B,128] with entries in {-1,0,+1} = `np.sign(obs_int16)`):
```
pre = s @ emb + b0           # [B,H]   == sparse "embedding bag": sum of +/- rows of emb for the nonzero features
h   = relu(pre)              # relu'(0) = 0 (torch: grad passes iff output > 0, i.e. pre > 0)
z   = h @ out.weight^T + out.bias     # logits [B,16]
v   = (h @ value.weight^T + value.bias)[:,0]      # [B]
```
`s @ emb`: nonzeros only; implement as a loop over nonzero features adding/subtracting `emb[i,:]` (sign is exactly +-1,
so no multiplies). Per-sample nnz <= 128.

Export tensors (`export_route.tensors_from_numpy`): embeddings.weight=emb [128,H], embeddings.bias=b0,
action.0.weight=out.weight [16,H], action.0.bias=out.bias, value.0.weight=value.weight [1,H], value.0.bias=value.bias [1].

### 2.2 twisterl BasicPolicy (`synth/train.py`)  [synth model `qumo.rlsynth.v0`]
Constructed by qiskit-gym as `BasicPolicy(obs_shape=[obs], num_actions, embedding_size=emb, common_layers, policy_layers=[], value_layers=[], obs_perms, act_perms, device)`.
Modules created in this order (this is also the RNG draw order for init): `embeddings = nn.Linear(obs_size, emb)`;
`common = Sequential(Linear, ReLU, Linear, ReLU, ...)` (omitted if `common_layers` empty); `action = Sequential(Linear(in, policy_layers..., num_actions))`
with ReLU between layers and NO final ReLU; `value` likewise ending in `Linear(.., 1)`. With `policy_layers=value_layers=()` both heads are a single Linear.
Because ReLU modules occupy Sequential indices, state-dict keys skip odd numbers when there is more than one layer:
`common.0.*, common.2.*, ...`, `action.0.*` (or `action.0.*, action.2.*` with a hidden policy layer). The container keeps the Sequential index
(`basic_mlp_tensors` sorts by index). Verified in `synth_ppo_golden.json:state_dict_keys` (two arch cases).

state_dict (and thus safetensors) order = parameters order: embeddings.{weight,bias}, common.*, action.*, value.*; `embeddings.weight` is
torch layout [emb, obs] (nn.Linear weight [out,in]); the container stores its TRANSPOSE [obs, emb] (`first_layer_transposed = true`).
All other weights stay [out,in]. Init bounds: each Linear U(+-1/sqrt(in_features)), weight then bias.

Forward on a batch of dense one-hot-ish float obs x [B,obs] with per-sample permutation index `p` (`-1` = none):
```
if p >= 0:  x = gather(x, 1, argsort(obs_perm[p]))    # x'[j] = x[obs_perm_inv[j]]  <=> active index i moves to obs_perm[p][i]
h0 = relu(x @ We^T + be)                               # sparse: be + sum_{i active} Wt[obs_perm[p][i], :]   (Wt = container [obs,emb])
h  = h0; for each common layer: h = relu(h @ W^T + b)
raw = action(h)  (Linear[+relu] chain, last Linear has no relu)
v   = value(h)   -> [B,1]
logits = raw.index_select(1, act_perm[p]) if p >= 0   # logits[a] = raw[act_perm[p][a]]
```
(The inference convention `obs' = obs_perms[p][i], logits_out[a] = raw[act_perms[p][a]]` of the manifest is exactly this.)
Backward of the act gather: `draw[act_perm[p][a]] = dlogits[a]` (act_perm is a permutation). Pinned by
`synth_ppo_golden.json` (uses the REAL `twisterl.nn.policy.BasicPolicy.forward` for the golden forward).
No conv policy is used by QUMO (`Conv1dPolicy` not needed).

---------------------------------------------------------------------------------------------------------------
## 3. Losses and manual backward

Notation: B = batch size, `valid[b,k]` bool mask, `p = softmax`, `lp = log_softmax`, `1[.]` indicator.
All reductions are `.mean()` over the batch (divide by the batch size actually used, including a short last batch).

### 3.1 Masked log-softmax (route)
`zm = where(valid, z, -1e9)` (`masked_fill(~valid, -1e9)`, NOT a mask applied after softmax); `lp = zm - logsumexp(zm)`
(max-shifted). Invalid slots end up with p == 0 exactly and receive exactly zero gradient (masked_fill blocks it).
In Mojo simply loop over valid slots only; the valid set after slot permutation is an arbitrary subset of the 16 slots (not a prefix).

### 3.2 Behaviour-cloning loss (`train_route.py`), soft-label cross entropy
```
Qb        = Q (clipped: min(Q, 1e6)) with invalid slots set to 1e9  (masked_fill(~valid, 1e9))   # after slot permutation
target    = softmax( -(Qb - min_k Qb) / tau )      # over the 16 slots; invalid slots -> exactly 0;  tau = 0.5 default
loss      = - mean_b sum_k target[b,k] * log_softmax(zm)[b,k]
```
`target` is constant (no grad). Gradient wrt the masked logits:
```
dz[b,k] = ( p[b,k] * sum_k' target[b,k'] - target[b,k] ) / B      for valid k;   0 for invalid k
```
(sum_k target == 1 up to f32 rounding; torch's autograd evaluates exactly the formula above; using `p - target` is
within 1e-7 and acceptable). The value head is NOT in the BC loss: `value.weight/bias` get `grad = None`, therefore the
Adam update SKIPS them (no step, no state) and they keep their init values (random U(+-1/sqrt(H)) for fresh runs, or the loaded
`--init` values) all the way to export. Golden asserts this (`adam_state_after_3.has_value_state == false`).

Backward through the network (G = gradient accumulator):
```
dW1 = dz^T @ h              [16,H]        db1 = sum_b dz                [16]
dh  = dz @ W1               [B,H]
dpre = dh * (pre > 0)
dE  = s^T @ dpre            [128,H]  (sparse: for each nonzero feature i of sample b: dE[i,:] += s[b,i]*dpre[b,:])
db0 = sum_b dpre            [H]
```

### 3.3 Slot-permutation augmentation (BC only)
Per minibatch, per sample: `perm = argsort(rand(B,16))` (uniform random permutation of 16, new slot j takes old slot perm[j]):
```
obs3 = s.view(B,16,8)                                 # feature index = 8*k + d
s2   = gather(obs3, 1, perm[:, :, None] over d)        # s2[b, 8*j + d] = s[b, 8*perm[b,j] + d]
Q2   = gather(Q, 1, perm);  valid2 = gather(valid, 1, perm)
```
The permutation is applied to the already-clipped Q (before masking with 1e9) and to `valid`; `target` is then computed from
`Q2`/`valid2`. The environment's own slot order is shuffled, so any uniform permutation is statistically equivalent; golden gives the
explicit `slot_perm` used. Validation (`evaluate`) uses NO permutation.

### 3.4 Route PPO loss (`train_ppo.py`)
Per minibatch (`st` float32 sign obs [n,128], `At`, action `at`, `lpt` old log-prob (float32 from the float64 rollout), `advt` normalized adv, `rett`):
```
logits, v = net(st); zm = masked_fill(logits, ~valid, -1e9); lp = log_softmax(zm)
nlp   = lp[b, at[b]];   ratio = exp(nlp - lpt)
s1    = ratio * adv;    s2 = clamp(ratio, 1-c, 1+c) * adv
pl    = -mean(min(s1, s2))
ent   = -mean_b sum_k exp(lp) * where(valid, lp, 0)           # entropy of the masked policy
vl    = mean((v - ret)^2)                                      # v:[B], ret:[B]
loss  = pl + 0.5 * vl - ent_coef * ent                         # vf coef is the literal 0.5; defaults c = 0.1, ent_coef = 0.005
```
Backward:
```
live[b]   = (adv >= 0) ? ratio <= 1+c : ratio >= 1-c            # the min()/clamp gradient rule (see below)
dratio[b] = live ? -adv[b]/B : 0 ;   dnlp[b] = dratio[b] * ratio[b]
dz[b,k]   = -p[b,k]*dnlp[b] + [k==at[b]]*dnlp[b]                 # d nlp / d z_k = onehot - p
          + ent_coef/B * p[b,k] * (lp[b,k] + ent_b[b])           # entropy bonus: d(-ent_coef*mean ent)/dz_k  (valid k only)
dz[b,k] = 0 for invalid k
dv[b]     = 0.5 * 2 * (v[b] - ret[b]) / B
```
then `router_backward` of 3.2 with `dh += dv[:,None] * wv` and `dwv = dv^T @ h`, `dbv = sum dv`.
Clip rule (verified against torch for ties and both signs, `ppo_route_golden.json:clip_rule_microtest`): torch splits a `min` tie 50/50 between
`s1` and `s2`, and the clamp passes the gradient inside [1-c,1+c], so inside the interval the full `-adv/B` flows; outside, the gradient flows only
through the branch that is the minimum. Exactly ON the clip bound torch may drop one half (its backward compares the f32 ratio with the f64 bound);
this happens with probability ~0 and the port should not try to reproduce it.
Gradient clipping `nn.utils.clip_grad_norm_(net.parameters(), 0.5)` (after backward, before `opt.step()`):
```
total = sqrt( sum over ALL 6 parameter tensors of sum g^2 )        # one global L2 norm (torch accumulates per-tensor norms in f32)
coef  = min( 0.5 / (total + 1e-6), 1.0 );   g *= coef  (every grad, even when coef == 1.0)
```

### 3.5 twisterl PPO loss (`synth`)  (`twisterl.rl.ppo.PPO`)
Full batch (the entire collected dataset, N samples, **no minibatches, no shuffling**), repeated `num_epochs` (10) times per `learn_step`:
```
old_logp = Categorical(logits = recorded rollout logits).log_prob(actions)      # computed once, no_grad (logits - logsumexp)
pred_logits, pred_vals = policy(dense_obs, perm_indices)                        # section 2.2; ALL actions valid: no masking
logp   = Categorical(logits=pred_logits).log_prob(actions)  # = lp[b,a]
ent    = Categorical.entropy().mean()                        # -sum p*lp
ratio  = exp(logp - old_logp);   surr1 = ratio*adv;  surr2 = clamp(ratio, 1-c, 1+c)*adv
policy_loss = -mean(min(surr1,surr2))
value_loss  = mse_loss(pred_vals[N,1], rets.unsqueeze(1))    # mean over N
loss = policy_loss + vf_coef*value_loss - ent_coef*ent       # defaults vf_coef 0.8, ent_coef 0.01, clip 0.1; no grad clipping
```
`normalize_advantage` is False by default and QUMO never sets it (if enabled: `(a-mean)/(std(unbiased) + 1e-8)` using torch's UNBIASED std,
computed under no_grad). Backward = the same formulas as 3.4 without masks and with `vf_coef` instead of 0.5:
`dz = -p*dnlp + onehot*dnlp + ent_coef/B * p*(lp+ent_b)`, `dv = vf_coef*2*(v-ret)/B`, then back through the action head (with ReLU between
layers of a deeper head), value head, **sum of both head gradients into the shared hidden**, the common layers, and the sparse embedding bag
(`dWt[obs_perm[p][i], :] += dpre0[b,:]` for every active index). Golden: `synth_ppo_golden.json` (two architectures, with symmetry).
Inputs the optimizer needs from the (Rust) collector: sparse obs index lists, perm index per sample (`-1` if none), the logits that were
used to sample, actions, GAE advantages and returns.

---------------------------------------------------------------------------------------------------------------
## 4. Optimizer: `torch.optim.Adam`
Defaults everywhere: betas (0.9, 0.999), eps 1e-8, weight_decay 0, amsgrad False, maximize False; lr = 2e-3 (BC, scheduled), 2e-4 (route PPO),
3e-4 (twisterl). Single-tensor non-capturable CPU path (foreach variants are the same math). Per parameter with a gradient, step t = 1,2,...:
```
m  = m + (g - m) * (1 - beta1)                 # lerp_(g, 1-beta1)   [weight 0.1 < 0.5 -> start + w*(end-start)]
v  = v * beta2 + (1 - beta2) * g * g            # mul_(beta2).addcmul_(g, g, value=1-beta2)
bc1 = 1 - beta1^t ;  bc2 = 1 - beta2^t           # python doubles
step_size = lr / bc1
denom = sqrt(v) / sqrt(bc2) + eps                # eps is added AFTER the division by sqrt(bc2)
p  = p - step_size * m / denom                   # addcdiv_(m, denom, value=-step_size), float32 tensors
```
Notes that matter for the port:
* Parameters whose `.grad is None` are skipped entirely (BC value head; there `opt.zero_grad()` sets grads to None, torch>=2 default).
* The step counter and moments are per optimizer and persist across the whole run: BC across epochs; route PPO across ALL PPO iterations (one
  optimizer created before the loop); twisterl across all `learn_step`s (`Algorithm.__init__`). Loading a checkpoint (`--init`) does NOT reset anything.
* The update is DENSE: a sparse embedding-bag implementation must still apply Adam to every row, because rows with zero gradient and non-zero
  momentum still move (golden `adam_trace`, 4th step with zero gradient). Rows never touched since init have m = v = 0 and update exactly 0, so they may be
  skipped as an exact optimization (track a per-row "ever touched" flag).
* m, v, p all float32; scalars `step_size`, `sqrt(bc2)` are doubles then cast per op. Matching torch to ~1e-7 relative is enough; Adam is sign-like for
  gradients much larger than eps = 1e-8 (update ~ lr * sign(g)), so a tiny rounding difference in a near-zero gradient can flip a full lr-sized step: the weights-after tolerance must scale with lr (section 9).

### 4.1 Scheduler (BC only): `CosineAnnealingLR(opt, T_max=epochs)`, eta_min 0
`sched.step()` once per epoch AFTER the epoch's minibatches. The lr used during epoch e (0-based) is
`lr_e = base * (1 + cos(pi * e / T_max)) / 2` (torch uses a recursive form; max |difference| from the closed form in float64 is 2e-19, golden
`adam_cosine_golden.json:cosine.lr_per_epoch_torch` for T_max=30, base 2e-3: 0.002, 0.0019945..., ..., 5.478e-6 at e=29). PPO has no scheduler.
If the wall-clock `--budget_s` is exceeded the loop breaks after finishing the current epoch; the schedule is not rescaled.

---------------------------------------------------------------------------------------------------------------
## 5. Training loops in exact order

### 5.1 `train_route.py` (BC / DAgger)
1. `torch.manual_seed(seed); np.random.seed(seed)` (numpy global seed is unused later).
2. Load records (`REC` packed little-endian, 328 bytes: obs int16[128] @0, A int16 @256, act int16 @258, job int16 @260, tb int16 @262, Q f32[16] @264); concatenate files in the given order.
3. Drop degenerate states: `Qv = where(arange(16) < A, Q, nan)`; `spread = nanmax(Qv,1) - nanmin(Qv,1)`; keep rows with `spread > 1e-6` (float32 values; A = 0 or 1 or all-equal Q are dropped; NaN rows from A = 0 compare False).
4. Split: `idx = default_rng(seed).permutation(len(d))`; `nval = max(1000, len(d)//20)`; `val = d[idx[:nval]]`, `train = d[idx[nval:]]`.
5. `tensors()`: `s = float32(sign(obs))`, `A int64`, `Q = float32(min(Q, 1e6))`.
6. Net init (section 2.1); if `--init`, overwrite all six tensors from the model dir (container f16-decoded to f32; value tensors only if present, reshaped to [1,H],[1]).
7. `Adam(lr=2e-3)`, `CosineAnnealingLR(T_max=epochs)`, `g = torch.Generator().manual_seed(seed)` (one stream used both for `randperm` and for slot-permutation `rand`).
8. For each epoch: `order = randperm(N, generator=g)`; for `b in range(0, N, 512)` (last batch short, kept): gather rows `order[b:b+512]`; build `valid = arange(16) < A`; slot-permute (3.3); target (3.2); forward; masked CE; `zero_grad; backward; step`; accumulate `loss.item()*len(batch)`.
   Epoch loss logged = sum / N. Then `sched.step()`, then `evaluate(net, val)`.
9. `evaluate` (no_grad, float32): `pick = argmax(masked logits)` (first max on ties; invalid slots -1e9); `Qm = masked_fill(~valid, 1e9)`; `qmin = Qm.min(1)`; `regret = Qm[pick] - qmin`; `agree = mean(Qm[pick] <= qmin + 1e-6)` (the `+1e-6` is a float32 add); logs `val_agree`, `val_regret` (means).
10. Write `weights.npz` (arrays emb [128,H], b0 [H], w1 [16,H], b1 [16], wv [1,H], bv [1], all float32) + `train_log.json` (`samples=N`, `val=nval`, hidden, epochs_run, tau, lr, seed, data basenames, wall_s, per-epoch log). The Mojo port may skip npz and write the container directly (section 7); if npz is kept see 7.3.

### 5.2 `train_ppo.py` (optional PPO fine-tune; requires a working Python-mirror env, i.e. an env port)
Per iteration: build jobs; rollouts in worker processes using float64 weights `(emb,b0,w1,b1,wv,bv)` (`net.numpy()` cast to float64):
`s = sign(obs)` (float64), `h = max(b0 + s@emb, 0)`, `lg = (w1@h + b1)[:A]`, `p = exp(lg - max)/sum`, `a = rng.choice(A, p=p)` (inverse-CDF on the cumulative sum
normalized by its last element, `u = rng.random()`, `searchsorted(side='right')`), `logp = log(p[a] + 1e-12)`, `v = wv.flatten() @ h + bv[0]`.
Then GAE (float64, over the concatenated trajectories, `D[i]` = last step of a trajectory):
```
for i from last to first:  nv = 0 if D[i] else V[i+1];  delta = R[i] + gamma*nv - V[i];  last = delta + gamma*lam*(0 if D[i] else last);  adv[i] = last
ret = adv + V ;  adv_norm = (adv - mean) / (std_ddof0 + 1e-8)      # numpy std, population; ret uses the UN-normalized adv
```
(`gamma = 0.99`, `lam = 0.95`; golden `ppo_route_golden.json:gae`). Tensors are cast to float32. 4 epochs; each epoch `perm = randperm(n)` (global torch generator, seeded by `manual_seed(seed)`); minibatches of 1024 (last short kept); loss 3.4; `clip_grad_norm_(0.5)`; `Adam(lr=2e-4)` step. No slot permutation in PPO. `net.numpy()` is re-exported each iteration; final `weights.npz` + `ppo_log.json`.
Note the net is constructed with random init then fully overwritten by `--init` (so init randomness is irrelevant; only `randperm` consumes the torch stream).

### 5.3 `synth/train.py` (twisterl via qiskit-gym)
`torch.manual_seed(seed)` only affects the policy init (rollouts use an unseeded Rust RNG, per the docstring: trainings are not bit-reproducible by design).
Per `learn_step`: `policy.to_rust()` sync -> evals (Rust) -> collect (Rust) -> `data_to_torch` (dense float obs `zeros(N, obs_size)` set to 1.0 at the sparse indices; perms default -1)
-> `policy.train()` -> 10 full-batch `train_step`s -> returns the LAST step's `{value, policy, entropy, total}`. Best-checkpoint rule (Python): tuple `(difficulty, success, reward) >= best` saves `model.safetensors`;
difficulty += 1 when `success >= 0.85` and `< diff_max`. `model_last.safetensors` at the end. `--init`/`__cont` warm start = `load_state_dict(load_file(path))`.

---------------------------------------------------------------------------------------------------------------
## 6. What the Mojo kernels must provide (minimum set)
1. Sparse sign/one-hot embedding bag forward + backward (rows accumulate in a fixed, single-threaded or fixed-partition order for determinism).
2. Small dense `gemv/gemm` (h @ W^T, dz^T @ h, dz @ W) in float32.
3. ReLU fwd/bwd (`pre > 0` mask), masked log-softmax / softmax over 16 slots, categorical log-prob / entropy over A actions, gather, scatter by `act_perm`.
4. Losses of 3.2 / 3.4 / 3.5 with the hand-derived backward (no autograd).
5. Adam (4), global-norm clip, cosine lr, per-epoch shuffle, per-sample 16-slot permutation, GAE (float64).
6. RNG: uniform [0,1), normal (init only), integers, shuffle, categorical sampling. Use QUMO's existing xoshiro256** `Rng`; Box-Muller for the one-time N(0,1) init.
7. Export: float32 -> float16 (RNE), FNV-1a 64, SHA-256, JSON writer, 64-byte aligned container writer; optionally npz and safetensors I/O.

---------------------------------------------------------------------------------------------------------------
## 7. Serialization semantics

### 7.1 float32 -> float16 (`write_model`)
`arr.astype("<f2")` = IEEE round-to-nearest-even; |x| >= 65520 -> +-inf (65519.99 -> 65504), subnormals preserved (min 2^-24), 2^-25 and below the
halfway point -> 0 (exact tie 2^-25 -> 0 by RNE), NaN stays NaN. `write_model` then raises `ValueError("tensor X overflows f16")` if ANY converted value
is non-finite. Decoding f16 -> f32 is exact. Which tensors are f16: every tensor whose name does not end in `.bias` when `dtype="f16"` (default for both
models); all `.bias` tensors are f32. Tensor order, 64-byte alignment padding (zero bytes before each tensor and at the end), little-endian, row-major.
The manifest stores `fnv1a64` (16 hex) and `sha256` of the whole `weights.bin`; the writer re-reads and checks decoded equality.
Test vectors in the manifest are computed from the DECODED (f16-rounded) weights, float64 for the router (`RouteMLP`), float32 for synth (`refenv.Policy`).
Golden: `f16_golden.json` (326 values: normals, tiny values, halfway ties, 65504/65519.99/65520/65536/1e6/-70000, subnormals, +-0): `f32_bits -> f16_bits -> decoded f32 bits`;
Mojo `Float16` conversion (LLVM `fptrunc`) is RNE and must match bit-for-bit; check overflow detection separately (4 values overflow in the golden set).

### 7.2 safetensors (only for synth warm-start and `export_model.py` input; optional once `export` reads the container directly)
File = `u64 LE header_len` | JSON header (UTF-8, space-padded so `8 + header_len` is a multiple of 8) | raw data. Header maps `name -> {"dtype":"F32","shape":[..],"data_offsets":[begin,end]}` (offsets relative to the data section, tensors contiguous, row-major LE);
optional `"__metadata__"`. Writer order in the header is sorted by (dtype, name) (alphabetical for all-F32) — readers must not depend on it. See `torch_golden/safetensors_sample.safetensors` and `misc_golden.json:safetensors` (full header + data hex). Keys are the state_dict names of 2.2.

### 7.3 npz (`np.savez`, uncompressed) if the BC -> export hand-off stays as a file
ZIP with STORED members `emb.npy, b0.npy, w1.npy, b1.npy, wv.npy, bv.npy`; each `.npy` v1.0: magic `\x93NUMPY`, bytes `01 00`, u16 LE header length, ASCII dict
`{'descr': '<f4', 'fortran_order': False, 'shape': (128, 64), }` space-padded and `\n`-terminated so that (10 + header_len) % 64 == 0, then raw LE float32; needs CRC-32 per member. Byte-exact example:
`misc_golden.json:npz`. Recommendation: skip it and have the Mojo trainer write `qumo.rlroute.v0` directly (the existing `export_npz.py` does only: load weights, merge `train_log.json` minus `log` and `training_meta.json` into `training`, call `export_router`).

---------------------------------------------------------------------------------------------------------------
## 8. RNG and reproducibility policy
* Nothing in the original pipeline is reproducible bit-for-bit across runs (8 torch threads, unseeded Rust collector in synth, `randperm`/`rand` streams, multiprocess rollouts). Models are treated as fixed hashed artifacts.
  The port needs only statistical equivalence: same distributions, same hyper-parameters, same loss/optimizer math, plus exact single-step parity against the goldens (injecting the batch order / slot permutations / minibatch contents).
* Torch Mersenne/Philox streams and numpy PCG64 outputs are NOT to be replicated. Documented only (`misc_golden.json`): `torch.manual_seed(0); randn(4) = [1.5410, -0.2934, -2.1788, 0.5684]`; `default_rng(7).permutation(10) = [8,0,7,1,3,6,2,4,5,9]`.
* Distribution requirements: `torch.randn`: N(0,1) (init of `emb` only); `Linear` init: U(-b,b); `randperm(N)`: uniform random permutation (Fisher-Yates); `argsort(rand(B,16))`: uniform random permutation of 16 per sample (draw independently per sample and per batch); `default_rng(seed).permutation` for the train/val split: uniform permutation; `Generator.choice(A, p)`: inverse CDF.
* Manifest `test_vectors.obs_logits` observations come from `default_rng(0).integers(-3,4,128)` and `random(128) < 0.5`; with a different RNG the vectors differ but stay valid (they are recomputed from the stored weights). `parity.py`/`make_fixtures.py` use `default_rng(...).normal` only to create random test weights.
* The shared `Rng` in `route_env.py` (xoshiro256** seeded by SplitMix64) is already bit-exact with Mojo; use it for any seeded Mojo-side sampling.

---------------------------------------------------------------------------------------------------------------
## 9. Golden data (`torch_golden/`) and tolerances
Generated by `gen_golden.py` (real torch 2.14.1; also asserts numpy-reference == autograd while generating). `gradcheck.py` is numpy-only (finite differences in float64, 4 loss families, passes with worst relative error 4.9e-6).
`ref_np.py` is an executable reference of every formula in sections 3-4 (manual backward, Adam, clip, cosine) usable as a line-by-line template for the Mojo code.

| file | content |
|---|---|
| `bc_golden.json` | H=8, B=6, tau=0.5. `init` weights (incl. value head), 3 steps each with: lr (cosine, T_max=3, base 2e-3: 0.002, 0.0015, 0.0005), raw `obs_sign`, `A`, `Q_raw` (1e9 in invalid slots), the explicit `slot_perm`, `permuted_obs/Q`, `valid`, `target`, loss, all gradients (value-head grads are `null`), weights after the step; Adam state after 3 steps. Value head weights must stay equal to `init` |
| `ppo_route_golden.json` | H=8, n=9, lr 5e-3, clip 0.1, ent 0.005, vf 0.5, max_norm 0.5: 3 steps (sample 0 has ratio exactly 1); loss parts, ratios, raw and clipped grads, `grad_norm_before_clip` (18.58, 4.65, 3.59), `clip_coef` (0.027, 0.108, 0.139: clipping fires every step), weights after; GAE golden (adv, ret, normalized adv); clip-rule micro-test |
| `synth_ppo_golden.json` | two BasicPolicy architectures (`emb6_common5`, `emb6_common5-4_pol3_val2`), obs 12, 4 actions, 2 symmetries, perm_idx in {-1,0,1}: state_dict keys + init bounds, init weights (torch layout), batch (sparse obs, recorded logits, actions, advs, rets), 3 full-batch steps with lr 1e-3: metrics, grads (torch layout), weights after |
| `adam_cosine_golden.json` | cosine lr for T_max=30/base 2e-3; scalar Adam trace including a zero-grad step with momentum |
| `f16_golden.json` | f32 bits -> f16 bits -> decoded bits |
| `misc_golden.json`, `safetensors_sample.safetensors`, `npz_sample.npz` | RNG/format reference bytes |

Tolerances that were validated (numpy float32 reference vs torch float32): loss abs 2e-5 (relative for large), gradients `atol 3e-6, rtol 2e-4`, global grad norm rel 1e-4,
weights after Adam `atol 3e-5` at lr <= 5e-3 (Adam amplifies tiny-gradient noise: scale the weight tolerance with lr, roughly `0.01 * lr`), f16 bits exact.
A Mojo float32 implementation with a different summation order will sit at the same level; compare against these, not bit-exact. For high-confidence checking also run the Mojo kernels in Float64 and compare to the float64 numpy reference (`ref_np.py`) at 1e-10.

Acceptance beyond unit parity (end-to-end, since training is stochastic): on the same `d0.bin` + seed the Mojo BC trainer should land within noise of the Python run on `val_agree` / `val_regret` (compare 3 seeds; Python numbers come from `train_log.json`), and the exported model must pass the existing router parity test (`parity.py`-equivalent) and `models-verify`.

---------------------------------------------------------------------------------------------------------------
## 10. Gotchas checklist
1. BC value head: no gradient, no Adam state, weights frozen at init (or `--init` values). Random-init value head is exported as is.
2. Adam must be dense over every parameter (momentum on rows with zero current gradient); skip only never-touched rows.
3. `eps` is added after `sqrt(v)/sqrt(bc2)`; `step_size = lr/bc1` with `t` counting every `opt.step()` since optimizer creation.
4. Cosine lr is per epoch, applied to all steps within the epoch; epoch index starts at 0.
5. Masked slots: logits -1e9 BEFORE log-softmax; targets exactly 0 there; never multiply 0 by a -1e9 log-prob in a summation without skipping (Mojo: skip invalid slots).
6. Slot permutation permutes `obs` blocks of 8 features, `Q` and `valid` together; `target` is built after the permutation.
7. `torch.randperm`-style epoch shuffle, last minibatch short but kept; loss normalization uses the actual batch size per step and N for the logged epoch mean.
8. Global grad-norm clipping applies to route PPO only (not BC, not twisterl); coefficient is applied unconditionally (1.0 when below the max).
9. twisterl PPO is full-batch x10 epochs on the same data with a persistent Adam; state_dict indices skip ReLU slots (`common.0`, `common.2`, ...); `embeddings.weight` is [emb,obs] in torch and transposed [obs,emb] in the container; entropy has no mask; `rets`/`advs` come from the collector.
10. `+1e-6` in `agree` is float32 arithmetic (`qmin` ~ 4..16 has ulp ~ 5e-7..1e-6: add in float32 to reproduce the same boundary).
11. f16 overflow aborts the export; biases are always f32; encoded vectors in the manifest use decoded weights.
12. `np.std` in route PPO is population (ddof=0); torch `.std()` (twisterl, only when `normalize_advantage`) is unbiased.
13. `sign(obs)` of int16 gives {-1,0,1}; only the sign is used, magnitudes are ignored at training and inference.
14. `Q` stored as 1e9 in invalid slots; `tensors()` clips Q at 1e6 first, then masks with 1e9 after the permutation — equivalent to masking by `valid`; in Mojo just ignore invalid slots.
