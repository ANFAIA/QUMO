# qiskit-gym 0.4.1 environments: exact specification for a native Mojo port

Scope: the four environments QUMO trains (`PermutationEnv`, `LinearFunctionEnv`,
`CliffordEnv`, `PauliNetworkEnv`), everything they depend on, and how twisterl 0.5.1
drives them. Written so that a Mojo port can reproduce them bit-for-bit where the
behaviour is deterministic and distributionally where it is random.

Everything below was checked against the REAL environments (the compiled Rust
extension `qiskit_gym_rs` of the installed wheel). See section 13 for the evidence.

Executable form of this spec: `envs_golden/envs_ref.py` (pure Python, line-by-line
transcription of the Rust). When this text and `envs_ref.py` disagree, the code is right
(it is the one that was fuzzed against the Rust binary).

## 0. Provenance and where the logic lives

* Installed in the scratch venv (all pins from `training/ai_transpiler/requirements.txt` were available,
  nothing substituted): qiskit-gym 0.4.1, twisterl 0.5.1, torch 2.14.1, safetensors 0.8.0, numpy 2.5.4,
  qiskit 2.5.2, gymnasium 1.4.0 (Python 3.12.13, macOS arm64 wheels).
* The wheel ships only the compiled Rust extension (`qiskit_gym/qiskit_gym_rs.cpython-312-darwin.so`) plus
  thin Python wrappers. The Rust source was read from the public Apache-2.0 repository at git tag `0.4.1`
  (`https://github.com/AI4quantum/qiskit-gym`, the repo is not under `qiskit-community`): files
  `rust/src/envs/{permutation,linear_function,clifford,pauli,symmetry,metrics,common}.rs` and
  `rust/src/pauli/{pauli,pauli_dag,pauli_network}.rs`. The twisterl driver code was read from tag `0.5.1` of
  `https://github.com/AI4quantum/twisterl` (`rust/src/{rl,collector,python_interface,nn}`).
  Downloaded source was only read, never executed; the source was then validated against the shipped binary
  (section 13).
* ALL environment logic (state, scramble generation, step, reward, termination, symmetries, Pauli DAG) is in
  Rust. Python (`qiskit_gym/envs/synthesis.py`, `adapters.py`) only (a) builds the gateset from a coupling map
  (section 2), (b) wraps the raw env in a gymnasium class (training uses the raw env, `RLSynthesis` passes
  `env._raw_env` to PPO), (c) converts circuits to env states (`get_state`, section 12) and env solutions back to
  circuits. None of (c) is needed during training.
* Per-instance env state in Rust is a trait object `Box<dyn Env>` (`twisterl::rl::env::Env`); twisterl's Rust
  collector/evaluator call it directly and `dyn_clone` the whole env (all fields, incl. the Pauli env's current
  symmetry index) before every episode.

## 1. Python-visible surface (what `train.py` touches)

`PyBaseEnv` methods: `num_actions()`, `obs_shape()`, `difficulty` (get/set), `set_state(list[int])`, `reset()`,
`step(int)`, `masks()`, `is_final()`, `reward()`, `observe()`, `twists()`. QUMO's `train.py` calls
`CliffordGym.from_coupling_map(...)`, reads `env.config["num_qubits"]`, `env.obs_shape()`, `env.num_actions()`,
`env.twists()`, sets `env.difficulty`, and hands `env._raw_env` to PPO.

Rust constructor signatures (pyo3):

```
PermutationEnv / LinearFunctionEnv / CliffordEnv(num_qubits, difficulty, gateset, depth_slope, max_depth,
    metrics_weights=None, add_inverts=None, add_perms=None, track_solution=None)      # None -> True
PauliNetworkEnv(num_qubits, difficulty, gateset, depth_slope, max_depth, max_rotations,
    pauli_diff_scale=None(8), num_qubits_decay=None(0.5), final_pauli_layers=None(max_rotations+2),
    metrics_weights=None, add_perms=None(True), pauli_layer_reward=None(0.01), track_solution=None(True))
```

The Python subclasses (`PermutationGym`, ...) override the defaults: `difficulty=1, depth_slope=2, max_depth=128,
add_inverts=True, add_perms=True, track_solution=True`; `PauliGym` additionally `max_rotations=5,
pauli_diff_scale=16` (NOT Rust's 8), `num_qubits_decay=0.5, final_pauli_layers=None -> 7,
pauli_layer_reward=0.01`. `PauliGym` has no `add_inverts` (the Pauli env never inverts).

`from_coupling_map(...)` only forwards `{num_qubits, difficulty=1, gateset, depth_slope=2, max_depth=128,
metrics_weights=None, add_inverts=True, add_perms=True}` (Pauli: without `add_inverts`). It does NOT forward
`max_rotations` / `pauli_diff_scale` / `num_qubits_decay` / `final_pauli_layers` / `pauli_layer_reward`, so
QUMO's Pauli training always uses `max_rotations=5, pauli_diff_scale=16, num_qubits_decay=0.5,
final_pauli_layers=7, pauli_layer_reward=0.01`. (The `max_rotations` parameter of `train.make_env` is unused.)
`track_solution` is irrelevant to learning (it only records the solution); the Mojo trainer can drop it.

## 2. Construction: gateset, action indexing, observation size, coupling graph

`BaseSynthesisEnv.from_coupling_map(coupling_map, basis_gates, ...)` (qiskit_gym/envs/synthesis.py):

```
coupling_map = sorted(list of (q1,q2) tuples)           # plain lexicographic sort of the given tuples; duplicates are kept
num_qubits   = max(max(e) for e in coupling_map) + 1    # so a trailing isolated qubit is NOT counted
gateset = []
for gate_name in basis_gates (in the given order):
    if gate_name in {H, S, Sdg, SX, SXdg}:  for q in 0..num_qubits-1: gateset.append((gate_name, (q,)))
    else (CX, CZ, SWAP):                    for (q1,q2) in coupling_map: gateset.append((gate_name, (q1,q2)))
```

Action index = position in `gateset`. `num_actions = len(gateset)`.
Allowed basis: Permutation {SWAP}, LinearFunction {CX, SWAP}, Clifford and Pauli {H,S,Sdg,SX,SXdg,CX,CZ,SWAP}.
Rust parses names case-insensitively and rejects unknown names / wrong arity with ValueError.

How QUMO builds the coupling map (`train.py`): `edges` = sorted undirected pairs (a<b) from
`common/topology.py`; `coupling_list(edges, bidirectional)`:

| kind | basis | coupling passed | actions |
|---|---|---|---|
| perm | `("SWAP",)` | one direction, `(a,b)` with a<b, sorted | `SWAP(a,b)` per edge, in edge order; A = |E| |
| lf | `("CX",)` | both directions, sorted | `CX(a,b)`, `CX(b,a)` ... sorted lexicographically over all directed pairs; A = 2|E| |
| clifford | `("H","S","CX")` | both directions, sorted | `H(0..n-1)`, `S(0..n-1)`, then sorted directed `CX`; A = 2n + 2|E| |
| pauli | `("H","S","CX")` | same as clifford | same as clifford |

Example `clifford line:3`: `[H0,H1,H2,S0,S1,S2,CX(0,1),CX(1,0),CX(1,2),CX(2,1)]`.

Observation size (`obs_shape`, flattened row-major, the policy input width):

| env | obs_shape | flat size |
|---|---|---|
| permutation | [n, n] | n^2 |
| linear_function | [n, n] | n^2 |
| clifford | [2n, 2n] | 4n^2 |
| pauli_network | [2n, 2n + max_rotations] = [2n, 2n+5] | 2n(2n+5) |

Observations are SPARSE: the sorted ascending list of flat indices of the 1-entries (the Python gym adapter
densifies with `np.zeros(prod(shape), int8); full[idx]=1`; twisterl's `data_to_torch` also densifies as multi-hot
float). The sparse list is what the policy's EmbeddingBag consumes.

## 3. The common Env protocol and how twisterl drives it

Shared fields: `depth`, `success`, `reward_value` (f32), `difficulty`, `depth_slope`, `max_depth`, `gateset`,
`metrics`, `metrics_values`, `metrics_weights`, (and the state).

* `masks()` = `[not success] * num_actions` (true means "allowed"). All true while unsolved, all false once solved.
* `is_final()` = `depth == 0 or success`.
* `success()` = the `success` flag (= `solved()` evaluated at reset / set_state / after each step).
* `reward()` = `reward_value` (f32).
* `step(a)` always decrements `depth` (saturating at 0) even if the env is already final or `a` is invalid.
* `set_difficulty(d)`: just stores `d` (takes effect at the next `reset()`).
* Construction state: identity state, `depth = 1`, `success = true`, `reward_value = 1.0`, i.e. `is_final()` is true
  until the first `reset()`/`set_state()`.

twisterl's drivers (rust/src/collector/ppo.rs, rl/solve.rs, rl/evaluate.rs). This protocol decides how rewards
align with observations and is therefore part of the environment contract:

```
PPOCollector.single_collect(env):                       # env = clone of the training env
    env.reset()
    loop:
        obs   = env.observe()                           # Pauli: draws and stores a random symmetry index
        masks = env.masks()
        reward= env.reward()                            # reward of ARRIVING in this state (initial reward after reset)
        logits,value,perm = policy.forward_with_perm(obs, masks)     # logits masked: not allowed -> -1e10
        action = argmax(logits - ln(|ln(U)|))  (Gumbel-max sampling, U~Uniform(0,1) f32 per action)
        record (obs, logits, action, value, reward, perm)
        if env.is_final(): break                        # NOTE: the terminal state is recorded too (with an action that is never executed)
        env.step(action)
evaluate(env, ..): for each episode: env = clone; env.reset(); single_solve:
    total = 0
    while not env.is_final():  total += env.reward(); obs = env.observe(); masks = env.masks();
                               probs = softmax-like masked predict; action = argmax | sample; env.step(action)
    total += env.reward();  success = 1.0 if env.success() else 0.0
solve(num_searches): runs single_solve on clones, keeps the best by `success` (strict >, first wins ties)
```

Consequences for the port: (1) exactly one `observe()` per decision, immediately before the `step()` it conditions
(the Pauli env relies on it); (2) a state that is already solved at `reset()` yields a 1-row episode with reward 1.0
(plain envs: all masks false); (3) the reward attached to row t is the reward earned by the PREVIOUS action (row 0
carries 1.0 if the scramble happened to be the identity, else 0.0).

Curriculum (twisterl `Algorithm.learn_step` + QUMO `train.py`, see also `qiskit_gym/rl/configs.py` PPOConfig):
`env.difficulty = start_diff (1)`; every iteration: evaluate the CURRENT policy at the CURRENT difficulty with
`diff_metric = "ppo_deterministic"` (200 episodes of `reset()` + greedy rollout, success rate over episodes), then
collect/train; afterwards `if success_rate >= diff_threshold (0.85) and difficulty < diff_max (QUMO: 256): difficulty += 1`.
Difficulty is therefore one integer; it influences ONLY `reset()` (length of the scramble / the Pauli budgets) and the
episode horizon `depth` (below).

## 4. Reward and metrics (identical in all four envs)

Per `step(a)` (if `a < num_actions`; otherwise penalty = 0 and no state change):

```
prev = metrics.snapshot()                     # (n_cnots, n_layers_cnots, n_layers, n_gates)
metrics.apply_gate(gate)
penalty = weighted_delta(metrics.snapshot(), prev)
reward  = achieved - penalty          (+ pauli_layer_reward * rotations_cleaned_this_step   for the Pauli env)
achieved = 1.0 if solved else 0.0
```

All arithmetic is IEEE f32, evaluated exactly as `((w0*d0 + w1*d1) + w2*d2) + w3*d3` with `d_i` the (non-negative,
saturating) integer increments converted to f32, `reward = (achieved - penalty) + (pauli_layer_reward * f32(count))`.
Default weights: `n_cnots = 0.01`, `n_layers_cnots = 0.0`, `n_layers = 0.0`, `n_gates = 0.0001` (a dict passed as
`metrics_weights` overrides entries by key; unknown keys ignored). Verified bit-exact against the Rust env, also with
non-default weights including the layer terms (section 13).

`MetricsTracker(num_qubits)` (reset to zero/`-1` at every `reset()`/`set_state()`):

```
cx(c,t):   if c==t or c>=n or t>=n: return
           n_cnots+=1; n_gates+=1
           L = max(last_gates[c], last_gates[t]) + 1; last_gates[c]=last_gates[t]=L; layers.add(L)
           K = max(last_cxs[c],  last_cxs[t])  + 1; last_cxs[c]=last_cxs[t]=K;  cnot_layers.add(K)
one(t):    if t>=n: return;  n_gates+=1; L = last_gates[t]+1; last_gates[t]=L; layers.add(L)
H,S,Sdg,SX,SXdg(q) -> one(q);   CX(c,t) -> cx(c,t);   CZ(c,t) -> one(t); cx(c,t); one(t)
SWAP(a,b) -> cx(a,b); cx(b,a); cx(a,b)
snapshot = (n_cnots, |cnot_layers| , |layers|, n_gates)      # sets, so counts are the number of DISTINCT layer indices
```

Typical default penalties (f32): 1q gate 0.0001, CX 0.0101, CZ 0.0103, SWAP 0.0303 (3 CX: 3*0.01+3*0.0001). The
`last_*` arrays make the layer counts history dependent, which only matters if the layer weights are non-zero.

## 5. Common reset / set_state behaviour of the three "tableau" envs (Permutation, LinearFunction, Clifford)

```
reset():
    state = identity
    repeat `difficulty` times:  a ~ Uniform{0..A-1};  apply gateset[a] to state          # NO inversion, NO rejection
    depth = min(depth_slope * difficulty, max_depth)                                      # = min(2d, 128) with QUMO's settings
    _reset_internals()
set_state(v):  state = decode(v) (no validation, no cleaning);  depth = max_depth (128);  _reset_internals()
_reset_internals():  success = solved(); metrics.reset(); metrics_values = snapshot();
                     reward_value = 1.0 if success else 0.0;  inverted = false;  (solution lists cleared)
```

Notes: the scramble is a plain uniform random walk of exactly `difficulty` gate applications starting from the
identity; gates that cancel (e.g. the same involution twice) are not filtered, so the identity can be produced
(rate in `reset_empirical.json`, e.g. permutation line:4 d=2 -> 0.33). A solved reset gives an immediately-final
episode. Every gate in these envs acts as an involution on the bit state (S and Sdg both act as `S`, SX and SXdg as `SX`;
S∘S = identity on the symplectic bits), so a scramble is undone by re-applying the scramble gates in reverse order.

Step (all three):

```
step(a):
    penalty = 0
    if a < A:  penalty = metrics update for gateset[a];  state = gate(state)          # (Permutation: only SWAP entries change the state)
    maybe_random_invert()                                                              # if add_inverts: with probability 1/2: state = state^-1
    depth = max(depth-1, 0)
    success = solved();   reward_value = f32(achieved) - penalty
```

`maybe_random_invert` is `gen_bool(0.5)` (exactly 1/2) from a fresh thread RNG draw per step; it is applied
AFTER the gate and BEFORE `solved()` is evaluated (the order relative to the depth decrement is immaterial). The inverse
of the identity is the identity, so success/reward are identical in both branches; only the observation differs.
`reset()` and `set_state()` never invert. `add_inverts=True` is what `from_coupling_map` uses by default and what
QUMO trains with. A singular matrix reaching `invert()` panics ("LFState/CFState is singular") - with a valid reset
and group-element actions that cannot happen; it can only happen via `set_state` with garbage.

Tracked solution (only needed when SOLVING with an `add_inverts` env, e.g. when the trained policy is rolled out in
the real env; irrelevant to training): while `inverted == false` the action is appended to `solution`, otherwise to
`solution_inv`; each invert flips `inverted`; final solution = `solution ++ reverse(solution_inv)`. (Permutation
records only if `a < A`; LF/Clifford record regardless.)

## 6. PermutationEnv

State `s[0..n)`, a permutation, identity = solved. `observe() = [i*n + s[i] for i in 0..n)` (ascending).

```
gate SWAP(a,b):  s[a], s[b] = s[b], s[a]            # swaps the VALUES stored at positions a and b
other gates:     no state change (they still count in metrics)
invert:          inv[s[i]] = i
solved:          s[i] == i for all i
```

`set_state(v)`: `s = v` as usize (any list, not validated; `inverse` would index out of range if not a permutation).
Reset scramble = `difficulty` random SWAP applications, SWAP drawn uniformly over the A gateset entries.
Python `get_state` and the relation to circuits: section 12.

## 7. LinearFunctionEnv

State: n x n bit matrix `M` as n row masks (`bit j of row[i]` = M[i][j]); identity = solved.
`observe() = [i*n + j for i in 0..n for j in 0..n if M[i][j]]` (row-major ascending).

```
CX(c,t):  if c != t: row[t] ^= row[c]
SWAP(a,b): if a != b: swap(row[a], row[b])            # row swap
others:   no state change
invert:   Gauss-Jordan over GF(2) (result is unique, any correct inverse is bit-identical)
```

`set_state(v)`: `M[i][j] = v[i*n+j] > 0`.

## 8. CliffordEnv

State: 2n x 2n bit matrix as 2n row masks (row i in `0..n` = "X rows", rows `n..2n` = "Z rows"; bit j = column j),
identity = solved. Phases are NOT modelled. `observe() = [i*2n + j ... if bit]`.

```
H(q):          swap(row[q], row[n+q])
S(q), Sdg(q):  row[n+q] ^= row[q]
SX(q), SXdg(q):row[q]   ^= row[n+q]
CX(c,t):       if c != t: row[t] ^= row[c];  row[n+c] ^= row[n+t]
CZ(a,b):       if a != b: row[n+a] ^= row[b];  row[n+b] ^= row[a]
SWAP(a,b):     if a != b: swap(row[a],row[b]); swap(row[n+a],row[n+b])
invert:        Gauss-Jordan over GF(2) of the 2n x 2n matrix
```

`set_state(v)`: `bit(i,j) = v[i*2n+j] > 0` (arbitrary, not required to be symplectic as long as add_inverts is off or
the matrix is invertible).

## 9. PauliNetworkEnv (the complicated one)

### 9.1 State

* `n`; `max_rotations = 5` (columns shown); `final_pauli_layers = 7` (rotations created by `reset`, shown or not).
* `data`: 2n rows x (2n + R) columns of bits. Columns `0..2n` = Clifford tableau (identity = solved). Column
  `2n + i` = rotation `i` (original index `i`): bits `data[q][2n+i] = x_q`, `data[n+q][2n+i] = z_q` of its Pauli.
  Row ops act on whole rows, i.e. simultaneously on the tableau and on all rotation columns.
* DAG over rotation indices: node list in petgraph node-index order (initially `[0..R)`); edge `i1 -> i2` for
  `i2 < i1` iff Pauli `i1` and `i2` anticommute (computed once from the ORIGINAL x/z masks; Clifford conjugation
  preserves commutation, so it is static). A node is in the "front layer" iff it has no outgoing edge among the
  nodes still in the DAG, i.e. there is no remaining earlier rotation `rj < ri` that anticommutes with it.
* Rotation Pauli phases are tracked in Rust (`Pauli.base_phase`) but never influence `observe`, `reward`,
  `is_final` or `success`, only the `solution` encoding. The Mojo trainer can ignore phases.
* `solved = (node count == 0) and (data[r] & (2^(2n)-1) == 1<<r for all r in 0..2n)`.
* Initial construction: identity tableau, no rotations, `solved`, `depth=1`.

Rotation label parsing (used by `set_state` and the internal builder): a label string `L` (length n after an optional
phase prefix `[+-]?[ij1]?`) maps character index `k` to qubit `n-1-k` (qiskit order: leftmost char = highest qubit);
`X`: x=1; `Z`: z=1; `Y`: x=z=1.

### 9.2 Gates (`PauliNetwork::act`), action `gateset[a] = (name, q0[, q1])`

```
H(i):    swap(row[i], row[n+i])
S(i):    row[n+i] ^= row[i]
Sdg(i):  S(i) three times (== S on the bits)
SX(i):   row[i] ^= row[n+i]
SXdg(i): SX(i) three times
CNOT(i,j) [internal]:  row[i] ^= row[j];  row[n+j] ^= row[n+i];  then clean()
CX(q0,q1)  -> CNOT(q0,q1)                  # !! the control of the tableau/rotation update is q1, target q0
CZ(q0,q1)  -> H(q1); CNOT(q0,q1); H(q1)
SWAP(q0,q1)-> CNOT(q0,q1); CNOT(q1,q0); CNOT(q0,q1)
```

Note the CX convention is MIRRORED w.r.t. the Clifford env: `CNOT(i,j)` is the symplectic action of a CX with control
`j` and target `i`. This is why `PauliGym._reconstruct_circuit_from_solution` emits `circuit.cx(*gate_args[::-1])`.
`clean()` runs after every CNOT (hence at most once per CX, and 3x per SWAP, and once per CZ); H/S/SX never clean.
A rotation's weight (|x| ∪ |z| count) is invariant under H/S/SX (the extra clean after the final `H` of CZ in
`refenv.py` is therefore a no-op).

### 9.3 clean() (`clean_and_return_with_phases`) and the petgraph node order

```
removed = true
while removed:
    removed = false
    nodes = current node list (index order)                        # snapshot for this pass
    front = [ri for ri in nodes if no rj in nodes with rj < ri and anti[ri][rj]]          # in node-list order
    kill  = []
    for ri in front:
        if popcount(x_ri | z_ri) <= 1  (computed from data columns 2n+ri):
            report (axis, qubit, ri)             # not needed for training except the count
            zero data column 2n+ri; kill.append(ri); removed = true
    # petgraph retain_nodes: iterate node indices from HIGH to LOW; every killed node is swap-removed
    for pos in len(nodes)-1 down to 0:
        if nodes[pos] in kill:  nodes[pos] = nodes[last]; pop last          # the last node moves into the hole
```

Important details: (a) the front layer is computed once per pass; a node that only becomes front because an earlier
one was removed in the same pass is considered in the NEXT pass; (b) removal uses swap-remove so the order of
`nodes` is NOT the creation order after removals; that order decides which rotation occupies visible column 0,1,...;
(c) weight 0 never occurs for generated rotations (weight >= 2 at creation, conjugation preserves non-identity);
`set_state` with an all-`I` label would panic in Rust (`unwrap` on `which_qubit`).
The number of rotations removed in a `step` = `count`, rewarded with `pauli_layer_reward` each.

### 9.4 observe()

```
dense = zeros(2n x (2n+5));  dense[r][c] = data[r][c] for c < 2n
for i, ri in enumerate(nodes[:5]):    dense[r][2n+i] = data[r][2n+ri]  for r in 0..2n      # rotation k occupies visible column k
if qubit_perms nonempty (add_perms):
    p = random index in 0..len(qubit_perms);  CURRENT_PERM = p                              # stored in the env, used by step()
    dense = apply_perm(dense, qubit_perms[p])
return sorted indices r*(2n+5)+c of the 1 entries
apply_perm(dense, perm):   # row i <- row perm[i], row n+i <- row n+perm[i] (all columns), then
                           # col i <- col perm[i], col n+i <- col n+perm[i] for the 2n tableau columns only
                           # rotation columns (>= 2n) keep their position (rows were already permuted)
```

Rotations beyond the 5th stay in the DAG (they must be removed before `solved`) but are invisible. When fewer than
5 rotations are active the remaining visible columns are zero.

### 9.5 step(action)

```
actual = act_perms[CURRENT_PERM][action]   if act_perms nonempty else action          # un-permute (model frame -> real qubits)
if actual < A:  penalty = metrics(gateset[actual]);  count = len(act(gateset[actual]))
depth = max(depth-1, 0);  success = solved()
reward_value = (f32(success) - penalty) + (pauli_layer_reward * f32(count))
```

CURRENT_PERM starts at 0 (the identity automorphism, the lexicographically smallest) and is overwritten by every
`observe()`. With `add_perms` on, an action >= A PANICS (index out of bounds); with `add_perms=False` it is a silent
no-op. `twists()` returns `([], [])` for this env (symmetries are internal) so the policy gets no obs/act perms.

### 9.6 set_state(vec)

`vec = [R, tableau(4n^2 entries, row-major, entry>0 -> 1), then R times: len, ord(ch)*len]`, labels may carry a phase
prefix (`-`, `i`, `-i`, ...) which is part of `len`. If `vec` is empty nothing happens. Only the first
`max_rotations` rotation labels are kept (the rest are dropped; the tableau columns beyond are ignored), a malformed
state panics. Then `network = PauliNetwork::new(...)` (DAG over ALL kept rotations, NO initial `clean()`, so a weight-1
rotation stays in the DAG until some later CNOT cleans it), `depth = max_depth`, `success = solved()`,
metrics reset, `reward_value = 1.0 if success else 0.0`.

### 9.7 reset()

```
pauli_difficulty = difficulty // pauli_diff_scale                   # integer division, scale = 16
rotations = generate_paulis(pauli_difficulty)                       # up to final_pauli_layers (=7) labels
tableau   = random_clifford_tableau(difficulty)                     # `difficulty` random gates, NOT pauli_difficulty
network   = PauliNetwork::new(tableau, rotations); network.clean()  # the clean is a no-op for generated rotations (weights >= 2)
depth = min(depth_slope*difficulty, max_depth); success = solved(); metrics.reset(); reward_value = 1.0 if success else 0.0
```

`random_clifford_tableau(d)`: start from identity; do nothing if `d == 0` or there are no CX actions. Repeat `d` times,
`u` = f32 uniform on multiples of 2^-24 (`k * 2^-24`, `k` uniform in `[0, 2^24)`):

```
if u > 0.3f32       (k >= 5033166; probability 11744050/2^24 ~ 0.7):   (q0,q1) = uniform element of VALID_PAIRS
                                                                       row[q1] ^= row[q0];  row[n+q0] ^= row[n+q1]
elif u > 0.15f32    (k in [2516583, 5033165]; 2516583/2^24 ~ 0.15):    q = uniform{0..n-1}: swap(row[q], row[n+q])        # H
else                (k <= 2516582; 2516583/2^24 ~ 0.15):               q = uniform{0..n-1}: row[n+q] ^= row[q]            # S
```

`VALID_PAIRS` = `(q0,q1)` of EVERY CX action in gateset order (both directions for QUMO), selection uniform over that
list. Only H, S, CX are ever used, whatever the gateset (SX/CZ/SWAP are never scrambled). The tableau here is
the plain Clifford-env convention (row[q1] ^= row[q0]); because QUMO's coupling is bidirectional the mirrored CX
convention of `act` does not change the distribution. The exact distribution is given in `reset_exact.json`
(`pauli_tableau`).

`generate_paulis(pd)`:

```
paulis = []; rem = pd
while rem > 0 and len(paulis) < final_pauli_layers:
    r = pauli_under_diff(rem)
    if r is None: break
    (label, cost) = r;  paulis.append(label);  rem = max(rem - max(cost,1), 0)         # saturating
```

`pauli_under_diff(D)` (D = remaining budget; `ALL_DISTS` = sorted distinct graph distances >= 1 between pairs `a<b`
of the CX-coupling graph (BFS on the undirected graph of VALID_PAIRS, unreachable pairs absent);
`DIST_PAIRS[d]` = list of pairs `(a,b), a<b` at distance d in lexicographic order):

```
valid = [d in ALL_DISTS if d <= D];  if valid empty: return None
qubits = {};  pd = D
nd = uniform(valid);  (q1,q2) = uniform(DIST_PAIRS[nd]);  qubits += {q1,q2};  pd = max(pd - nd, 0)
loop:
    vd  = [d in valid if d <= pd];   rem_q = [q in 0..n-1 not in qubits]
    if pd == 0 or vd empty or rem_q empty: break
    if gen::<f32>() <= num_qubits_decay (0.5):  break        # P(stop) = (2^23+1)/2^24  (k <= 8388608)
    nd = uniform(vd)
    vp = [(a,b) in DIST_PAIRS[nd] if a in qubits or b in qubits]
    if vp empty: continue                                    # re-enters the loop: the stop coin is flipped AGAIN
    (q1,q2) = uniform(vp); qubits += {q1,q2}; pd = max(pd - nd, 0)
layer = ['I'] * n;  for q in qubits: layer[q] = uniform('X','Y','Z')    # independent per qubit
return (''.join(layer), D - pd)
```

QUIRK (verified, matters for non-reflection-symmetric graphs): the string is later parsed with the qiskit convention
(character `k` <-> qubit `n-1-k`), so a qubit `q` chosen above ends up on qubit `n-1-q` in the env. E.g. on a star with
centre 0, all generated rotations act on qubit n-1, not on the centre. Reproduce by writing the generated label with
position `q`, then mapping position `k` to bit `n-1-k`. (`envs_golden/stats_lib.py` has the exact rational
distribution of the generated label lists; `reset_exact.json` contains it.)

Consequence: for `difficulty < 16` there are NO rotations (pure Clifford synthesis with an extra 5 zero columns); a
rotation appears every time the budget `difficulty // 16` is >= 1 (distance-1 pair = a CX-adjacent pair). With the
QUMO caps (difficulty <= 256 -> budget <= 16, 7 layers max) there are at most 7 rotations, at most 5 visible.

### 9.8 Pauli-env symmetries

See section 10; `qubit_perms` (the raw automorphisms, `perm[i]` = image of i) and the matching `act_perms` are used
inside observe/step as above.

## 10. Symmetries (`twists`) for Permutation / LinearFunction / Clifford

`compute_twists_square` (Perm, LF) / `compute_twists_clifford`, only when `add_perms` (default true):

```
adjacency[a][b] = adjacency[b][a] = true for every 2-qubit gate (CX, CZ, SWAP) of the gateset with a != b
autos = all automorphisms of that graph (perm[i] = image of i); if no edge at all: all n! permutations
        (petgraph VF2 enumeration of graph-onto-itself, then sort lexicographically and dedup)
for perm in autos (lexicographic order):
    act_perm = for every gate g (in gateset order): index of the gate obtained by relabelling q -> perm[q]
               (SWAP keys are unordered: (min,max); CX/CZ keys are ordered; 1q gates keyed by qubit)
               if ANY image is not in the gateset: skip this perm entirely
    obs_perm = Square:   obs_perm[r*n + c] = perm[r]*n + perm[c]
               Clifford: with m(i) = perm[i] if i<n else n+perm[i-n]:  obs_perm[r*2n + c] = m(r)*2n + m(c)
    keep (obs_perm, act_perm)
if none kept: use the identity permutation if its action perm exists
```

How the policy uses them (twisterl `Policy._raw_predict`): for each forward pass a uniformly random index `p` is
drawn; observation indices are mapped `i -> obs_perms[p][i]` before the embedding; the resulting logits are
re-indexed `logits_out[a] = raw[act_perms[p][a]]`; `p` is stored in the rollout and reused in the PPO update.
`full_predict` (MCTS/AZ only) averages over all p. `twists()` of the plain envs return clones of these tables.
Verified to equal `training/ai_transpiler/common/symmetry.py::twists` on 11 topologies x 3 kinds
(`envs_golden/verify_twists.py`), the automorphism count equals the table in `gateset_twists.json`
(e.g. line:n -> 2, ring:5 -> 10, star:5 -> 24, grid:2x3 -> 4, heavyhex H7 -> 8).

For the Pauli env the same `compute_qubit_perms` result is used INTERNALLY (section 9.4/9.5) and `twists()` is empty.
Orientation check (what is consistent): the model sees state `P` with `P[i] = S[perm[i]]`, so a model action on
model-qubit `q` is the real gate on qubit `perm[q]`; `act_perms[p][a]` = index of the gate with qubits mapped through
`perm` (this is exactly "unpermute" in `step`).

## 11. Random number requirements for the Mojo port

Rust uses `rand::thread_rng()` everywhere (unseeded, per call). Only the distributions matter:

| draw | distribution |
|---|---|
| scramble action (Perm/LF/Cliff) | uniform integer in `[0, A)` |
| `maybe_random_invert` | Bernoulli(1/2), once per `step` when `add_inverts` |
| Pauli: CX/H/S choice | f32 `u = k*2^-24`, thresholds in section 9.7 (P(CX)=11744050/2^24, P(H)=P(S)=2516583/2^24) |
| Pauli: pair / qubit / distance / axis choices | uniform over the listed candidates |
| Pauli: decay stop | `u <= 0.5f32` i.e. probability (2^23+1)/2^24 |
| Pauli: symmetry index in observe() | uniform in `[0, n_sym)` per observe() call |
| policy permutation index (non-Pauli) | uniform in `[0, n_sym)` per forward pass (policy side, not env) |

The number of random numbers consumed per reset is data dependent, so no bit-identical RNG stream is possible or
needed; use the golden distributions (below) for validation.

## 12. Circuit <-> state conversion (Python `get_state`, only for evaluation / inference, not training)

* Permutation: `argsort(pattern)` of `LinearFunction(circuit).permutation_pattern()`.
* LinearFunction: `LinearFunction(Clifford(circuit).adjoint()).linear.flatten()` (0/1 ints).
* Clifford: `Clifford(circuit).adjoint().tableau[:, :-1].T.flatten()` (0/1 ints).
* Pauli: `[R] + tableau(adjoint of the Clifford part, same slicing as Clifford) + for each rotation [len, ord(ch)...]`
  with rotation labels `Pauli(single-qubit axis string).evolve(current_clifford).adjoint().to_label()` (may begin with `-`,
  `i`, `-i`); `current_clifford` accumulates the Clifford gates seen so far; rx/ry/rz parameters are stored in Python
  and re-attached on the solution (`decode_pauli_solution`, marker `0x80000000 | axis<<21 | qubit<<11 | index<<1 | phase`).
* Verified equivalent (200/200 random circuits each) to this env-level rule: the state equals the identity with the
  circuit's gates applied as env actions in REVERSED circuit order (`cx(c,t)` = env `CX(c,t)` for Perm/LF/Clifford;
  `cx(c,t)` = env `CX(t,c)` for the Pauli env, because of the mirrored convention). Golden examples:
  `envs_golden/get_state.json`.

## 13. Evidence: how the spec was validated against the real binary

All scripts live in `envs_golden/` (venv python). Summary of what passed:

1. `verify_twists.py`: gateset order and `twists()` identical to `envs_ref` on 14 topologies x 4 kinds (Pauli: real
   `twists()` empty; internal tables validated by behaviour in item 3).
2. `verify_steps.py`: 21 (kind, topology) configs x 150 random `set_state` starts x up to 40 steps (~60k steps), incl.
   out-of-range actions, arbitrary non-group matrices, Pauli with weight-1 rotations and R > max_rotations.
   `observe()`, `reward()` (exact f32 equality), `is_final()`, `masks()` identical at every step.
   Same with non-default metrics weights (layer terms non-zero) and non-default Pauli arguments (`max_rotations=2`,
   `pauli_diff_scale`, `num_qubits_decay`, `final_pauli_layers`, `pauli_layer_reward`, `depth_slope=3`, `max_depth=20`).
3. `verify_pauli_perms.py`: Pauli with `add_perms=True`: the hidden symmetry index is recovered from the real
   observation; reward/finality/next observation reproduced for 5 topologies x 120 trajectories.
4. `verify_random.py`: (A) post-reset horizon = `min(2d,128)` for d in 3,40,64,65,200 in all four envs; (B) exact DP
   distribution of the post-reset state vs 40000 real resets, chi-square |z| <= 2.7 on 14 (kind, topology, d) cases
   (real) and <= 1.5 (ref sampler); (C) add_inverts: from explicit states the post-step observation is always
   `{S', inv(S')}` with P(inverted) in [0.48, 0.52].
5. `verify_pauli_reset.py` + `verify_pauli_gen_exact.py`: Pauli tableau vs exact DP (5 cases), rotation-list vs the
   exact rational distribution of the generator on 5 (topology, difficulty) cases (|z| <= 2.2), two-sample agreement
   real vs `envs_ref` on 13 cases; rotation count empirically 3 at d=48 and exactly 7 at d=160/256 (cap
   `final_pauli_layers`, measured through the 0.01 cleaning bonus on long random walks).
6. `replay_golden.py`: replays all golden JSON against `envs_ref.py` (self-consistency of the files).

## 14. Golden files (`envs_golden/`) and how to use them from Mojo

| file | content | test |
|---|---|---|
| `defaults.json` | versions, constructor defaults, f32 thresholds | read constants |
| `gateset_twists.json` | per (kind, topology): coupling map passed, gateset `[name,q0,q1]` (q1=-1 for 1q), `obs_shape`, `obs_perms` + `act_perms` (Pauli: internal `qubit_perms` + `act_perms`) | build gateset and symmetry tables, compare element-wise |
| `steps_<kind>.json` | explicit `set_state` vectors + action sequences + per-step `obs`, `reward` (f32 as double), `is_final`, `allowed`, with `add_inverts=False` (and `add_perms=False` for Pauli) | exact equality of obs/is_final, f32 equality of reward |
| `steps_custom_args.json` | same with non-default metrics weights and non-default Pauli constructor args | exact |
| `inverts.json` | explicit state + action -> the two possible observations (`obs_plain`, `obs_inverted`), reward, is_final | each outcome in the set, ~50/50 over many trials |
| `pauli_perm_cases.json` | Pauli with `add_perms=True`: `obs_by_perm[p]` for each automorphism p (the real env produced only members of this list), and for each (action, p): `actual_action`, reward, is_final, next obs under p=0 | exact |
| `reset_exact.json` | exact rational reset distributions: scramble states (Perm/LF/Clifford, several d), Pauli tableau (d=1..11), Pauli rotation-label lists (d=16,32,48) | draw N>=1e5 resets, chi-square against the table |
| `reset_empirical.json` | real-env measurements (N=1e5): visible-rotation-count histograms and first-rotation weight histograms for several topologies/difficulties, solved-at-reset rates | tolerance comparison |
| `get_state.json` | circuits (gate lists) -> real `get_state` vectors for all four kinds | only if the port converts circuits |

State encodings in the JSON: Perm = list `s`; LF/Clifford `set_state` vectors are flat 0/1 matrices (row-major); inside
`reset_exact.json` LF/Clifford states are lists of row bitmasks (bit j = column j) and Pauli tableaux the 2n row bitmasks
of the tableau columns; Pauli `set_state` vectors follow section 9.6; `obs` are sorted sparse indices.

## 15. Differences between this spec and the existing `training/ai_transpiler/common/refenv.py`

`refenv.py` is the inference-side mirror ("without random inversions / random symmetries") and is consistent with the
Rust for the quantities it models (all its gate actions, observation layout and clean() order agree with this spec).
Points a Mojo TRAINING env must add or get right that `refenv.py` does not cover:

1. No scramble / reset generator (all of sections 5, 9.7), no depth/horizon, no reward, no metrics, no random inversion,
   no internal Pauli symmetry draw in observe(), no `masks`.
2. Pauli `set_state` does NOT clean initially-trivial rotations (`refenv.pauli_start` + fixtures call `clean`
   explicitly); the Rust only cleans inside CNOT (and in `reset`).
3. Pauli `CZ`: Rust cleans once (inside its CNOT); `refenv` additionally cleans after the final H, which is a no-op.
4. The mirror quirk of the rotation label generator (section 9.7) and the `final_pauli_layers=7` vs 5 visible columns.
5. `pauli_twists` in `symmetry.py` builds an inference ensemble; the training env instead applies ONE random
   automorphism per `observe()` and un-permutes the action in `step()` (section 9.5).

## 16. Pitfalls checklist for the port

* Row-index conventions: tableau row `i<n` is X-row of qubit i, `n+i` Z-row; Pauli env CX is mirrored; Pauli `act` CZ and
  SWAP are built from the mirrored CNOT.
* The plain envs' random inversion happens EVERY step with probability 1/2 (including the step that solves the
  problem, where it is harmless) and never on `reset`/`set_state`.
* Reward at row t belongs to the transition INTO state t; terminal rows are recorded; solved states have all-false masks.
* `reset()` can return an already-solved state (episode of length one).
* `set_state` uses `depth = max_depth` (128) independent of difficulty; `reset` uses `min(2*difficulty, 128)`.
* Difficulties above 64 saturate the horizon at 128 while the scramble keeps growing.
* The Pauli env must see `observe()` before every `step()` when `add_perms` is on; `step(a >= A)` panics then.
* Keep everything in f32 for rewards.
* Packed-integer representation is safe for QUMO sizes: rows fit in 32 bits for n <= 12 (Clifford 2n bits, Pauli
  2n + 7 hidden columns).
* Not needed for training: `track_solution`, rotation phases, `solution()`, Python `get_state`/circuit rebuilding.
