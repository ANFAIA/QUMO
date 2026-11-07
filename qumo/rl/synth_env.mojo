"""Training environment: reset / step / reward / metrics / masks over PolicyEnv (qiskit-gym semantics).

STUB (W-1): the signatures of MOJO_PORT.md section 5 only. Component C6 replaces
this file; every behavioural body aborts or raises with a "STUB" message.
"""

from std.os import abort
from qumo.rl.gateset import EnvConfig
from qumo.rl.symmetry import Twists


struct TrainEnv(Copyable):
    var cfg: EnvConfig
    var seed: UInt64
    var difficulty: Int
    var err: Int

    def __init__(out self, cfg: EnvConfig, seed: UInt64) raises:
        self.cfg = cfg.copy()
        self.seed = seed
        self.difficulty = cfg.difficulty
        self.err = 0
        raise Error("QM-E640: qumo.rl.synth_env.TrainEnv is a STUB (component C6)")

    def num_actions(self) -> Int:
        return len(self.cfg.actions) // 3

    def obs_size(self) -> Int:
        abort("qumo.rl.synth_env.TrainEnv.obs_size: STUB (component C6)")

    def reseed(mut self, seed: UInt64):
        abort("qumo.rl.synth_env.TrainEnv.reseed: STUB (component C6)")

    def set_difficulty(mut self, d: Int):
        abort("qumo.rl.synth_env.TrainEnv.set_difficulty: STUB (component C6)")

    def reset(mut self):
        abort("qumo.rl.synth_env.TrainEnv.reset: STUB (component C6)")

    def set_state_perm(mut self, s: List[Int]):
        abort("qumo.rl.synth_env.TrainEnv.set_state_perm: STUB (component C6)")

    def set_state_bits(mut self, v: List[Int]):
        """LF / Clifford flat 0/1 matrix."""
        abort("qumo.rl.synth_env.TrainEnv.set_state_bits: STUB (component C6)")

    def set_state_pauli(mut self, v: List[Int]) raises:
        """envs.md 9.6 vector."""
        raise Error("QM-E640: qumo.rl.synth_env.TrainEnv.set_state_pauli is a STUB (component C6)")

    def observe(mut self, mut obs: List[Int]):
        """Sorted ascending; Pauli draws and stores the hidden symmetry."""
        abort("qumo.rl.synth_env.TrainEnv.observe: STUB (component C6)")

    def all_allowed(self) -> Bool:
        """masks() == [not success] * A."""
        abort("qumo.rl.synth_env.TrainEnv.all_allowed: STUB (component C6)")

    def is_final(self) -> Bool:
        abort("qumo.rl.synth_env.TrainEnv.is_final: STUB (component C6)")

    def success(self) -> Bool:
        abort("qumo.rl.synth_env.TrainEnv.success: STUB (component C6)")

    def reward(self) -> Float32:
        abort("qumo.rl.synth_env.TrainEnv.reward: STUB (component C6)")

    def step(mut self, a: Int):
        """a >= A with add_perms: clamped to 0, error_code() set, never raised."""
        abort("qumo.rl.synth_env.TrainEnv.step: STUB (component C6)")

    def twists(self) -> Twists:
        """Plain kinds: symmetry.twists; Pauli: empty."""
        abort("qumo.rl.synth_env.TrainEnv.twists: STUB (component C6)")

    def error_code(self) -> Int:
        """0 = ok; set (never raised) inside tasks, MOJO_PORT.md section 3 rule 7."""
        return self.err
