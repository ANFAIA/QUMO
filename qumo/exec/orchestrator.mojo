"""Heterogeneous orchestration: schedule quantum jobs over a pool of devices.

Best-of synthesis:
  * CUDA-Q `mqpu` + `*_async`: many virtual QPUs (GPUs, CPUs, real QPUs)
    behind one API; jobs fan out across them.
  * Braket tasks / ProgramSets + `Tracker`: every job has a latency and a
    monetary cost model (queue, per-task and per-shot fees) so the planner can
    trade speed for money *before* submitting.
  * HPC scheduling: HEFT (Heterogeneous Earliest Finish Time, Topcuoglu et
    al. 2002) list scheduling over a dependency DAG — the standard algorithm
    for CPU/GPU/accelerator task graphs, here extended with capability
    constraints (max qubits, Clifford-only devices).
  * Explainability for agents: `Schedule.explain()` states why each job went
    where it went.

Execution runs dependency "waves"; independent small jobs inside a wave run in
parallel threads. Jobs mapped to remote QPUs are executed on the local
emulator and flagged (the OpenQASM 3 payload is what would be submitted).
Jobs mapped to a GPU device run on the GPU state vector (`qumo.gpu`) when the
build has `-D QUMO_GPU` and a device is usable; otherwise they run on the CPU
and `JobResult.backend` says so ("... (gpu fallback: <reason>)").
"""

from std.time import perf_counter_ns
from std.sys.defines import is_defined
from max.algorithm import parallelize

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import sample, expectation, SampleResult
from qumo.exec.dispatch import sample_auto, frame_eligible_gpu, sample_frame_gpu
from qumo.gpu.device import gpu_available
from qumo.gpu.statevector_gpu import sample_gpu, evolve_gpu

comptime INF = 1e300


struct DeviceKind:
    comptime CPU_SV = 0
    comptime GPU_SV = 1
    comptime STABILIZER = 2
    comptime QPU = 3


def device_kind_name(k: Int) -> String:
    if k == DeviceKind.CPU_SV: return "cpu-statevector"
    if k == DeviceKind.GPU_SV: return "gpu-statevector"
    if k == DeviceKind.STABILIZER: return "stabilizer"
    return "qpu"


@fieldwise_init
struct Device(Copyable, Writable):
    var name: String
    var kind: Int
    var max_qubits: Int
    var amp_ops_per_s: Float64  # simulators: gate x amplitude updates per second
    var overhead_s: Float64  # per job: launch, queue, network
    var shot_time_s: Float64  # QPUs: wall time per shot
    var cost_per_task: Float64
    var cost_per_shot: Float64
    var clifford_only: Bool

    @staticmethod
    def cpu(name: String = "cpu0", max_qubits: Int = 30) -> Device:
        return Device(name, DeviceKind.CPU_SV, max_qubits, 2.5e9, 1e-5, 0.0, 0.0, 0.0, False)

    @staticmethod
    def gpu(name: String = "gpu0", max_qubits: Int = 33) -> Device:
        return Device(name, DeviceKind.GPU_SV, max_qubits, 6e10, 2e-3, 0.0, 0.0, 0.0, False)

    @staticmethod
    def stabilizer(name: String = "tableau0") -> Device:
        return Device(name, DeviceKind.STABILIZER, 100000, 5e9, 1e-5, 0.0, 0.0, 0.0, True)

    @staticmethod
    def qpu(name: String, max_qubits: Int, queue_s: Float64, shot_time_s: Float64,
            cost_per_task: Float64, cost_per_shot: Float64) -> Device:
        return Device(name, DeviceKind.QPU, max_qubits, 0.0, queue_s, shot_time_s, cost_per_task, cost_per_shot, False)

    def write_to(self, mut w: Some[Writer]):
        w.write(self.name, "<", device_kind_name(self.kind), ", max_qubits=", self.max_qubits, ">")


struct JobKind:
    comptime SAMPLE = 0
    comptime ESTIMATE = 1


struct Job(Copyable):
    var name: String
    var kind: Int
    var circuit: Circuit
    var observable: PauliSum
    var params: List[Float64]
    var shots: Int
    var deps: List[Int]
    var seed: UInt64

    def __init__(out self, name: String, kind: Int, circuit: Circuit, observable: PauliSum,
                 params: List[Float64], shots: Int, deps: List[Int], seed: UInt64):
        self.name = name
        self.kind = kind
        self.circuit = circuit.copy()
        self.observable = observable.copy()
        self.params = params.copy()
        self.shots = shots
        self.deps = deps.copy()
        self.seed = seed


struct JobResult(Copyable):
    var device: String
    var emulated: Bool
    var expectation: Float64
    var samples: SampleResult
    var seconds: Float64
    var ok: Bool
    var error: String
    var backend: String  # simulator that actually ran, incl. GPU fallback notes

    def __init__(out self):
        self.device = ""
        self.emulated = False
        self.expectation = 0.0
        self.samples = SampleResult(0, 0, "")
        self.seconds = 0.0
        self.ok = False
        self.error = ""
        self.backend = ""


def _run_job_cpu(job: Job, mut r: JobResult) raises:
    if job.kind == JobKind.SAMPLE:
        r.samples = sample_auto(job.circuit, job.shots, job.seed, job.params)
        r.backend = r.samples.method
    else:
        r.expectation = expectation(job.circuit, job.observable, job.params)
        r.backend = "statevector:exact"


def _run_job_gpu(job: Job, mut r: JobResult) raises:
    """A job HEFT placed on a GPU device: GPU state vector when compiled and
    available, GPU Pauli frames for Clifford sample jobs without feed-forward
    (noisy or not), otherwise the CPU, with the reason appended to `backend`."""
    var reason: String
    comptime if is_defined["QUMO_GPU"]():
        var ok: Bool
        try:
            ok = gpu_available()
        except:
            ok = False
        if not ok:
            reason = "no usable GPU device"
        elif job.kind == JobKind.ESTIMATE:
            var sv = evolve_gpu(job.circuit, job.params)
            r.expectation = sv.expectation(job.observable)
            r.backend = sv.method_name()
            return
        elif len(job.params) == 0 and frame_eligible_gpu(job.circuit):
            r.samples = sample_frame_gpu(job.circuit, job.shots, job.seed)
            r.backend = r.samples.method
            return
        elif job.circuit.is_dynamic() or job.circuit.has_noise():
            reason = "dynamic or noisy non-Clifford circuit (GPU state-vector sampler is static-only)"
        elif job.circuit.num_params() == 0 and job.circuit.is_clifford():
            reason = "Clifford circuit with feed-forward (stabilizer tableau)"
        else:
            r.samples = sample_gpu(job.circuit, job.shots, job.seed, job.params)
            r.backend = r.samples.method
            return
    else:
        reason = "QUMO built without -D QUMO_GPU"
    _run_job_cpu(job, r)
    r.backend += " (gpu fallback: " + reason + ")"


struct Schedule(Copyable, Writable):
    var device_of: List[Int]
    var start: List[Float64]
    var finish: List[Float64]
    var cost: List[Float64]
    var order: List[Int]
    var makespan: Float64
    var total_cost: Float64
    var notes: List[String]

    def __init__(out self, n: Int):
        self.device_of = List[Int](length=n, fill=-1)
        self.start = List[Float64](length=n, fill=0.0)
        self.finish = List[Float64](length=n, fill=0.0)
        self.cost = List[Float64](length=n, fill=0.0)
        self.order = List[Int]()
        self.makespan = 0.0
        self.total_cost = 0.0
        self.notes = List[String]()

    def write_to(self, mut w: Some[Writer]):
        w.write("Schedule(makespan=", self.makespan, " s, cost=$", self.total_cost, ")")
        for line in self.notes:
            w.write("\n  ", line)


def estimate_job_seconds(job: Job, dev: Device) -> Float64:
    """Cost model; INF when the device cannot run the job."""
    var c = job.circuit.copy()
    var n = c.num_qubits
    if n > dev.max_qubits:
        return INF
    if dev.clifford_only and not c.is_clifford():
        return INF
    var gates = Float64(max(c.num_unitary_gates(), 1))
    if dev.kind == DeviceKind.QPU:
        var shots = Float64(max(job.shots, 1000 if job.kind == JobKind.ESTIMATE else 1))
        return dev.overhead_s + shots * dev.shot_time_s
    if dev.kind == DeviceKind.STABILIZER:
        var work = gates * Float64(n) * Float64(n) / 64.0 + Float64(max(job.shots, 1)) * gates * Float64(n) / 64.0
        return dev.overhead_s + work / dev.amp_ops_per_s
    var amps = Float64(1 << min(n, 62))
    return dev.overhead_s + gates * amps / dev.amp_ops_per_s + Float64(job.shots) * 1e-8


def estimate_job_cost(job: Job, dev: Device) -> Float64:
    if dev.kind != DeviceKind.QPU:
        return 0.0
    var shots = Float64(max(job.shots, 1000 if job.kind == JobKind.ESTIMATE else 1))
    return dev.cost_per_task + shots * dev.cost_per_shot


struct Workflow(Movable):
    var jobs: List[Job]

    def __init__(out self):
        self.jobs = List[Job]()

    def add_sample(mut self, name: String, circuit: Circuit, shots: Int,
                   params: List[Float64] = List[Float64](), deps: List[Int] = List[Int](),
                   seed: UInt64 = 1234) -> Int:
        self.jobs.append(Job(name, JobKind.SAMPLE, circuit, PauliSum(), params, shots, deps, seed))
        return len(self.jobs) - 1

    def add_estimate(mut self, name: String, circuit: Circuit, observable: PauliSum,
                     params: List[Float64] = List[Float64](), deps: List[Int] = List[Int](),
                     seed: UInt64 = 1234) -> Int:
        self.jobs.append(Job(name, JobKind.ESTIMATE, circuit, observable, params, 0, deps, seed))
        return len(self.jobs) - 1

    def plan(self, devices: List[Device], cost_weight: Float64 = 0.0) raises -> Schedule:
        """HEFT list scheduling. `cost_weight` (s per $) trades time for money."""
        var n = len(self.jobs)
        var nd = len(devices)
        if nd == 0:
            raise Error("QM-E460: no devices in the pool. Hint: add Device.cpu() at least.")
        var w = List[List[Float64]]()
        var avg = List[Float64](length=n, fill=0.0)
        for i in range(n):
            var row = List[Float64]()
            var s = 0.0
            var cnt = 0
            for d in range(nd):
                var t = estimate_job_seconds(self.jobs[i], devices[d])
                row.append(t)
                if t < INF:
                    s += t
                    cnt += 1
            if cnt == 0:
                raise Error("QM-E461: job '" + self.jobs[i].name + "' (" + String(self.jobs[i].circuit.num_qubits) + " qubits) fits no device in the pool. Hint: add a device with more qubits or a stabilizer device for Clifford circuits.")
            avg[i] = s / Float64(cnt)
            w.append(row^)
            for dep in self.jobs[i].deps:
                if dep < 0 or dep >= i:
                    raise Error("QM-E462: job '" + self.jobs[i].name + "' depends on job " + String(dep) + " which is not an earlier job.")
        # upward rank (succ lists)
        var succ = List[List[Int]]()
        for _ in range(n):
            succ.append(List[Int]())
        for i in range(n):
            for dep in self.jobs[i].deps:
                succ[dep].append(i)
        var rank = List[Float64](length=n, fill=0.0)
        var i = n - 1
        while i >= 0:
            var m = 0.0
            for s in succ[i]:
                m = max(m, rank[s])
            rank[i] = avg[i] + m
            i -= 1
        var order = List[Int]()
        for k in range(n):
            order.append(k)
        # sort by rank desc; ties by index (stable, deterministic) - insertion sort (n small)
        for a in range(1, n):
            var key = order[a]
            var b = a - 1
            while b >= 0 and (rank[order[b]] < rank[key] or (rank[order[b]] == rank[key] and order[b] > key)):
                order[b + 1] = order[b]
                b -= 1
            order[b + 1] = key
        var sch = Schedule(n)
        var avail = List[Float64](length=nd, fill=0.0)
        for j in order:
            var ready = 0.0
            for dep in self.jobs[j].deps:
                ready = max(ready, sch.finish[dep])
            var best = -1
            var best_score = INF
            var best_finish = INF
            for d in range(nd):
                if w[j][d] >= INF:
                    continue
                var st = max(ready, avail[d])
                var fin = st + w[j][d]
                var score = fin + cost_weight * estimate_job_cost(self.jobs[j], devices[d])
                if score < best_score:
                    best_score = score
                    best = d
                    best_finish = fin
            sch.device_of[j] = best
            sch.start[j] = best_finish - w[j][best]
            sch.finish[j] = best_finish
            sch.cost[j] = estimate_job_cost(self.jobs[j], devices[best])
            avail[best] = best_finish
            sch.order.append(j)
            sch.total_cost += sch.cost[j]
            sch.makespan = max(sch.makespan, best_finish)
            var why = String("job ") + String(j) + " '" + self.jobs[j].name + "' (" + String(self.jobs[j].circuit.num_qubits) + "q) -> " + devices[best].name
            why += " [start " + String(sch.start[j]) + " s, est " + String(w[j][best]) + " s"
            if sch.cost[j] > 0:
                why += ", $" + String(sch.cost[j])
            why += "; rank " + String(rank[j]) + "]"
            sch.notes.append(why)
        return sch^

    def run(self, devices: List[Device], cost_weight: Float64 = 0.0) raises -> List[JobResult]:
        """Plan, then execute dependency waves (parallel inside a wave)."""
        var sch = self.plan(devices, cost_weight)
        var n = len(self.jobs)
        var results = List[JobResult]()
        for _ in range(n):
            results.append(JobResult())
        var done = List[Bool](length=n, fill=False)
        var remaining = n
        var rp = results.unsafe_ptr().as_unsafe_any_origin()
        var jp = self.jobs.unsafe_ptr().as_unsafe_any_origin()
        var dev_names = List[String]()
        var dev_is_qpu = List[Bool]()
        var dev_is_gpu = List[Bool]()
        for j in range(n):
            dev_names.append(devices[sch.device_of[j]].name)
            dev_is_qpu.append(devices[sch.device_of[j]].kind == DeviceKind.QPU)
            dev_is_gpu.append(devices[sch.device_of[j]].kind == DeviceKind.GPU_SV)
        var np_ = dev_names.unsafe_ptr().as_unsafe_any_origin()
        var qp = dev_is_qpu.unsafe_ptr().as_unsafe_any_origin()
        var gp = dev_is_gpu.unsafe_ptr().as_unsafe_any_origin()
        while remaining > 0:
            var wave = List[Int]()
            var small = True
            for j in range(n):
                if done[j]:
                    continue
                var ready = True
                for dep in self.jobs[j].deps:
                    if not done[dep]:
                        ready = False
                if ready:
                    wave.append(j)
                    if self.jobs[j].circuit.num_qubits > 14 or dev_is_gpu[j]:
                        small = False  # GPU jobs share one device: run them in order
            var wp = wave.unsafe_ptr().as_unsafe_any_origin()

            def exec_one(k: Int) {imm wp, imm rp, imm jp, imm np_, imm qp, imm gp}:
                var j = wp[unsafe_offset=k]
                ref job = jp[unsafe_offset=j]
                var r = JobResult()
                r.device = np_[unsafe_offset=j]
                r.emulated = qp[unsafe_offset=j]
                var t0 = perf_counter_ns()
                try:
                    if gp[unsafe_offset=j]:
                        _run_job_gpu(job, r)
                    else:
                        _run_job_cpu(job, r)
                    r.ok = True
                except e:
                    r.error = String(e)
                r.seconds = Float64(perf_counter_ns() - t0) / 1e9
                rp[unsafe_offset=j] = r^

            if small and len(wave) > 1:
                parallelize(exec_one, len(wave))
            else:
                for k in range(len(wave)):
                    exec_one(k)
            for j in wave:
                done[j] = True
                remaining -= 1
        # keep buffers referenced by raw pointers alive past the parallel calls
        # (Mojo destroys values ASAP after their last syntactic use)
        _ = len(dev_names)
        _ = len(dev_is_qpu)
        _ = len(dev_is_gpu)
        return results^
