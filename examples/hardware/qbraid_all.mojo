"""List every qBraid device of the account and run a Bell pair on each free one, all in Mojo.

Reproduces: qBraid - list devices and submit a job through the qBraid Runtime API v2
Source: https://docs.qbraid.com/v2/api-reference/rest/post-quantum-jobs (qBraid documentation)
Framework: qBraid - https://github.com/qBraid/qBraid

No Python: the qBraid client (qumo.exec.qbraid_api), the OpenQASM parser and exporter
(qumo.ir.qasm) and the HTTP/TLS stack (qumo.net) are QUMO's own Mojo code.

A device is run only when it is free and runnable: ONLINE, gate-model, accepts qasm2 or
qasm3, has no per-task / per-shot / per-minute price, and qBraid's quote for the job is 0
credits. Everything else (QPUs, simulators billed per minute) is listed but never submitted.

Run:  .venv/bin/mojo run -I . examples/hardware/qbraid_all.mojo        (needs QBRAID_API_KEY)
      add --list to only list the devices
example-meta: {"framework": "qbraid", "tier": "slow", "timeout": 1200, "requires": ["network"], "args": ["--list"]}
"""

from std.sys import argv
from std.time import sleep

from qumo.exec.qbraid_api import (
    QbraidDevice, qbraid_cost_estimate, qbraid_devices, qbraid_job, qbraid_result_counts, qbraid_submit,
)
from qumo.ir.qasm import parse_qasm, to_qasm
from qumo.testing.checks import CheckLog

comptime SHOTS = 200
comptime BELL = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"


def _pad(s: String, n: Int) -> String:
    var out = s
    while out.byte_length() < n:
        out += " "
    return out^


def _has(xs: List[String], s: String) -> Bool:
    for x in xs:
        if x == s:
            return True
    return False


def _price(d: QbraidDevice) -> String:
    if d.per_task < 0 and d.per_shot < 0 and d.per_minute < 0:
        return "no price"
    var out = String()
    if d.per_task > 0:
        out += String(d.per_task) + "/task "
    if d.per_shot > 0:
        out += String(d.per_shot) + "/shot "
    if d.per_minute > 0:
        out += String(d.per_minute) + "/min "
    return out if out.byte_length() > 0 else String("0")


def _free(d: QbraidDevice) -> Bool:
    return d.per_task == 0.0 and d.per_shot == 0.0 and d.per_minute == 0.0


def main() raises:
    var log = CheckLog("qbraid/qbraid_all")
    var only_list = len(argv()) > 1 and String(argv()[1]) == "--list"
    var devices = qbraid_devices()
    _ = log.check("qBraid lists devices for this account", len(devices) > 0, String(len(devices)))
    print(_pad("device QRN", 42), _pad("type", 10), _pad("status", 12), _pad("qubits", 7), _pad("inputs", 22),
          "credits")
    var runnable = List[QbraidDevice]()
    for d in devices:
        if d.status == "RETIRED":
            continue
        var ins = String()
        for x in d.run_input_types:
            ins += x + " "
        print(_pad(d.qrn, 42), _pad(d.device_type, 10), _pad(d.status, 12),
              _pad(String(d.qubits) if d.qubits >= 0 else String("-"), 7), _pad(ins, 22), _price(d))
        var qasm = _has(d.run_input_types, "qasm2") or _has(d.run_input_types, "qasm3")
        if d.status == "ONLINE" and d.paradigm == "gate_model" and qasm and _free(d) and d.device_type == "SIMULATOR":
            runnable.append(d.copy())
    print("\nfree, runnable simulators:", len(runnable))
    if only_list:
        log.finish()
        return
    var qasm3 = to_qasm(parse_qasm(BELL))
    for d in runnable:
        var quote = qbraid_cost_estimate(d.qrn, SHOTS)
        if not quote.available or quote.credits != 0.0:
            print("skip", d.qrn, "(quote is not 0 credits)")
            continue
        var fmt = String("qasm2") if _has(d.run_input_types, "qasm2") else String("qasm3")
        var program = BELL if fmt == "qasm2" else qasm3
        var job: String
        try:
            job = qbraid_submit(d.qrn, fmt, program, SHOTS, "qumo-bell-all")
        except e:
            print("FAIL", d.qrn, "submission:", String(e))
            _ = log.check(d.qrn + " accepts the Bell job", False, String(e))
            continue
        var state = String("QUEUED")
        var raw = String()
        var msg = String()
        for _ in range(100):
            var j = qbraid_job(job)
            state = j.status
            raw = j.raw_status
            msg = j.message
            if state == "DONE" or state == "ERROR" or state == "CANCELLED":
                break
            sleep(3.0)
        if state != "DONE":
            print("FAIL", d.qrn, fmt, "->", raw, msg)
            _ = log.check(d.qrn + " finishes the Bell job", False, raw)
            continue
        var counts = qbraid_result_counts(job)
        var line = String()
        var total = 0
        var good = 0
        for kv in counts.items():
            line += kv.key + ":" + String(kv.value) + " "
            total += kv.value
            if kv.key == "00" or kv.key == "11":
                good += kv.value
        print("OK  ", _pad(d.qrn, 36), fmt, "| counts", line, "| P(00|11) =", Float64(good) / Float64(max(total, 1)))
        _ = log.within(d.qrn + " Bell P(00 or 11)", Float64(good) / Float64(max(total, 1)), 0.9, 1.0)
    log.finish()
