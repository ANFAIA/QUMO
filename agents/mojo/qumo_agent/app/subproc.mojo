"""Parallel subagents as child processes (DESIGN.md 2.10 `run_subagent_children`, X2, B.7).

When one turn has two or more `task` calls, each subagent runs in its own
`qumo-agent subagent` child: one request line on its stdin, one result line on its stdout,
logs on the inherited stderr (so a chatty child never blocks on an undrained pipe). At most
`parallel` children run at once; the next request starts when one finishes; the results
come back in request order.

The parent writes the whole request and closes the child's stdin before it reads, and the
child reads its whole stdin before it writes anything, so the two cannot deadlock. The
stdout pipes of all running children are polled together; a child is reaped only after its
stdout reached EOF. A child whose output grows past `SUBAGENT_RESULT_MAX_BYTES` is killed.

A failed child is not an exception: its `SubagentResult` has `ok = False` and an `error`
text (`exit <code>`, `result larger than ... bytes`, `malformed result: ...`, or the error
the child reported), and the harness raises `QA-E120: subagent <type> failed: <error>`
(X11).
"""

from std.memory import ArcPointer

from qumo_agent.app.options import AppOptions
from qumo_agent.core.stats import zero_stats
from qumo_agent.harness.env import SubagentRequest, SubagentResult, subagent_result_from_json
from qumo_agent.native.libc import (
    c_poll, buf_addr, store_u32, store_u16, load_u16, errno, strerror_text, SIGKILL, EINTR,
)
from qumo_agent.native.spawn import (
    Child, SpawnSpec, spawn, harness_child_env, STDIO_PIPE, STDIO_INHERIT,
)
from qumo_agent.util.pyfmt import cp_head
from qumo_agent.util.redact import redact_all, secret_env_names

comptime SUBAGENT_RESULT_MAX_BYTES = 268435456     # 256 MiB, one result line
comptime _POLL_SLICE_MS = 1000
comptime _STATE_WAITING = 0
comptime _STATE_RUNNING = 1
comptime _STATE_DONE = 2


def _failure(error: String) -> SubagentResult:
    return SubagentResult(False, String(), False, String(), String(), zero_stats(), error)


def last_nonempty_line(buf: List[UInt8]) -> String:
    """The last line of `buf` that holds something besides white space ("" if none)."""
    var end = len(buf)
    while end > 0:
        var c = Int(buf[end - 1])
        if c == 0x0A or c == 0x0D or c == 0x20 or c == 0x09:
            end -= 1
        else:
            break
    if end == 0:
        return String()
    var start = end
    while start > 0 and Int(buf[start - 1]) != 0x0A:
        start -= 1
    return String(from_utf8_lossy=Span(buf)[start:end])


def _finish(req: SubagentRequest, buf: List[UInt8], code: Int, secrets: List[String]) -> SubagentResult:
    """The result of a child that reached EOF with exit status `code`."""
    var line = last_nonempty_line(buf)
    if line.byte_length() == 0:
        return _failure(redact_all("exit " + String(code), secrets))
    try:
        var r = subagent_result_from_json(line)
        if not r.ok and r.error.byte_length() == 0:
            r.error = "exit " + String(code)
        r.error = redact_all(r.error, secrets)
        return r^
    except:
        return _failure(redact_all("malformed result: " + cp_head(line, 200), secrets))


def run_subagent_children(opts: AppOptions, var reqs: List[SubagentRequest], parallel: Int) raises -> List[SubagentResult]:
    var n = len(reqs)
    var limit = parallel if parallel > 1 else 1
    var secrets = secret_env_names(opts.api_key_env)
    var env = harness_child_env(opts.api_key_env)
    var results = List[SubagentResult]()
    var children = List[Optional[ArcPointer[Child]]]()
    var buffers = List[List[UInt8]]()
    var state = List[Int]()
    for _ in range(n):
        results.append(_failure("did not run"))
        children.append(Optional[ArcPointer[Child]](None))
        buffers.append(List[UInt8]())
        state.append(_STATE_WAITING)
    var flags = opts.child_flags()
    var next_start = 0
    var running = 0
    var done = 0
    while done < n:
        # Start children while there is room.
        while running < limit and next_start < n:
            var i = next_start
            next_start += 1
            var argv = List[String]()
            argv.append(opts.self_binary)
            argv.append("subagent")
            for f in flags:
                argv.append(f)
            try:
                var child = spawn(SpawnSpec(opts.self_binary, argv^, env.copy(), String(), STDIO_PIPE, STDIO_PIPE,
                                            STDIO_INHERIT, True))
                var line = reqs[i].to_json() + "\n"
                try:
                    child.write_all(line.as_bytes())
                except:
                    pass        # the child died early: its stdout and exit code say why
                child.close_stdin()
                children[i] = Optional[ArcPointer[Child]](ArcPointer(child^))
                state[i] = _STATE_RUNNING
                running += 1
            except e:
                results[i] = _failure(redact_all(String(e), secrets))
                state[i] = _STATE_DONE
                done += 1
        if running == 0:
            continue
        # Poll the stdout of every running child together.
        var idx = List[Int]()
        for i in range(n):
            if state[i] == _STATE_RUNNING:
                idx.append(i)
        var pfds = List[UInt8](length=8 * len(idx), fill=0)
        var base = buf_addr(pfds)
        for m in range(len(idx)):
            store_u32(base + 8 * m, children[idx[m]].value()[].stdout_fd)
            store_u16(base + 8 * m + 4, 0x1)     # POLLIN
            store_u16(base + 8 * m + 6, 0)
        var rc = c_poll(base, len(idx), _POLL_SLICE_MS)
        if rc < 0:
            var e = errno()
            _ = len(pfds)
            if e == EINTR:
                continue
            raise Error("QA-E160: poll failed: [Errno " + String(e) + "] " + strerror_text(e))
        if rc == 0:
            _ = len(pfds)
            continue
        var ready = List[Int]()
        for m in range(len(idx)):
            if load_u16(base + 8 * m + 6) != 0:
                ready.append(idx[m])
        _ = len(pfds)
        for j in ready:
            var got = children[j].value()[].read_available(children[j].value()[].stdout_fd, buffers[j], 0)
            if got == 0:
                var code = children[j].value()[].wait()
                results[j] = _finish(reqs[j], buffers[j], code, secrets)
                children[j] = Optional[ArcPointer[Child]](None)
                buffers[j] = List[UInt8]()
                state[j] = _STATE_DONE
                running -= 1
                done += 1
            elif got > 0 and len(buffers[j]) > SUBAGENT_RESULT_MAX_BYTES:
                children[j].value()[].signal_group(SIGKILL)
                _ = children[j].value()[].wait()
                results[j] = _failure("result larger than " + String(SUBAGENT_RESULT_MAX_BYTES) + " bytes")
                children[j] = Optional[ArcPointer[Child]](None)
                buffers[j] = List[UInt8]()
                state[j] = _STATE_DONE
                running -= 1
                done += 1
    return results^
