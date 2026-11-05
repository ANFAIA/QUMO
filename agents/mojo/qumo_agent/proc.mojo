"""Child processes with a persistent, bidirectional line channel.

`ChildProcess(program, args)` starts `program` without a shell parsing anything we
do not control: `/bin/sh -c 'exec "$0" "$@" 0<&R 1>&W R<&- W>&-' program args...`.
Only two integers (the pipe fds) go into the script; the program path and every
argument travel as positional parameters, so they are never shell-parsed. The
child's stderr is inherited, so its logs go where ours go.

Pipe ends: `Pipe.fd_in` is the READ end, `Pipe.fd_out` the WRITE end.
"""

from std.os.process import Pipe, Process
from std.time import sleep

from qumo.ai.mcp import LineReader


def _heap(s: String) -> String:
    """A copy of `s` in a NUL-terminated heap buffer.

    Mojo 1.2's Process.run takes each argv C string from a per-element copy. A
    string that is inline (short) or not NUL-terminated in its buffer gets a
    temporary that is gone, or reused for the next element, before exec: the child
    saw `/bin/echo` called with ["ab", "lit"] print "lit lit lit", and a short
    program path arrive as garbage. A heap buffer that already ends in NUL is shared
    by the copies, so its pointer stays valid. Literals are static and unaffected."""
    var out = String(s)
    out.reserve_bytes(max(64, s.byte_length() + 1))
    _ = out.as_c_string_span()  # NUL-terminate the heap buffer in place
    return out^


def spawn(path: String, args: List[String]) raises -> Process:
    """Process.run (no shell; argv[0] is added by it) with the inline-string bug
    worked around for the path and every argument."""
    var a = List[String](capacity=len(args))
    for x in args:
        a.append(_heap(x))
    return Process.run(_heap(path), a)


struct ChildProcess(Movable):
    var proc: Process
    var to_child: Pipe  # parent writes fd_out; child reads fd_in as its stdin
    var from_child: Pipe  # child writes fd_out as its stdout; parent reads fd_in
    var reader: LineReader
    var closed: Bool

    def __init__(out self, program: String, args: List[String]) raises:
        var to_child = Pipe(in_close_on_exec=False, out_close_on_exec=True)
        var from_child = Pipe(in_close_on_exec=True, out_close_on_exec=False)
        var r = to_child.fd_in.value().value
        var w = from_child.fd_out.value().value
        var rs = String(r)
        var ws = String(w)
        var script = String("exec \"$0\" \"$@\" 0<&") + rs + " 1>&" + ws + " " + rs + "<&- " + ws + ">&-"
        var argv: List[String] = ["-c", script, program]
        for a in args:
            argv.append(a)
        self.proc = spawn("/bin/sh", argv)
        # Keep only the parent's ends: the write end towards the child and the read end
        # from it. Closing the others lets EOF propagate both ways and keeps these fds
        # out of any later child.
        to_child.set_output_only()
        from_child.set_input_only()
        var rfd = from_child.fd_in.value().value
        self.to_child = to_child^
        self.from_child = from_child^
        self.reader = LineReader(Int(rfd))
        self.closed = False

    def send_line(mut self, line: String) raises:
        """Write one line (a newline is appended)."""
        if self.closed:
            raise Error("child input is closed")
        var s = line + "\n"
        self.to_child.write_bytes(s.as_bytes())

    def read_line(mut self, mut line: String) raises -> Bool:
        """Next line of the child's stdout without the newline; False at EOF."""
        return self.reader.read_line(line)

    def close_input(mut self) raises:
        """Close the child's stdin (it sees EOF)."""
        if not self.closed:
            self.to_child.set_input_only()
            self.closed = True

    def finish(mut self, grace_s: Float64 = 5.0) raises -> Int:
        """Close stdin, wait up to `grace_s` for the child to exit, kill it otherwise.
        Returns the exit code, or -signal when it was terminated by a signal."""
        self.close_input()
        var waited = 0.0
        while waited < grace_s:
            var st = self.proc.poll()
            if st.exit_code:
                return st.exit_code.value()
            if st.term_signal:
                return -st.term_signal.value()
            sleep(0.02)
            waited += 0.02
        _ = self.proc.kill()
        var st = self.proc.wait()
        if st.exit_code:
            return st.exit_code.value()
        if st.term_signal:
            return -st.term_signal.value()
        return -1

    def pid(self) -> Int:
        return Int(self.proc.child_pid)
