"""Router teacher records (328 bytes each): reader that concatenates files and applies the behaviour-cloning filter.

Record layout (little endian, written by `route_tool gen`; MOJO_PORT.md 8.1, repo.md 4.2):

    offset  size  field
    0       256   int16 obs[128]     obs[d + 8k]: layer d in 0..7, active-swap slot k in 0..15
    256     2     int16 A            number of valid active swaps (1..16)
    258     2     int16 act          behaviour action slot
    260     2     int16 job          job index (clamped to int16)
    262     2     int16 tb           teacher argmin slot
    264     64    float32 Q[16]      teacher cost per slot (CX units); 1e9 for invalid slots (k >= A)

`read_teacher_files` concatenates the files in the given order and keeps exactly the rows the previous trainer kept:
`spread = nanmax(Q[:A]) - nanmin(Q[:A]) > 1e-6` evaluated in f32 (written as keep-if-greater, so `A = 0`, whose
spread is NaN, and `A = 1`, whose spread is 0, are both dropped). Q is clipped with `min(Q, 1e6)` after the spread
is taken (the spread uses the raw values, as the reference does). Only the sign of an observation is ever used by
the router, so `obs` keeps the int16 value saturated to the Int8 range (sign and small magnitudes survive).

Errors: QM-E646 on a file whose size is not a multiple of 328 bytes, an unreadable file, an `A` outside 0..16, or a
valid-slot cost that is NaN or -inf (the reference would silently train on NaN targets).
"""

from std.memory import bitcast

comptime REC_BYTES = 328
comptime TEACHER_OBS = 128
comptime TEACHER_SLOTS = 16
comptime OFF_A = 256
comptime OFF_ACT = 258
comptime OFF_JOB = 260
comptime OFF_TB = 262
comptime OFF_Q = 264
comptime SPREAD_MIN: Float32 = 1e-6
comptime Q_CLIP: Float32 = 1e6


struct TeacherData(Movable):
    """Filtered training rows in struct-of-arrays form (`n` rows kept out of `raw` records read)."""

    var n: Int
    var raw: Int  # records read before the filter (added to the section 5 fields)
    var obs: List[Int8]  # [n, 128], int16 saturated to Int8
    var A: List[Int16]  # [n]
    var act: List[Int16]
    var job: List[Int16]
    var tb: List[Int16]
    var q: List[Float32]  # [n, 16], clipped at 1e6 (slots k >= A hold whatever the file had, clipped)

    def __init__(out self):
        self.n = 0
        self.raw = 0
        self.obs = List[Int8]()
        self.A = List[Int16]()
        self.act = List[Int16]()
        self.job = List[Int16]()
        self.tb = List[Int16]()
        self.q = List[Float32]()


def _e646(what: String, msg: String) -> Error:
    return Error("QM-E646: teacher data '" + what + "' is malformed: " + msg + ". Hint: files must be `route_tool gen` output (328-byte records).")


@always_inline
def _i16_at(buf: List[UInt8], off: Int) -> Int:
    var u = Int(buf[off]) | (Int(buf[off + 1]) << 8)
    if u >= 32768:
        return u - 65536
    return u


@always_inline
def _f32_at(buf: List[UInt8], off: Int) -> Float32:
    var u = (
        UInt32(buf[off])
        | (UInt32(buf[off + 1]) << 8)
        | (UInt32(buf[off + 2]) << 16)
        | (UInt32(buf[off + 3]) << 24)
    )
    return bitcast[DType.float32](u)


def _sat_i8(v: Int) -> Int8:
    return Int8(max(-128, min(127, v)))


def parse_teacher_bytes(buf: List[UInt8], mut td: TeacherData, what: String, apply_filter: Bool = True) raises:
    """Appends the records of `buf` (a whole number of 328-byte records) to `td`; `what` names the source in errors.
    With `apply_filter` False every record is kept unchanged (Q is still clipped), which tests use to see raw rows."""
    if len(buf) % REC_BYTES != 0:
        raise _e646(what, String(len(buf)) + " bytes is not a multiple of " + String(REC_BYTES))
    var recs = len(buf) // REC_BYTES
    td.raw += recs
    td.obs.reserve(len(td.obs) + recs * TEACHER_OBS)
    td.q.reserve(len(td.q) + recs * TEACHER_SLOTS)
    for r in range(recs):
        var base = r * REC_BYTES
        var a = _i16_at(buf, base + OFF_A)
        if a < 0 or a > TEACHER_SLOTS:
            raise _e646(what, "record " + String(r) + " has A = " + String(a) + " (expected 0..16)")
        # spread of the valid slots in f32, exactly the reference's nanmax - nanmin (raw values, before the clip)
        var mx: Float32 = 0.0
        var mn: Float32 = 0.0
        for k in range(a):
            var qv = _f32_at(buf, base + OFF_Q + 4 * k)
            if qv != qv:
                raise _e646(what, "record " + String(r) + " has a NaN cost in valid slot " + String(k))
            if qv < -Float32.MAX_FINITE:
                raise _e646(what, "record " + String(r) + " has a -inf cost in valid slot " + String(k))
            if k == 0:
                mx = qv
                mn = qv
            else:
                if qv > mx:
                    mx = qv
                if qv < mn:
                    mn = qv
        var keep = True
        if apply_filter:
            if a < 2:
                keep = False  # A = 0: NaN spread; A = 1: spread 0
            else:
                keep = (mx - mn) > SPREAD_MIN
        if not keep:
            continue
        for i in range(TEACHER_OBS):
            td.obs.append(_sat_i8(_i16_at(buf, base + 2 * i)))
        td.A.append(Int16(a))
        td.act.append(Int16(_i16_at(buf, base + OFF_ACT)))
        td.job.append(Int16(_i16_at(buf, base + OFF_JOB)))
        td.tb.append(Int16(_i16_at(buf, base + OFF_TB)))
        for k in range(TEACHER_SLOTS):
            var qv = _f32_at(buf, base + OFF_Q + 4 * k)
            if qv > Q_CLIP:
                qv = Q_CLIP
            td.q.append(qv)
        td.n += 1


def read_teacher_files(paths: List[String]) raises -> TeacherData:
    """Reads and concatenates the teacher files in order and applies the BC row filter (see the module docstring).
    QM-E646 on a malformed or unreadable file."""
    var td = TeacherData()
    for p in paths:
        var buf: List[UInt8]
        try:
            with open(p, "r") as f:
                buf = f.read_bytes()
        except e:
            raise _e646(p, "cannot read the file (" + String(e) + ")")
        parse_teacher_bytes(buf, td, p)
    return td^


def _put_i16(mut b: List[UInt8], v: Int):
    var x = max(-32768, min(32767, v))
    var u = UInt16(x & 0xFFFF)
    b.append(UInt8(u & 0xFF))
    b.append(UInt8(u >> 8))


def encode_teacher_record(
    mut b: List[UInt8], obs: List[Int], A: Int, act: Int, job: Int, tb: Int, q: List[Float32]
):
    """Appends one 328-byte record to `b` (int16 fields saturate, as `route_tool gen` does). `obs` has up to 128
    entries (missing ones are 0), `q` up to 16 (missing ones are written as 1e9, the invalid-slot value)."""
    for i in range(TEACHER_OBS):
        _put_i16(b, obs[i] if i < len(obs) else 0)
    _put_i16(b, A)
    _put_i16(b, act)
    _put_i16(b, job)
    _put_i16(b, tb)
    for k in range(TEACHER_SLOTS):
        var qv: Float32 = q[k] if k < len(q) else Float32(1e9)
        var u = bitcast[DType.uint32](qv)
        b.append(UInt8(u & 0xFF))
        b.append(UInt8((u >> 8) & 0xFF))
        b.append(UInt8((u >> 16) & 0xFF))
        b.append(UInt8(u >> 24))
