"""Small text helpers shared by `Target` and the compiler reports: `basis_str` and `fmt_num`."""

from qumo.ir.gates import op_name


def basis_str(kinds: List[Int]) -> String:
    if len(kinds) == 0:
        return "any"
    var s = String()
    for i in range(len(kinds)):
        if i > 0:
            s += ","
        s += op_name(kinds[i])
    return s


def fmt_num(x: Float64, digits: Int = 3) -> String:
    """Compact float formatting for reports (agents parse these)."""
    if x != x:
        return "nan"
    if x == 0.0:
        return "0"
    var ax = abs(x)
    var sign = String("-") if x < 0 else String("")
    if ax >= 1e-3 and ax < 1e9:
        var scale = 1.0
        for _ in range(digits):
            scale *= 10.0
        var v = Int(ax * scale + 0.5)
        var ip = v // Int(scale)
        var fp = v - ip * Int(scale)
        var s = sign + String(ip)
        if fp > 0:
            var fs = String(fp)
            var pad = digits - fs.byte_length()
            var z = String()
            for _ in range(pad):
                z += "0"
            fs = z + fs
            # trim trailing zeros
            var bs = fs.as_bytes()
            var end = len(bs)
            while end > 0 and Int(bs[end - 1]) == ord("0"):
                end -= 1
            s += "." + String(unsafe_from_utf8=bs[0:end])
        return s
    # scientific
    var e = 0
    var m = ax
    while m >= 10.0:
        m /= 10.0
        e += 1
    while m < 1.0:
        m *= 10.0
        e -= 1
    var mi = Int(m * 100.0 + 0.5)
    if mi >= 1000:
        mi = mi // 10
        e += 1
    var ms = String(mi // 100)
    var frac = mi % 100
    if frac != 0:
        if frac % 10 == 0:
            ms += "." + String(frac // 10)
        else:
            ms += "." + (String("0") if frac < 10 else String("")) + String(frac)
    return sign + ms + "e" + String(e)
