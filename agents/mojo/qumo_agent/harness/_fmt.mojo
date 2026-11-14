"""Python `str.format` for the verbatim templates of texts.mojo (named fields only).

One pass over the template, so a value that itself contains `{name}` is never
substituted again. `{{` and `}}` become `{` and `}`; a field whose name is not given is
kept as written.
"""


def fmt(template: String, keys: List[String], values: List[String]) -> String:
    var bs = template.as_bytes()
    var n = len(bs)
    var out = String()
    var i = 0
    var run = 0
    while i < n:
        var c = Int(bs[i])
        if c == 0x7B:  # {
            if i + 1 < n and Int(bs[i + 1]) == 0x7B:
                out += String(unsafe_from_utf8=bs[run:i])
                out += "{"
                i += 2
                run = i
                continue
            var j = i + 1
            while j < n and Int(bs[j]) != 0x7D and Int(bs[j]) != 0x7B:
                j += 1
            if j < n and Int(bs[j]) == 0x7D:
                var name = String(unsafe_from_utf8=bs[i + 1:j])
                var found = -1
                for k in range(len(keys)):
                    if keys[k] == name:
                        found = k
                        break
                if found >= 0:
                    out += String(unsafe_from_utf8=bs[run:i])
                    out += values[found]
                    i = j + 1
                    run = i
                    continue
            i += 1
            continue
        if c == 0x7D and i + 1 < n and Int(bs[i + 1]) == 0x7D:  # }}
            out += String(unsafe_from_utf8=bs[run:i])
            out += "}"
            i += 2
            run = i
            continue
        i += 1
    out += String(unsafe_from_utf8=bs[run:n])
    return out^


def fmt1(template: String, k1: String, v1: String) -> String:
    var keys: List[String] = [k1]
    var values: List[String] = [v1]
    return fmt(template, keys, values)


def fmt2(template: String, k1: String, v1: String, k2: String, v2: String) -> String:
    var keys: List[String] = [k1, k2]
    var values: List[String] = [v1, v2]
    return fmt(template, keys, values)


def fmt3(template: String, k1: String, v1: String, k2: String, v2: String, k3: String, v3: String) -> String:
    var keys: List[String] = [k1, k2, k3]
    var values: List[String] = [v1, v2, v3]
    return fmt(template, keys, values)
