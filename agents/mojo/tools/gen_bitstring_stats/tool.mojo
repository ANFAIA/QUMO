"""Bitstring statistics (gen_bitstring_stats): shots, Shannon entropy and parity.

Reference generated QUMO tool built on qumo_agent.tool_sdk. It runs with an empty
environment in a temporary directory, and a human reviews it before any agent can
call it. New tools follow the same shape: check every argument, return fail(...)
for bad input, and return a small deterministic JSON object.
"""

from std.math import log

from qumo.ai.json import JsonValue, JsonWriter, parse_json
from qumo_agent.tool_sdk import ToolInfo, ToolOutput, fail, ok_json, arg_string, serve_tool

comptime NAME = "gen_bitstring_stats"
comptime VERSION = "0.1.0"
comptime DESCRIPTION = "Counts, Shannon entropy and parity of a counts dictionary. Input: counts, a JSON object string mapping bitstrings (Qiskit style, spaces between registers allowed) to non-negative integer counts, e.g. {\"00\": 500, \"11\": 500}. Output: a JSON object with shots, outcomes (distinct bitstrings seen), width (bits), entropy_bits (Shannon entropy of the measured distribution, in bits), parity_even and parity_odd (probability of an even or odd number of 1s) and most_frequent (the most frequent bitstring; ties go to the smallest)."
comptime SCHEMA = "{\"type\":\"object\",\"properties\":{\"counts\":{\"type\":\"string\",\"description\":\"JSON object mapping bitstrings to counts, e.g. {\\\"00\\\": 500, \\\"11\\\": 500}.\"}},\"required\":[\"counts\"],\"additionalProperties\":false}"


def clean_bitstring(key: String) -> String:
    """The key without spaces, or "" when it has a character other than 0, 1 or space."""
    var out = String()
    for b in key.as_bytes():
        var c = Int(b)
        if c == 0x20:
            continue
        if c != 0x30 and c != 0x31:
            return String()
        out += "1" if c == 0x31 else "0"
    return out^


def ones(bits: String) -> Int:
    var n = 0
    for b in bits.as_bytes():
        if Int(b) == 0x31:
            n += 1
    return n


def handle(args: JsonValue) raises -> ToolOutput:
    var counts_text = arg_string(args, "counts", "")
    if counts_text.byte_length() == 0:
        return fail("counts is required: a JSON object such as {\"00\": 500, \"11\": 500}")
    var counts: JsonValue
    try:
        counts = parse_json(counts_text)
    except:
        return fail("counts is not valid JSON: pass an object such as {\"00\": 500, \"11\": 500}")
    if not counts.is_object():
        return fail("counts must be a JSON object, got " + counts.kind_name())
    if len(counts) == 0:
        return fail("counts is empty")

    var keys = List[String]()
    var values = List[Int]()
    var width = -1
    var shots = 0
    for i in range(len(counts)):
        var key = counts.key_at(i)
        var bits = clean_bitstring(key)
        if bits.byte_length() == 0:
            return fail("bitstring '" + key + "' must contain only 0, 1 and spaces")
        if width < 0:
            width = bits.byte_length()
        elif bits.byte_length() != width:
            return fail("bitstrings have different widths: '" + key + "' is not " + String(width) + " bits")
        var v = counts.at(i)
        if not v.is_number():
            return fail("count of '" + key + "' must be a non-negative integer")
        var n: Int
        try:
            n = v.as_int()
        except:
            return fail("count of '" + key + "' must be a non-negative integer")
        if n < 0:
            return fail("count of '" + key + "' must be a non-negative integer")
        for j in range(len(keys)):
            if keys[j] == bits:
                return fail("bitstring '" + bits + "' appears twice")
        keys.append(bits)
        values.append(n)
        shots += n
    if shots == 0:
        return fail("counts add up to 0 shots")

    var entropy = 0.0
    var even = 0
    var outcomes = 0
    var best = -1
    for i in range(len(keys)):
        var n = values[i]
        if n == 0:
            continue
        outcomes += 1
        var p = Float64(n) / Float64(shots)
        entropy -= p * log(p) / log(2.0)   # std.math.log2 is off by ~1e-10 in Mojo 1.2
        if ones(keys[i]) % 2 == 0:
            even += n
        if best < 0 or n > values[best] or (n == values[best] and keys[i] < keys[best]):
            best = i
    if entropy == 0.0:
        entropy = 0.0   # never print -0.0

    var w = JsonWriter()
    w.begin_object()
    w.kv_int("shots", shots)
    w.kv_int("outcomes", outcomes)
    w.kv_int("width", width)
    w.kv_float("entropy_bits", entropy)
    w.kv_float("parity_even", Float64(even) / Float64(shots))
    w.kv_float("parity_odd", Float64(shots - even) / Float64(shots))
    w.kv_string("most_frequent", keys[best])
    w.end_object()
    return ok_json(w.finish())


def main() raises:
    serve_tool(ToolInfo(NAME, "Bitstring statistics", DESCRIPTION, SCHEMA, VERSION), handle)
