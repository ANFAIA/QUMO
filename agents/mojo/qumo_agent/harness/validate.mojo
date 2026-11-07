"""Portions ported from deepagents 0.7.32 and langchain 1.4.8 / langchain-core 1.7.4
(Copyright (c) LangChain, Inc., MIT License; see agents/mojo/LICENSES/langchain-MIT.txt).

Tool-argument validation as LangGraph's ToolNode reports pydantic v2 (lax mode) errors
(DESIGN.md 2.7.9, Appendix C.2). Supported JSON-schema subset: `string`, `integer`,
`boolean`, `array` of `string`, `anyOf [T, null]`, `enum`, `exclusiveMinimum`,
`minimum`, `maximum`, `exclusiveMaximum`, `required`, `default`. Unknown keys are
ignored (and dropped from the coerced arguments), except for `task`.
"""

from qumo.ai.json import JsonValue, JsonKind, JsonWriter, parse_json

from qumo_agent.util.pyfmt import py_str_repr, py_repr_json
from qumo_agent.harness._fmt import fmt, fmt1
from qumo_agent.harness.texts import TOOL_INVOCATION_ERROR, TASK_UNEXPECTED_ARGUMENT_MSG


@fieldwise_init
struct ArgCheck(Copyable):
    var ok: Bool
    var args: JsonValue             # coerced values; properties with a "default" filled in
    var errors: List[String]        # "loc: msg" in property order


comptime _MSG_STRING = "Input should be a valid string"
comptime _MSG_INT = "Input should be a valid integer"
comptime _MSG_INT_PARSE = "Input should be a valid integer, unable to parse string as an integer"
comptime _MSG_INT_FRACTION = "Input should be a valid integer, got a number with a fractional part"
comptime _MSG_INT_TOO_BIG = "Unable to parse input string as an integer, exceeded maximum size"
comptime _MSG_BOOL = "Input should be a valid boolean"
comptime _MSG_BOOL_PARSE = "Input should be a valid boolean, unable to interpret input"
comptime _MSG_LIST = "Input should be a valid list"
comptime _MSG_REQUIRED = "Field required"


def _num_text(v: Float64) -> String:
    """A schema bound as pydantic prints it (0 for 0, 1.5 for 1.5)."""
    if v == Float64(Int(v)):
        return String(Int(v))
    return String(v)


def _parse_int_text(s: String) -> Optional[Int]:
    """Python int(s) for an ASCII decimal with optional sign and surrounding whitespace."""
    var t = s.strip()
    var bs = t.as_bytes()
    if len(bs) == 0:
        return None
    var i = 0
    var neg = False
    if Int(bs[0]) == 0x2B or Int(bs[0]) == 0x2D:
        neg = Int(bs[0]) == 0x2D
        i = 1
    if i >= len(bs):
        return None
    var v = 0
    var prev_us = True
    while i < len(bs):
        var c = Int(bs[i])
        if c == 0x5F and not prev_us:   # single underscores between digits
            prev_us = True
            i += 1
            continue
        if c < 0x30 or c > 0x39:
            return None
        v = v * 10 + (c - 0x30)
        prev_us = False
        i += 1
    if prev_us:
        return None
    return -v if neg else v


@fieldwise_init
struct _Coerced(Copyable):
    var ok: Bool
    var json: String          # the coerced value as JSON text
    var error: String         # message when not ok


def _check_bounds(schema: JsonValue, v: Float64) raises -> String:
    if schema.has("exclusiveMinimum"):
        var b = schema.get("exclusiveMinimum").as_float()
        if not (v > b):
            return "Input should be greater than " + _num_text(b)
    if schema.has("minimum"):
        var b = schema.get("minimum").as_float()
        if not (v >= b):
            return "Input should be greater than or equal to " + _num_text(b)
    if schema.has("exclusiveMaximum"):
        var b = schema.get("exclusiveMaximum").as_float()
        if not (v < b):
            return "Input should be less than " + _num_text(b)
    if schema.has("maximum"):
        var b = schema.get("maximum").as_float()
        if not (v <= b):
            return "Input should be less than or equal to " + _num_text(b)
    return String()


def _enum_message(values: JsonValue) raises -> String:
    var items = List[String]()
    for i in range(len(values)):
        var e = values.at(i)
        items.append(py_str_repr(e.as_string()) if e.is_string() else py_repr_json(e))
    var out = String("Input should be ")
    for i in range(len(items)):
        if i > 0:
            out += " or " if i == len(items) - 1 else ", "
        out += items[i]
    return out^


def _coerce_scalar(schema: JsonValue, v: JsonValue, loc: String, mut errors: List[String]) raises -> _Coerced:
    """Validates `v` against one non-union schema; appends errors (with `loc`)."""
    var ty = schema.get("type").as_string() if schema.get("type").is_string() else String()
    if ty == "string":
        if not v.is_string():
            # A Literal (enum) reports its choices whatever the input type.
            errors.append(loc + ": " + (_enum_message(schema.get("enum")) if schema.has("enum") else String(_MSG_STRING)))
            return _Coerced(False, "", "")
        if schema.has("enum"):
            var ev = schema.get("enum")
            var found = False
            for i in range(len(ev)):
                if ev.at(i).is_string() and ev.at(i).as_string() == v.as_string():
                    found = True
            if not found:
                errors.append(loc + ": " + _enum_message(ev))
                return _Coerced(False, "", "")
        return _Coerced(True, v.to_json(), "")
    if ty == "integer":
        var iv: Int
        if v.is_bool():
            iv = 1 if v.as_bool() else 0
        elif v.is_number():
            var f = v.as_float()
            var integral: Bool
            try:
                iv = v.as_int()
                integral = True
            except:
                iv = 0
                integral = False
            if not integral:
                if abs(f) >= 9.2e18 or f == Float64(Int(f)):
                    errors.append(loc + ": " + _MSG_INT_TOO_BIG)
                else:
                    errors.append(loc + ": " + _MSG_INT_FRACTION)
                return _Coerced(False, "", "")
        elif v.is_string():
            var p = _parse_int_text(v.as_string())
            if not p:
                errors.append(loc + ": " + _MSG_INT_PARSE)
                return _Coerced(False, "", "")
            iv = p.value()
        else:
            errors.append(loc + ": " + _MSG_INT)
            return _Coerced(False, "", "")
        var bound = _check_bounds(schema, Float64(iv))
        if bound.byte_length() > 0:
            errors.append(loc + ": " + bound)
            return _Coerced(False, "", "")
        if schema.has("enum"):
            var ev = schema.get("enum")
            var found = False
            for i in range(len(ev)):
                if ev.at(i).is_number() and ev.at(i).as_float() == Float64(iv):
                    found = True
            if not found:
                errors.append(loc + ": " + _enum_message(ev))
                return _Coerced(False, "", "")
        return _Coerced(True, String(iv), "")
    if ty == "number":
        var fv: Float64
        if v.is_bool():
            fv = 1.0 if v.as_bool() else 0.0
        elif v.is_number():
            fv = v.as_float()
        elif v.is_string():
            try:
                fv = Float64(v.as_string().strip())
            except:
                errors.append(loc + ": Input should be a valid number, unable to parse string as a number")
                return _Coerced(False, "", "")
        else:
            errors.append(loc + ": Input should be a valid number")
            return _Coerced(False, "", "")
        var bound = _check_bounds(schema, fv)
        if bound.byte_length() > 0:
            errors.append(loc + ": " + bound)
            return _Coerced(False, "", "")
        return _Coerced(True, v.to_json() if v.is_number() else String(fv), "")
    if ty == "boolean":
        if v.is_bool():
            return _Coerced(True, v.to_json(), "")
        if v.is_number():
            var f = v.as_float()
            if f == 0.0:
                return _Coerced(True, "false", "")
            if f == 1.0:
                return _Coerced(True, "true", "")
            errors.append(loc + ": " + _MSG_BOOL_PARSE)
            return _Coerced(False, "", "")
        if v.is_string():
            var s = v.as_string().lower()
            if s == "0" or s == "off" or s == "f" or s == "false" or s == "n" or s == "no":
                return _Coerced(True, "false", "")
            if s == "1" or s == "on" or s == "t" or s == "true" or s == "y" or s == "yes":
                return _Coerced(True, "true", "")
            errors.append(loc + ": " + _MSG_BOOL_PARSE)
            return _Coerced(False, "", "")
        errors.append(loc + ": " + _MSG_BOOL)
        return _Coerced(False, "", "")
    if ty == "array":
        if not v.is_array():
            errors.append(loc + ": " + _MSG_LIST)
            return _Coerced(False, "", "")
        var items = schema.get("items")
        var out = String("[")
        var ok = True
        for i in range(len(v)):
            var c: _Coerced
            if items.is_object():
                c = _coerce(items, v.at(i), loc + "." + String(i), errors)
            else:
                c = _Coerced(True, v.at(i).to_json(), "")
            if not c.ok:
                ok = False
                continue
            if i > 0 and out.byte_length() > 1:
                out += ","
            out += c.json
        out += "]"
        return _Coerced(ok, out, "")
    if ty == "null":
        if not v.is_null():
            errors.append(loc + ": Input should be None")
            return _Coerced(False, "", "")
        return _Coerced(True, "null", "")
    # object or unknown type: accepted as is
    return _Coerced(True, v.to_json(), "")


def _coerce(schema: JsonValue, v: JsonValue, loc: String, mut errors: List[String]) raises -> _Coerced:
    var any_of = schema.get("anyOf")
    if any_of.is_array():
        var nullable = False
        var inner = JsonValue(any_of.arena, -1)
        for i in range(len(any_of)):
            var s = any_of.at(i)
            if s.get("type").is_string() and s.get("type").as_string() == "null":
                nullable = True
            elif not inner.is_defined():
                inner = s^
        if v.is_null() and nullable:
            return _Coerced(True, "null", "")
        if inner.is_defined():
            return _coerce_scalar(inner, v, loc, errors)
        return _Coerced(True, v.to_json(), "")
    return _coerce_scalar(schema, v, loc, errors)


def validate_args(parameters_json: String, args: JsonValue) raises -> ArgCheck:
    """Validates `args` (an object) against the tool's wire `parameters` schema."""
    var schema = parse_json(parameters_json if parameters_json.byte_length() > 0 else "{}")
    var props = schema.get("properties")
    var required = schema.get("required")
    var errors = List[String]()
    var w = JsonWriter()
    w.begin_object()
    for i in range(len(props)):
        var name = props.key_at(i)
        var ps = props.at(i)
        var v = args.get(name)
        if not v.is_defined():
            var is_required = False
            if required.is_array():
                for j in range(len(required)):
                    if required.at(j).is_string() and required.at(j).as_string() == name:
                        is_required = True
            if is_required:
                errors.append(name + ": " + _MSG_REQUIRED)
            elif ps.has("default"):
                w.key(name)
                w.raw(ps.get("default").to_json())
            continue
        var c = _coerce(ps, v, name, errors)
        if c.ok:
            w.key(name)
            w.raw(c.json)
    w.end_object()
    if len(errors) > 0:
        return ArgCheck(False, args.copy(), errors^)
    return ArgCheck(True, parse_json(w.finish()), errors^)


def task_extra_key_errors(args: JsonValue) -> List[String]:
    """One `<key>: Value error, Unexpected argument '<key>'; ...` line per key other than
    `description` and `subagent_type`, in sorted key order (deepagents TaskToolSchema)."""
    var keys = List[String]()
    for k in args.keys():
        if k == "description" or k == "subagent_type":
            continue
        var seen = False
        for x in keys:
            if x == k:
                seen = True
        if not seen:
            keys.append(k)
    sort(keys)
    var out = List[String]()
    var template = String(TASK_UNEXPECTED_ARGUMENT_MSG).replace("'{key}'", "{key}")
    for k in keys:
        out.append(k + ": " + fmt1(template, "key", py_str_repr(k)))
    return out^


def invocation_error_text(tool: String, original_args: JsonValue, errors: List[String]) -> String:
    """TOOL_INVOCATION_ERROR with the original arguments' Python repr."""
    var keys: List[String] = ["tool_name", "tool_kwargs", "error"]
    var values: List[String] = [tool, py_repr_json(original_args), String("\n").join(errors)]
    return fmt(TOOL_INVOCATION_ERROR, keys, values)
