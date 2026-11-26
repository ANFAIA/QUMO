"""Pasqal / Pulser wire format of analog programs, and the device specs Pasqal Cloud returns.

Vendor code: it lives outside `qumo/` (docs/ARCHITECTURE.md, "Vendor code"). The neutral program
model (`AnalogProgram`, `AnalogPulse`, `Waveform`) is `qumo.ir.analog`; `qumo.sim.rydberg` simulates it.

`pulser_json(prog, ...)` writes Pulser's abstract representation (schema version "1") with the
target device embedded as given by the cloud, plus the device layout the register must sit on
(it was the method `AnalogProgram.to_pulser_json`; `waveform_pulser_json` was `Waveform.to_json`).
`AnalogDevice` parses a device's specs JSON, and `check_against_device` validates a program
against it before anything is sent (both were in `qumo/ir/analog.mojo`).
"""

from std.math import sqrt

from qumo.core.json import json_quote, parse_json
from qumo.ir.analog import AnalogProgram, Waveform, WaveformKind, E_ANALOG, c6_for_level


def _num(v: Float64) -> String:
    var s = String(v)
    if s.find(".") < 0 and s.find("e") < 0 and s.find("n") < 0:
        s += ".0"
    return s^


# ---------------------------------------------------------------------------
# Pulser abstract representation
# ---------------------------------------------------------------------------
def waveform_pulser_json(w: Waveform) -> String:
    """One waveform in Pulser's abstract representation (constant, ramp or interpolated)."""
    if w.kind == WaveformKind.CONSTANT:
        return ("{\"kind\":\"constant\",\"duration\":" + String(w.duration) + ",\"value\":"
                + _num(w.values[0]) + "}")
    if w.kind == WaveformKind.RAMP:
        return ("{\"kind\":\"ramp\",\"duration\":" + String(w.duration) + ",\"start\":" + _num(w.values[0])
                + ",\"stop\":" + _num(w.values[1]) + "}")
    var vals = String()
    var times = String()
    var n = len(w.values)
    for i in range(n):
        vals += ("," if i > 0 else "") + _num(w.values[i])
        times += ("," if i > 0 else "") + _num(Float64(i) / Float64(n - 1) if n > 1 else 0.0)
    return ("{\"kind\":\"interpolated\",\"duration\":" + String(w.duration) + ",\"values\":[" + vals
            + "],\"times\":[" + times + "]}")


def pulser_json(prog: AnalogProgram, device_json: String, layout_coords_json: String, layout_slug: String) -> String:
    """Pulser abstract representation (schema "1"). `device_json` is the device object (physical, as the
    cloud's specs return it, or its virtual form); `layout_coords_json` is the trap coordinates array of
    the layout, or "" for no layout (virtual devices)."""
    var reg = String()
    for i in range(prog.num_atoms()):
        reg += ("," if i > 0 else "") + "{\"name\":\"q" + String(i) + "\",\"x\":" + _num(prog.xs[i]) + ",\"y\":" + _num(prog.ys[i]) + "}"
    var ops = String()
    for i in range(len(prog.pulses)):
        ref p = prog.pulses[i]
        ops += ("," if i > 0 else "") + ("{\"op\":\"pulse\",\"channel\":\"ch0\",\"protocol\":\"min-delay\",\"amplitude\":"
                + waveform_pulser_json(p.amplitude) + ",\"detuning\":" + waveform_pulser_json(p.detuning) + ",\"phase\":" + _num(p.phase)
                + ",\"post_phase_shift\":0.0}")
    return ("{\"version\":\"1\",\"name\":" + json_quote(prog.name) + ",\"device\":" + device_json
            + (",\"layout\":{\"coordinates\":" + layout_coords_json + ",\"slug\":" + json_quote(layout_slug) + "}"
               if layout_coords_json.byte_length() > 0 else String())
            + ",\"register\":[" + reg + "],\"channels\":{\"ch0\":\"rydberg_global\"},\"variables\":{}"
            + ",\"operations\":[" + ops + "],\"measurement\":\"ground-rydberg\"}")


# ---------------------------------------------------------------------------
# Device specs: layout and limits
# ---------------------------------------------------------------------------
struct AnalogDevice(Copyable):
    var name: String
    var json: String  # the specs object, verbatim
    var layout_json: String  # coordinates array of the first pre-calibrated layout
    var layout_slug: String
    var trap_x: List[Float64]
    var trap_y: List[Float64]
    var rydberg_level: Int
    var max_amp: Float64
    var max_abs_detuning: Float64
    var clock_period: Int
    var min_duration: Int
    var max_duration: Int
    var min_avg_amp: Float64
    var max_sequence_duration: Int
    var max_runs: Int
    var max_atom_num: Int
    var min_atom_distance: Float64
    var max_radial_distance: Float64
    var min_layout_filling: Float64
    var virtual_json: String  # the same device as a Pulser VirtualDevice: no layouts, is_virtual = true

    def __init__(out self, name: String, specs_json: String) raises:
        var s = parse_json(specs_json)
        self.name = name
        self.json = s.to_json()
        var layouts = s.get("pre_calibrated_layouts")
        if not layouts.is_array() or len(layouts) == 0:
            raise Error(E_ANALOG + ": device '" + name + "' has no pre-calibrated layout")
        var lay = layouts.at(0)
        self.layout_slug = lay.get_string("slug", "")
        var coords = lay.get("coordinates")
        self.layout_json = coords.to_json()
        self.trap_x = List[Float64]()
        self.trap_y = List[Float64]()
        for i in range(len(coords)):
            self.trap_x.append(coords.at(i).at(0).as_float())
            self.trap_y.append(coords.at(i).at(1).as_float())
        self.rydberg_level = s.get_int("rydberg_level", 60)
        var ch = s.get("channels")
        var found = False
        self.max_amp = 0.0
        self.max_abs_detuning = 0.0
        self.clock_period = 4
        self.min_duration = 16
        self.max_duration = 0
        self.min_avg_amp = 0.0
        for i in range(len(ch)):
            var c = ch.at(i)
            if c.get_string("id", "") == "rydberg_global":
                found = True
                self.max_amp = c.get_float("max_amp", 0.0)
                self.max_abs_detuning = c.get_float("max_abs_detuning", 0.0)
                self.clock_period = c.get_int("clock_period", 4)
                self.min_duration = c.get_int("min_duration", 16)
                self.max_duration = c.get_int("max_duration", 0)
                self.min_avg_amp = c.get_float("min_avg_amp", 0.0)
        if not found:
            raise Error(E_ANALOG + ": device '" + name + "' has no rydberg_global channel")
        self.max_sequence_duration = s.get_int("max_sequence_duration", 0)
        self.max_runs = s.get_int("max_runs", 0)
        self.max_atom_num = s.get_int("max_atom_num", 0)
        self.min_atom_distance = s.get_float("min_atom_distance", 0.0)
        self.max_radial_distance = s.get_float("max_radial_distance", 0.0)
        self.min_layout_filling = s.get_float("min_layout_filling", 0.0)
        # VirtualDevice (device-schema.json): the physical keys minus pre_calibrated_layouts, is_virtual = true
        var v = String("{")
        var first = True
        for i in range(len(s)):
            var k = s.key_at(i)
            if k == "pre_calibrated_layouts" or k == "is_virtual":
                continue
            v += ("" if first else ",") + json_quote(k) + ":" + s.at(i).to_json()
            first = False
        v += (",\"is_virtual\":true}" if not first else "\"is_virtual\":true}")
        self.virtual_json = v^

    def c6(self) raises -> Float64:
        return c6_for_level(self.rydberg_level)


def check_against_device(prog: AnalogProgram, dev: AnalogDevice, runs: Int, virtual: Bool = False) raises:
    """Device rules Pasqal applies, checked locally (QM-E955 / QM-E956 before anything is sent).

    A physical device needs every atom on a trap of its layout and at least min_layout_filling of the
    layout's traps occupied (FRESNEL: 35% of 120 traps = 42 atoms). A virtual device (emulators) has no
    layout: atoms may sit anywhere that respects the distance limits."""
    var n = prog.num_atoms()
    if n == 0:
        raise Error(E_ANALOG + ": the register has no atoms")
    if dev.max_atom_num > 0 and n > dev.max_atom_num:
        raise Error("QM-E956: " + String(n) + " atoms exceed the device maximum of " + String(dev.max_atom_num))
    if runs < 1 or (dev.max_runs > 0 and runs > dev.max_runs):
        raise Error("QM-E956: runs must be in 1.." + String(dev.max_runs) + " on " + dev.name)
    if not virtual and dev.min_layout_filling > 0:
        var need = Int(Float64(len(dev.trap_x)) * dev.min_layout_filling + 0.999999)
        if n < need:
            raise Error("QM-E956: " + dev.name + " needs at least " + String(need) + " atoms on its " + dev.layout_slug
                        + " layout (minimum filling " + _num(dev.min_layout_filling) + "); this register has " + String(n)
                        + ". Hint: run small registers on an emulator (EMU_FREE, EMU_SV, EMU_MPS), which uses the "
                        + "device's virtual form")
    for i in range(n):
        var on_trap = virtual
        for t in range(len(dev.trap_x)):
            if on_trap:
                break
            if abs(dev.trap_x[t] - prog.xs[i]) < 1e-6 and abs(dev.trap_y[t] - prog.ys[i]) < 1e-6:
                on_trap = True
                break
        if not on_trap:
            raise Error(E_ANALOG + ": atom " + String(i) + " at (" + _num(prog.xs[i]) + ", " + _num(prog.ys[i])
                        + ") is not on a trap of layout " + dev.layout_slug)
        if dev.max_radial_distance > 0 and sqrt(prog.xs[i] ** 2 + prog.ys[i] ** 2) > dev.max_radial_distance + 1e-9:
            raise Error(E_ANALOG + ": atom " + String(i) + " is beyond the maximum radial distance")
        for j in range(i + 1, n):
            var d = sqrt((prog.xs[i] - prog.xs[j]) ** 2 + (prog.ys[i] - prog.ys[j]) ** 2)
            if d < dev.min_atom_distance - 1e-9:
                raise Error(E_ANALOG + ": atoms " + String(i) + " and " + String(j) + " are " + _num(d)
                            + " um apart; the minimum is " + _num(dev.min_atom_distance))
    if len(prog.pulses) == 0:
        raise Error(E_ANALOG + ": the program has no pulse")
    for k in range(len(prog.pulses)):
        ref p = prog.pulses[k]
        var d = p.duration()
        if d < dev.min_duration or (dev.max_duration > 0 and d > dev.max_duration) or d % dev.clock_period != 0:
            raise Error(E_ANALOG + ": pulse " + String(k) + " lasts " + String(d) + " ns; it must be a multiple of "
                        + String(dev.clock_period) + " ns between " + String(dev.min_duration) + " and "
                        + String(dev.max_duration))
        if p.amplitude.max_abs() > dev.max_amp + 1e-9:
            raise Error(E_ANALOG + ": pulse " + String(k) + " amplitude exceeds " + _num(dev.max_amp) + " rad/us")
        for v in p.amplitude.values:
            if v < 0.0:
                raise Error(E_ANALOG + ": pulse " + String(k) + " has a negative amplitude")
        if p.detuning.max_abs() > dev.max_abs_detuning + 1e-9:
            raise Error(E_ANALOG + ": pulse " + String(k) + " detuning exceeds +/-" + _num(dev.max_abs_detuning) + " rad/us")
        if p.amplitude.mean() < dev.min_avg_amp - 1e-9:
            raise Error(E_ANALOG + ": pulse " + String(k) + " average amplitude " + _num(p.amplitude.mean())
                        + " is below the device minimum " + _num(dev.min_avg_amp) + " rad/us")
    if dev.max_sequence_duration > 0 and prog.duration() > dev.max_sequence_duration:
        raise Error(E_ANALOG + ": the program lasts " + String(prog.duration()) + " ns; the device allows "
                    + String(dev.max_sequence_duration))
