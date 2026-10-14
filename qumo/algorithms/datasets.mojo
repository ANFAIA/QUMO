"""Toy datasets, loaders and preprocessing for quantum machine learning.

Everything is deterministic: generators take an explicit `seed` (QUMO's
xoshiro256** `Rng`), and the PennyLane/sklearn datasets that the QML examples
cross-check against are reproduced exactly where they are deterministic
(`parity_dataset`, `double_cake`, `bars_and_stripes`). Random datasets
(`circle_dataset`, `moons_dataset`) follow the same recipe as their origin but
use QUMO's RNG, so they are not bitwise equal to numpy/sklearn draws; examples
that need bitwise agreement load vendored fixtures with `load_csv` /
`load_json_matrix` instead of downloading anything.

Shapes: a feature matrix `X` is `List[List[Float64]]` (one row per sample),
labels `y` are `List[Float64]`. Label conventions follow the origin of each
dataset and are stated per function (+/-1 for PennyLane classifiers, 0/1 for
the sklearn/tutorial ones).

Error codes: QM-E831 (shape mismatch), QM-E834 (missing or malformed file).
"""

from std.math import pi, sqrt, cos, sin, floor, ceil
from std.os.path import exists

from qumo.sim.rng import Rng
from qumo.ai.json import parse_json, JsonValue


# ---------------------------------------------------------------------------
# Deterministic PennyLane datasets
# ---------------------------------------------------------------------------
def parity_dataset(n: Int) raises -> Tuple[List[List[Float64]], List[Float64]]:
    """All 2^n bitstrings of length n with parity labels.

    Row i is `format(i, "0nb")` as in PennyLane's tutorial_variational_classifier:
    the FIRST character (most significant bit of i) is wire 0. Label = +1 for
    odd parity and -1 for even parity (`2 * (sum(bits) % 2) - 1`).
    """
    if n < 1 or n > 20:
        raise Error("QM-E831: parity_dataset() needs 1 <= n <= 20, got " + String(n) + ".")
    var X = List[List[Float64]](capacity=1 << n)
    var y = List[Float64](capacity=1 << n)
    for i in range(1 << n):
        var row = List[Float64](capacity=n)
        var par = 0
        for q in range(n):
            var b = (i >> (n - 1 - q)) & 1
            row.append(Float64(b))
            par += b
        X.append(row^)
        y.append(Float64(2 * (par % 2) - 1))
    return (X^, y^)


def double_cake(num_sectors: Int = 3) raises -> Tuple[List[List[Float64]], List[Float64]]:
    """PennyLane tutorial_kernels_module `make_double_cake_data(num_sectors)`.

    2 * num_sectors points: an outer ring (radius 0.7) then an inner ring
    (radius 0.35), sector centres at angles (k + 0.5) * 2 pi / num_sectors.
    Outer label = 2 * (k mod 2) - 1, inner label = its negative (+/-1).
    """
    if num_sectors < 1:
        raise Error("QM-E831: double_cake() needs num_sectors >= 1, got " + String(num_sectors) + ".")
    var sector_angle = 2.0 * pi / Float64(num_sectors)
    var X = List[List[Float64]](capacity=2 * num_sectors)
    var y = List[Float64](capacity=2 * num_sectors)
    for ring in range(2):
        var scale = 1.0 if ring == 0 else 0.5
        for k in range(num_sectors):
            var angle = (Float64(k) + 0.5) * sector_angle
            var x0 = 0.7 * cos(angle)
            var x1 = 0.7 * sin(angle)
            var row: List[Float64] = [scale * x0, scale * x1]
            X.append(row^)
            var sector = Int(floor(angle / sector_angle))
            var lab = Float64(2 * (sector % 2) - 1)
            y.append(lab if ring == 0 else -lab)
    return (X^, y^)


def bars_and_stripes(rows: Int, cols: Int) raises -> List[Int]:
    """Valid bars-and-stripes patterns as basis-state integers (PennyLane tutorial_qcbm order).

    Pixel (r, c) is bit position p = r * cols + c of the bitstring, and the
    integer is `int(bitstring, 2)` (pixel 0 = most significant bit), i.e. the
    PennyLane `qml.probs` index with pixel p on wire p. With QUMO's mapping
    "PennyLane wire w = QUMO qubit n-1-w" it is also the QUMO basis index.

    Order (identical to the tutorial for rows == cols): first the column
    patterns (pixel (r, c) = bit c of i, i = 0 .. 2^cols - 2, so the all-ones
    pattern is left out), then the row patterns (pixel (r, c) = bit r of i,
    i = 1 .. 2^rows - 1, so the all-zeros pattern is left out).
    2^rows + 2^cols - 2 patterns in total.
    """
    if rows < 1 or cols < 1 or rows * cols > 30:
        raise Error("QM-E831: bars_and_stripes() needs rows, cols >= 1 and rows*cols <= 30, got " + String(rows) + "x" + String(cols) + ".")
    var n = rows * cols
    var out = List[Int]()
    for i in range((1 << cols) - 1):
        var v = 0
        for r in range(rows):
            for c in range(cols):
                var p = r * cols + c
                if ((i >> c) & 1) == 1:
                    v |= 1 << (n - 1 - p)
        out.append(v)
    for i in range(1, 1 << rows):
        var v = 0
        for r in range(rows):
            for c in range(cols):
                var p = r * cols + c
                if ((i >> r) & 1) == 1:
                    v |= 1 << (n - 1 - p)
        out.append(v)
    return out^


# ---------------------------------------------------------------------------
# Random datasets (QUMO RNG; same recipe as the origin, not bitwise equal)
# ---------------------------------------------------------------------------
def circle_dataset(n: Int, seed: UInt64, radius: Float64 = sqrt(2.0 / pi)) raises -> Tuple[List[List[Float64]], List[Float64]]:
    """PennyLane tutorial_data_reuploading_classifier `circle(n)`.

    Points uniform in [-1, 1]^2; label 1 inside the circle of `radius` around
    the origin, 0 outside (class indices, as in the tutorial). The default
    radius sqrt(2/pi) makes both classes equally likely.
    """
    if n < 0:
        raise Error("QM-E831: circle_dataset() needs n >= 0, got " + String(n) + ".")
    var rng = Rng(seed)
    var X = List[List[Float64]](capacity=n)
    var y = List[Float64](capacity=n)
    for _ in range(n):
        var a = 2.0 * rng.uniform() - 1.0
        var b = 2.0 * rng.uniform() - 1.0
        var row: List[Float64] = [a, b]
        X.append(row^)
        y.append(1.0 if sqrt(a * a + b * b) < radius else 0.0)
    return (X^, y^)


def moons_dataset(n: Int, noise: Float64, seed: UInt64, shuffle: Bool = True) raises -> Tuple[List[List[Float64]], List[Float64]]:
    """sklearn `make_moons(n, noise=noise, shuffle=shuffle)` recipe with QUMO's RNG.

    n // 2 points on the outer half circle (label 0), the rest on the inner
    one (label 1), plus N(0, noise^2) jitter on both coordinates; shuffled
    with a seeded Fisher-Yates permutation when `shuffle`.
    """
    if n < 2:
        raise Error("QM-E831: moons_dataset() needs n >= 2, got " + String(n) + ".")
    var rng = Rng(seed)
    var n_out = n // 2
    var n_in = n - n_out
    var X = List[List[Float64]](capacity=n)
    var y = List[Float64](capacity=n)
    for i in range(n_out):
        var t = pi * Float64(i) / Float64(max(n_out - 1, 1))
        var row: List[Float64] = [cos(t), sin(t)]
        X.append(row^)
        y.append(0.0)
    for i in range(n_in):
        var t = pi * Float64(i) / Float64(max(n_in - 1, 1))
        var row: List[Float64] = [1.0 - cos(t), 1.0 - sin(t) - 0.5]
        X.append(row^)
        y.append(1.0)
    if noise > 0.0:
        for ref row in X:
            row[0] += noise * rng.normal()
            row[1] += noise * rng.normal()
    if shuffle:
        var perm = _permutation(n, rng)
        X = take_rows(X, perm)
        y = take_values(y, perm)
    return (X^, y^)


def _permutation(n: Int, mut rng: Rng) -> List[Int]:
    """Seeded Fisher-Yates permutation of 0..n-1."""
    var p = List[Int](capacity=n)
    for i in range(n):
        p.append(i)
    var k = n - 1
    while k > 0:
        var j = rng.below(k + 1)
        var t = p[k]
        p[k] = p[j]
        p[j] = t
        k -= 1
    return p^


# ---------------------------------------------------------------------------
# Loaders (vendored data; never downloads)
# ---------------------------------------------------------------------------
def _read_text(path: String) raises -> String:
    if not exists(path):
        raise Error("QM-E834: dataset file '" + path + "' does not exist. Hint: run from the repository root or pass an absolute path; QUMO never downloads data.")
    try:
        with open(path, "r") as f:
            return f.read()
    except e:
        raise Error("QM-E834: cannot read dataset file '" + path + "': " + String(e) + ".")


def load_csv(path: String, label_column: Int = -1, skip_header: Bool = True) raises -> Tuple[List[List[Float64]], List[Float64]]:
    """Numeric CSV -> (X, y).

    `label_column` selects the label column (negative = counted from the end,
    -1 = last column); every other column is a feature. Blank lines and lines
    starting with '#' are ignored; the first remaining line is skipped when
    `skip_header`. Every row must have the same number of numeric fields
    (QM-E834 otherwise).
    """
    var text = _read_text(path)
    var X = List[List[Float64]]()
    var y = List[Float64]()
    var width = -1
    var header_pending = skip_header
    var line_no = 0
    for raw in text.split("\n"):
        line_no += 1
        var line = String(String(raw).strip())
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        if header_pending:
            header_pending = False
            continue
        var fields = line.split(",")
        var nf = len(fields)
        if width < 0:
            width = nf
            if width < 1:
                raise Error("QM-E834: '" + path + "' line " + String(line_no) + " has no fields.")
        elif nf != width:
            raise Error("QM-E834: '" + path + "' line " + String(line_no) + " has " + String(nf) + " fields, expected " + String(width) + ". Hint: every CSV row needs the same number of columns.")
        var lc = label_column if label_column >= 0 else width + label_column
        if lc < 0 or lc >= width:
            raise Error("QM-E834: label_column " + String(label_column) + " is outside the " + String(width) + " columns of '" + path + "'.")
        var row = List[Float64](capacity=width - 1)
        for k in range(nf):
            var s = String(String(fields[k]).strip())
            var v: Float64
            try:
                v = Float64(s)
            except:
                raise Error("QM-E834: '" + path + "' line " + String(line_no) + " column " + String(k) + " is not a number: '" + s + "'. Hint: encode class names as numbers before loading.")
            if k == lc:
                y.append(v)
            else:
                row.append(v)
        X.append(row^)
    if len(X) == 0:
        raise Error("QM-E834: '" + path + "' contains no data rows.")
    return (X^, y^)


def _parse_or_e834(text: String, path: String) raises -> JsonValue:
    try:
        return parse_json(text)
    except e:
        raise Error("QM-E834: '" + path + "' is not valid JSON: " + String(e))


def load_json_matrix(path: String, key: String) raises -> List[List[Float64]]:
    """Numeric matrix stored under `key` of a JSON object (via qumo.ai.json.parse_json).

    `key` may be a '/'-separated path into nested objects ("probes/X"); an
    empty key reads the document root. A flat numeric array is returned as a
    single row. Rows may have different lengths.
    """
    var text = _read_text(path)
    var node = _parse_or_e834(text, path)
    if key.byte_length() > 0:
        for part in key.split("/"):
            var k = String(part)
            if not node.is_object() or not node.has(k):
                raise Error("QM-E834: key '" + key + "' not found in '" + path + "'.")
            node = node.get(k)
    if not node.is_array():
        raise Error("QM-E834: '" + key + "' in '" + path + "' is a " + node.kind_name() + ", expected an array of numbers or of numeric arrays.")
    var out = List[List[Float64]]()
    var flat = List[Float64]()
    for i in range(len(node)):
        var item = node.at(i)
        if item.is_number():
            flat.append(item.as_float())
        elif item.is_array():
            var row = List[Float64](capacity=len(item))
            for j in range(len(item)):
                var v = item.at(j)
                if not v.is_number():
                    raise Error("QM-E834: '" + key + "'[" + String(i) + "][" + String(j) + "] in '" + path + "' is not a number.")
                row.append(v.as_float())
            out.append(row^)
        else:
            raise Error("QM-E834: '" + key + "'[" + String(i) + "] in '" + path + "' is neither a number nor an array.")
    if len(flat) > 0:
        if len(out) > 0:
            raise Error("QM-E834: '" + key + "' in '" + path + "' mixes numbers and arrays.")
        out.append(flat^)
    return out^


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _check_rect(X: List[List[Float64]], who: String) raises -> Int:
    if len(X) == 0:
        raise Error("QM-E831: " + who + "() got an empty matrix.")
    var w = len(X[0])
    for i in range(len(X)):
        if len(X[i]) != w:
            raise Error("QM-E831: " + who + "(): row " + String(i) + " has " + String(len(X[i])) + " entries, row 0 has " + String(w) + ".")
    return w


def standard_scale(X: List[List[Float64]]) raises -> Tuple[List[List[Float64]], List[Float64], List[Float64]]:
    """(Xs, mean, std) with Xs = (X - mean) / std per column (sklearn StandardScaler).

    `std` is the population standard deviation; a constant column gets std 1
    (as sklearn does), so it maps to 0.
    """
    var w = _check_rect(X, "standard_scale")
    var n = Float64(len(X))
    var mean = List[Float64](length=w, fill=0.0)
    var std = List[Float64](length=w, fill=0.0)
    for row in X:
        for k in range(w):
            mean[k] += row[k]
    for k in range(w):
        mean[k] /= n
    for row in X:
        for k in range(w):
            var d = row[k] - mean[k]
            std[k] += d * d
    for k in range(w):
        var s = sqrt(std[k] / n)
        std[k] = s if s > 0.0 else 1.0
    var Xs = apply_standard_scale(X, mean, std)
    return (Xs^, mean^, std^)


def apply_standard_scale(X: List[List[Float64]], mean: List[Float64], std: List[Float64]) raises -> List[List[Float64]]:
    """(X - mean) / std per column, with the statistics of a previous standard_scale()."""
    var out = List[List[Float64]](capacity=len(X))
    for i in range(len(X)):
        if len(X[i]) != len(mean) or len(mean) != len(std):
            raise Error("QM-E831: apply_standard_scale(): row " + String(i) + " has " + String(len(X[i])) + " entries, the scaler has " + String(len(mean)) + ".")
        var row = List[Float64](capacity=len(mean))
        for k in range(len(mean)):
            row.append((X[i][k] - mean[k]) / std[k])
        out.append(row^)
    return out^


def l2_normalize_rows(X: List[List[Float64]]) -> List[List[Float64]]:
    """Each row divided by its Euclidean norm (all-zero rows are left unchanged)."""
    var out = List[List[Float64]](capacity=len(X))
    for row in X:
        var s = 0.0
        for v in row:
            s += v * v
        var r = row.copy()
        if s > 0.0:
            var inv = 1.0 / sqrt(s)
            for ref v in r:
                v *= inv
        out.append(r^)
    return out^


def pad_rows(X: List[List[Float64]], width: Int, value: Float64 = 0.0) raises -> List[List[Float64]]:
    """Each row right-padded with `value` to `width` entries (QM-E831 if a row is longer)."""
    var out = List[List[Float64]](capacity=len(X))
    for i in range(len(X)):
        if len(X[i]) > width:
            raise Error("QM-E831: pad_rows(): row " + String(i) + " has " + String(len(X[i])) + " entries, more than width " + String(width) + ".")
        var r = X[i].copy()
        while len(r) < width:
            r.append(value)
        out.append(r^)
    return out^


def train_test_split(n: Int, test_fraction: Float64, seed: UInt64) raises -> Tuple[List[Int], List[Int]]:
    """(train_idx, test_idx) from a seeded permutation of 0..n-1 (sklearn semantics:
    n_test = ceil(test_fraction * n), test = first n_test of the permutation)."""
    if n < 2 or test_fraction <= 0.0 or test_fraction >= 1.0:
        raise Error("QM-E831: train_test_split() needs n >= 2 and 0 < test_fraction < 1, got n=" + String(n) + ", test_fraction=" + String(test_fraction) + ".")
    var n_test = Int(ceil(test_fraction * Float64(n)))
    n_test = min(max(n_test, 1), n - 1)
    var rng = Rng(seed)
    var p = _permutation(n, rng)
    var test = List[Int](capacity=n_test)
    var train = List[Int](capacity=n - n_test)
    for i in range(n):
        if i < n_test:
            test.append(p[i])
        else:
            train.append(p[i])
    return (train^, test^)


def take_rows(X: List[List[Float64]], idx: List[Int]) raises -> List[List[Float64]]:
    """Rows X[idx[0]], X[idx[1]], ... (copies)."""
    var out = List[List[Float64]](capacity=len(idx))
    for i in idx:
        if i < 0 or i >= len(X):
            raise Error("QM-E831: take_rows(): index " + String(i) + " outside 0.." + String(len(X) - 1) + ".")
        out.append(X[i].copy())
    return out^


def take_values(y: List[Float64], idx: List[Int]) raises -> List[Float64]:
    """Entries y[idx[0]], y[idx[1]], ... (labels companion of take_rows)."""
    var out = List[Float64](capacity=len(idx))
    for i in idx:
        if i < 0 or i >= len(y):
            raise Error("QM-E831: take_values(): index " + String(i) + " outside 0.." + String(len(y) - 1) + ".")
        out.append(y[i])
    return out^
