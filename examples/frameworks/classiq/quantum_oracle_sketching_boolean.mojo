"""Quantum oracle sketching of a Boolean function: the Classiq notebook's sketch on QUMO.

Reproduces: Classiq library - Quantum Oracle Sketching for Boolean Functions
Framework: QUMO - https://github.com/ANFAIA/QUMO
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/quantum_primitives/quantum_oracle_sketching_boolean/quantum_oracle_sketching_boolean.ipynb (Classiq Library, MIT)

The notebook approximates the phase oracle O|x> = (-1)^f(x)|x> by V = prod_t exp(i theta |x_t><x_t|), one factor per classical sample (x_t, f(x_t)), with theta = pi N / M, and measures how close V comes to O. QUMO runs each factor as gates: X gates map |x_t> to |1...1>, a multi-controlled phase (append_mcp) gives e^{i theta} on that basis state, and X gates undo the flips. The samples are the notebook's own numpy draws (np.random.seed(7) for its f table; seed(0) for the Bell and N = 8 samples; seed(0) with np.random.choice for the known-p samples), stored here as digit strings, so the notebook's printed numbers can be compared directly. Checks: (A) N = 8, f = [1,0,1,0,1,1,1,1], M = 4000: one factor against its exact phase, the sketched diagonal against exp(i pi N m_x f(x)) with m_x the counts of the samples, and the spectral norm ||V - O||_2 against the closed form and the printed 1.694e-01. (B) the 2-qubit Bell verification, f = [0,0,1,1], M = 500: the exact fidelity |<Phi-|V|Phi+>|^2 against the closed form from the counts and against the printed 0.9635 (10,000 shots, four shot sigma), and QUMO's sampled fidelity against the closed form within four sigma. (C) the known-p factors theta_t = pi / (p(x_t) M) of the notebook's quantum_oracle_sketch_known_p against exp(i pi f(x) m_x / p(x)).

Deviations from the notebook: (1) the unknown-p extension (cells 19 to 24: the sin(Lambda) block encoding and the QSVT threshold polynomial, its Bell circuit has depth 6197 in the notebook output) is not reproduced, because QUMO has no QSVT. (2) The 1500-trial sample-complexity sweep in cell 15 draws a figure and prints no number; it is not reproduced. (3) The notebook defines quantum_oracle_sketch_known_p but never calls it; part C runs it on the toy distribution p = [0.10, 0.15, 0.30, 0.45] with 500 numpy samples and checks the closed form. (4) Classiq synthesize and sample are replaced by QUMO evolve (exact amplitudes) and sample; the sampled Bell check uses QUMO's shot stream, so it matches the notebook's 10,000-shot value only in distribution.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/quantum_oracle_sketching_boolean.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, pi, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.mc import append_mcp
from qumo.sim.runner import bitstring, evolve, sample
from qumo.testing.checks import CheckLog


# ---------------------------------------------------------------------------
# The notebook's samples, one digit per sample. Line breaks are skipped.
# ---------------------------------------------------------------------------
def digits_of(text: String) -> List[Int]:
    var out = List[Int]()
    var bs = text.as_bytes()
    for i in range(len(bs)):
        var ch = Int(bs[i])
        if ch >= ord("0") and ch <= ord("9"):
            out.append(ch - ord("0"))
    return out^


def bell_samples() -> List[Int]:
    """Bell check samples x_t in {0,1,2,3}: np.random.seed(0), then np.random.randint(0, 4, size=500)."""
    return digits_of("""
0310333313120320002123320111101030312330230131332301113032033232302000112001301223011311323322302310
1203020330300002303233111011130312012020132210311302232333212232332230123212102230323002023223000120
1232231000330220030120031302111300011210012113100033121313003120233231113200103233202322113303002231
1030233131320113331113322203312313311102201103003212120201110303000013303221112110121100112331333221
2320331112111203103112023020200333003132303033121003030010222002023233011100331303012202133312123212
""")


def n8_samples() -> List[Int]:
    """The same seed-0 stream continued by np.random.randint(0, 8, size=4000): the N = 8 samples."""
    return digits_of("""
3744257162506302432121451667050700607103617412400257044417661415173513573667111603041450362144400725
0633774614373342122133442110702435632104144200531551756064735343647370533613755403600361232370326036
1431251415313245177416412565723127266472236605472002357665567165327411224653647524652140102035202300
1050273465330646436244653551465537135320615556016757156422366607557323221313562524157603504217012161
7536047115466602374162341604612473045450523172165324217402242740261244114542412401570212312024640210
7234023435602363316322264360451214761745612656663256223245130545313764745021564415640026577553170000
6245052763543745442134674050772435300144570600151146766140476736756266037723767622613753054555330015
4415550600667344223364525104130326516656003754170075772116077511003656376533177207760427562025664153
4556301742174106233535412672542211514574472321630311175610563072326252517763476736135713352703404536
1563621435756313043773125405010234531773205425671431732521601011460052320747605773731452410656300120
3322167747301204526647743534757356576244111710120271460646722066525027431173307241356756363254734170
0103562701213715370245246017655733636700547227614700374740433500454557572464175041402565630011121014
6434364623643653401353315363405722515520200423513223424675512341014220642241537147561725542225130307
4350243146225207343653670077235705727400474067442042006064774377235217350765171124414504216303151371
1714542303325400213303604402355025574334671022657751223073774531010070253252315066161545270314416555
1564254447226523363630050107577140762531540324520140011014761403065700431264050217350255062771204512
4373621034647432531712473727650100573520044036061236654675621373663475215007066266763154653774303001
1754422410762106521676415202663547322616713511313073156446375456410015063064126474323177672510401365
7464275057771161136513532772071321672207733567572715635030156271551012773057601151421402403327446063
3175554572767632521326315000222343046141720235224170221736341233070564331067036432001641075473357642
3761332731151570334205310667051674563274257236601243365767422242643410766014501521604440245325776137
2032252243521726547336221333067366315154133312750472015477416526036537503256153175334377466037563367
4372016514507061420516222653672625714663151234413151441745422050711117406127405437125704032564573210
6355700334265125656116462305444735601532234357605167171651456527231206374216664000452255154253123100
3031271364476555303775271375624305354552373560316632610206663334474145753730221276170331470775337605
5174661021260523724460475562673071361013746044436562420145203565031152733515500216714572143765410236
7570233551457057471106372634650615302576131774275226437775363373064233220415010441635336132241323503
4607133047445167076214435510664230050755657324332660417050214474725110531443300625642427015634506002
5773602607275172473237237751444662764006645426435122602261573615553051114467512020766560460671771556
0642447216563410447607332507113252615562533235632370341021413172373465715746336463655105642023037377
7654662651442472407216545605164455534110307604602342137156142340347373641616431200276223124500212423
7666623165617413134326166062566375552226623112071363133431155602135504765031660634507321321571120526
2334722427244413417537747522303125524160111170167660635164055606516455414056111312620736341232662127
7003634231307270363057744340237154471777571507504520506353026415027622076055227650470324776102010747
1363767727304326022454272373154536236613626334461137443540256000435551622071311240746572103314740524
3373541644025447273572641070223026544165152333530765531515024771175014603334271420460724121016464377
5136202656607751410730326742671533244264150736533706617244034724356233670320707620264064035623666422
5774262601312557155044105546113031353147726456045137000365473520136073534351223736632565354311212512
0153224700556327600453753601471747135552370150737514625332361173671377536466212754603767771060177576
6055635535340343313333362245442657417010503460767317376063736716156707565637237025310310622075125037
""")


def known_p_samples() -> List[Int]:
    """Known-p samples: np.random.seed(0), then np.random.choice(4, size=500, p=[0.10, 0.15, 0.30, 0.45])."""
    return digits_of("""
2332232332323300033333231313222323033333223033112232311132211131123030323330212122033220332313213030
3233133313233232333332332302323212333332322333133313131333203233330231201031233013313323322113321202
2321003223313203223333231331332120122223312132311223333033321103332302332313333323302132232131210333
3320211030310031233321022233221303332232212030223330222321231322323233323311332332333331320223032201
2303322130020212101333213230321313223232233012333323113130332223112231233333333202121323322332322212
""")


# ---------------------------------------------------------------------------
# The sketch on QUMO
# ---------------------------------------------------------------------------
def append_basis_phase(mut c: Circuit, theta: Float64, x: Int, n: Int) raises:
    """The notebook's apply_basis_phase: exp(i theta |x><x|) on qubits 0..n-1, bit k of x on qubit k.

    X gates map |x> to |1...1>; a multi-controlled phase gives e^{i theta} on that one basis state
    and 1 on every other, and X gates undo the flips. Needs n >= 2."""
    for q in range(n):
        if ((x >> q) & 1) == 0:
            c.x(q)
    var controls = List[Int]()
    for q in range(n - 1):
        controls.append(q)
    append_mcp(c, theta, controls, n - 1)
    for q in range(n):
        if ((x >> q) & 1) == 0:
            c.x(q)


def append_sketch(mut c: Circuit, n: Int, xs: List[Int], thetas: List[Float64]) raises:
    """Factor t is exp(i thetas[t] |xs[t]><xs[t]|). The factors are diagonal, so they commute."""
    for t in range(len(xs)):
        append_basis_phase(c, thetas[t], xs[t], n)


def keep_active(xs: List[Int], f: List[Int]) -> List[Int]:
    """The samples with f(x_t) = 1: the others contribute the identity, as in the notebook."""
    var out = List[Int]()
    for x in xs:
        if f[x] == 1:
            out.append(x)
    return out^


def counts_of(xs: List[Int], n: Int) -> List[Int]:
    var out = List[Int](length=n, fill=0)
    for x in xs:
        out[x] += 1
    return out^


def bell_circuit(act: List[Int], thetas: List[Float64], measure: Bool) raises -> Circuit:
    """The notebook's Bell check: prepare Phi+, apply the sketch, undo the Bell preparation.

    Phi- then maps to |01>, i.e. qvar = 1 (qubit 0 is the least significant bit)."""
    var c = Circuit(2, name="bell_sketch")
    c.h(0)
    c.cx(0, 1)
    append_sketch(c, 2, act, thetas)
    c.cx(0, 1)
    c.h(0)
    if measure:
        c.measure(0, 0)
        c.measure(1, 1)
    return c^


def main() raises:
    var log = CheckLog("classiq/quantum_oracle_sketching_boolean")

    # ---- A: N = 8, the notebook's f table and its M = 4000 samples ----
    var n_a = 3
    var big_n = 8
    var f_a: List[Int] = [1, 0, 1, 0, 1, 1, 1, 1]
    var xs_a = n8_samples()
    var m_a = len(xs_a)
    if m_a != 4000:
        raise Error("QM-E900: expected the notebook's 4000 samples, got " + String(m_a) + ". Hint: check the digit block.")
    var s8 = sqrt(Float64(big_n))

    # A1: one factor on the uniform superposition: amplitude 1/sqrt(8) times exp(i 0.7) on |5>.
    var one = Circuit(n_a, name="one_factor")
    for q in range(n_a):
        one.h(q)
    append_basis_phase(one, 0.7, 5, n_a)
    var sv_one = evolve(one)
    var dev_one = 0.0
    for x in range(big_n):
        var want_re = (cos(0.7) if x == 5 else 1.0) / s8
        var want_im = (sin(0.7) if x == 5 else 0.0) / s8
        dev_one = max(dev_one, max(abs(sv_one.re[x] - want_re), abs(sv_one.im[x] - want_im)))
    _ = log.close("A1 one factor exp(i theta |5><5|) matches on all 8 amplitudes", dev_one, 0.0, 1e-12)

    # A2: the sketch V = prod_t exp(i pi N |x_t><x_t| / M), read off as V_xx = amplitude * sqrt(N).
    var theta_a = pi * Float64(big_n) / Float64(m_a)
    var act_a = keep_active(xs_a, f_a)
    var th_a = List[Float64](length=len(act_a), fill=theta_a)
    var circ_a = Circuit(n_a, name="sketch_n8")
    for q in range(n_a):
        circ_a.h(q)
    append_sketch(circ_a, n_a, act_a, th_a)
    var sv_a = evolve(circ_a)
    var cnt_a = counts_of(xs_a, big_n)
    var dev_v = 0.0
    var qumo_norm = 0.0
    var ref_norm = 0.0
    for x in range(big_n):
        var v_re = sv_a.re[x] * s8
        var v_im = sv_a.im[x] * s8
        var m_x = Float64(cnt_a[x]) / Float64(m_a)
        var phase = pi * Float64(big_n) * m_x * Float64(f_a[x])
        var r_re = cos(phase)
        var r_im = sin(phase)
        dev_v = max(dev_v, max(abs(v_re - r_re), abs(v_im - r_im)))
        # O_xx = (-1)^f(x): the spectral norm of the diagonal V - O is the largest |V_xx - O_xx|.
        var o_xx = -1.0 if f_a[x] == 1 else 1.0
        qumo_norm = max(qumo_norm, sqrt((v_re - o_xx) * (v_re - o_xx) + v_im * v_im))
        ref_norm = max(ref_norm, sqrt((r_re - o_xx) * (r_re - o_xx) + r_im * r_im))
    _ = log.close("A2 sketched diagonal equals exp(i pi N m_x f(x)) from the counts (max deviation)", dev_v, 0.0, 1e-9)
    _ = log.close("A3 spectral norm ||V - O||_2 equals the closed form from the counts", qumo_norm, ref_norm, 1e-9)
    print("A: M =", m_a, "samples, ||V - O||_2 =", qumo_norm)
    _ = log.close("A4 spectral norm matches the notebook's printed 1.694e-01", qumo_norm, 0.1694, 5e-5)

    # ---- B: the Bell verification, N = 4, f = [0, 0, 1, 1], M = 500 ----
    var f_b: List[Int] = [0, 0, 1, 1]
    var xs_b = bell_samples()
    var m_b = len(xs_b)
    var theta_b = pi * 4.0 / Float64(m_b)
    var act_b = keep_active(xs_b, f_b)
    var th_b = List[Float64](length=len(act_b), fill=theta_b)
    var cnt_b = counts_of(xs_b, 4)
    var m3 = Float64(cnt_b[3]) / Float64(m_b)
    # V_00 = 1 (f = 0) and V_33 = exp(i pi N m_3) (f = 1), so <Phi-|V|Phi+> = (V_00 - V_33) / 2.
    var phi3 = 4.0 * pi * m3
    var ov_re = (1.0 - cos(phi3)) / 2.0
    var ov_im = -sin(phi3) / 2.0
    var ref_fid = ov_re * ov_re + ov_im * ov_im

    var sv_b = evolve(bell_circuit(act_b, th_b, False))
    var exact_fid = sv_b.re[1] * sv_b.re[1] + sv_b.im[1] * sv_b.im[1]
    print("B: exact fidelity", exact_fid, " closed form", ref_fid)
    _ = log.close("B1 exact fidelity equals |<Phi-|V|Phi+>|^2 from the counts", exact_fid, ref_fid, 1e-9)
    # The notebook's 10,000-shot estimate 0.9635 has shot sigma sqrt(p(1-p)/10000): four sigma as tolerance.
    var nb_tol = 4.0 * sqrt(0.9635 * (1.0 - 0.9635) / 10000.0)
    _ = log.close("B2 exact fidelity matches the notebook's printed 0.9635 within four shot sigma", exact_fid, 0.9635, nb_tol)

    var shots_b = 10000
    var res_b = sample(bell_circuit(act_b, th_b, True), shots_b, UInt64(2026))
    var samp_fid = res_b.probability(bitstring(1, 2))
    var samp_tol = 4.0 * sqrt(ref_fid * (1.0 - ref_fid) / Float64(shots_b))
    print("B: sampled fidelity from", shots_b, "shots", samp_fid)
    _ = log.close("B3 sampled fidelity from 10000 shots matches the closed form within four sigma", samp_fid, ref_fid, samp_tol)

    # ---- C: known non-uniform p, the notebook's quantum_oracle_sketch_known_p (defined, not called) ----
    var p_c: List[Float64] = [0.10, 0.15, 0.30, 0.45]
    var f_c: List[Int] = [0, 0, 1, 1]
    var xs_c = known_p_samples()
    var m_c = len(xs_c)
    var act_c = keep_active(xs_c, f_c)
    var th_c = List[Float64]()
    for x in act_c:
        th_c.append(pi / (p_c[x] * Float64(m_c)))
    var circ_c = Circuit(2, name="known_p_sketch")
    circ_c.h(0)
    circ_c.h(1)
    append_sketch(circ_c, 2, act_c, th_c)
    var sv_c = evolve(circ_c)
    var cnt_c = counts_of(xs_c, 4)
    var dev_c = 0.0
    for x in range(4):
        var v_re = sv_c.re[x] * 2.0
        var v_im = sv_c.im[x] * 2.0
        var m_x = Float64(cnt_c[x]) / Float64(m_c)
        var phase = pi * Float64(f_c[x]) * m_x / p_c[x]
        dev_c = max(dev_c, max(abs(v_re - cos(phase)), abs(v_im - sin(phase))))
    _ = log.close("C1 known-p factors pi/(p(x_t) M) give exp(i pi f(x) m_x / p(x)) on all four entries", dev_c, 0.0, 1e-9)

    log.finish()
