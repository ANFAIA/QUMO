"""Router teacher records (328 bytes each): reader that concatenates files and applies the BC filter.

STUB (W-1): the signatures of MOJO_PORT.md section 5. Component C8 replaces this file; every
behavioural body raises with a "STUB" message.
"""


struct TeacherData(Movable):
    var n: Int
    var obs: List[Int8]
    var A: List[Int16]
    var act: List[Int16]
    var job: List[Int16]
    var tb: List[Int16]
    var q: List[Float32]

    def __init__(out self):
        self.n = 0
        self.obs = List[Int8]()
        self.A = List[Int16]()
        self.act = List[Int16]()
        self.job = List[Int16]()
        self.tb = List[Int16]()
        self.q = List[Float32]()


def read_teacher_files(paths: List[String]) raises -> TeacherData:
    """QM-E646 on malformed data."""
    raise Error("QM-E640: qumo.rl.teacher_data.read_teacher_files is a STUB (component C8)")
