"""Router network (sign-embedding MLP) with forward and backward.

STUB (W-1): the signatures of MOJO_PORT.md section 5. Component C8 replaces this file; every
behavioural body raises with a "STUB" message.
"""

from qumo.compiler.policy import ModelFile


struct RouterNet(Movable):
    var hidden: Int
    var params: List[Float32]

    def __init__(out self, hidden: Int, seed: UInt64):
        self.hidden = hidden
        self.params = List[Float32]()

    @staticmethod
    def from_model(m: ModelFile) raises -> RouterNet:
        raise Error("QM-E640: qumo.rl.router_net.RouterNet.from_model is a STUB (component C8)")
