from abc import ABC
from math import pi

from qiskit import execute, Aer, QuantumRegister, QuantumCircuit

from qiskit.opflow.converters.pauli_basis_change import PauliBasisChange
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper

from VQE_utils import build_Ham, findpostrot
from myutils import getStrFinalRes


class HF_g(ABC):

    def __init__(
            self,
            molecule,
            refstate='HF',
            mapper=JordanWignerMapper(),
            backend='statevector_simulator',
            shots=1,
            wordiness=0,
    ):
        self.molecule = molecule
        self.refstate = refstate
        self.mapper = mapper
        setattr(self.mapper, 'name', 'JordanWigner')                            #TODO change if other mapper used
        self.backend = backend                                                  # 'qasm_simulator' or 'statevector_simulator'
        self.shots = shots
        self.wordiness = wordiness
        self.qbit_converter = QubitConverter(self.mapper)
        self.FullqHam, self.OneBodyqHam, self.TwoBodyqHam, self.fHam, self.nspinorb = build_Ham(self)
        self.qHam = self.FullqHam
        self.Plist = self.qHam.paulis
        self.nqbits = self.qHam.num_qubits
        self.prstring = list(map(findpostrot, self.Plist))
        self.qHam_dict = {Pauli(self.qHam[j].paulis): self.qHam[j].coeffs[0] for j in range(len(self.qHam))}

        self.postRots_dict = {a: b for a, b in zip(self.Plist, self.prstring)}
        self.commute_groups = [[k for k in self.postRots_dict.keys() if self.postRots_dict[k] == n] for n in set(self.postRots_dict.values())]

        for P in self.postRots_dict.keys():
            r_P = PauliBasisChange().get_diagonal_pauli_op(PauliOp(P)).primitive
            self.postRots_dict.update({P: (self.postRots_dict[P], r_P)})

        self.hfstates = {"H2": [0, 2], "He": [0, 1], "H2O": [0, 1, 2, 3, 4, 7, 8, 9, 10, 11], "N2": [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16], "LiH": [0, 1, 6, 7]}

    def initialize(self, Measure_State=False):
        init_qr = QuantumRegister(self.nqbits)
        init_gate = QuantumCircuit(init_qr)
        if self.refstate == 'HF':
            init_gate.x(self.hfstates[self.molecule.name])
        else:
            raise NotImplementedError('refstate ' + self.refstate)              #could add a initMP2

        Uinit = init_gate.to_gate(label='Init')
        # print(init_gate.draw())
        return Uinit

    def postrotations(self, POps):
        pr_qr = QuantumRegister(self.nqbits)
        pr_gate = QuantumCircuit(pr_qr)
        prstr = self.postRots_dict[POps[0]][0]
        if prstr == 'i' * self.nqbits:
            pass
        else:
            for n in range(self.nqbits):
                if prstr[self.nqbits - 1 - n] == 'i':
                    pass
                elif prstr[self.nqbits - 1 - n] == 'h':
                    pr_gate.h(n)
                elif prstr[self.nqbits - 1 - n] == 'r':
                    pr_gate.rx(-pi / 2, n)
                else:
                    print("couldn't build post rotations")

        #print(pr_gate.draw())                                                  #DEBUG
        postRots = pr_gate.to_gate(label='postRots' + str(POps))
        return postRots

    def psi_ansatz(self, Measure_State=False):
        psiQR = QuantumRegister(self.nqbits)
        psi = QuantumCircuit(psiQR)                                             # create quantum circ
        self.na = 0                                                             #TODO change that                  # counter for angles
        psi.append(self.initialize(), psiQR)

        if Measure_State:
            _backend = Aer.get_backend(self.backend)
            result = execute(psi, _backend).result().data(0)
            return getStrFinalRes(result)

        return psi, psiQR

    def expectation_value(self, distrib, POps):
        expValues = []
        for Pstring in POps:
            r_Pstring = self.postRots_dict[Pstring][1]
            coeff = self.qHam_dict[Pstring]
            r_distrib = {key: value for key, value in distrib.items()}
            for n in range(self.nqbits):                                        #Go through the rotated Pauli String
                if r_Pstring[n] == Pauli('Z'):
                    for key in r_distrib:
                        if key[self.nqbits - 1 - n] == '1':
                            r_distrib.update({key: r_distrib[key] * (-1)})
            term = sum(r_distrib.values())
            expval = term * coeff
            expValues.append(expval.real)
            self.eval_dict.update({Pstring: term})
        eval_energy = sum(expValues)
        return eval_energy

    def measure_energy(self, ):
        eval_energies = []
        self.eval_dict = {}
        for POps in self.commute_groups:                                        # Index to span all excitations of commute_groups
            psi, psiQR = self.psi_ansatz()
            psi.append(self.postrotations(POps), psiQR)                         # Post Rotations

            if self.backend == 'qasm_simulator':
                psi.measure_all()                                                       # Measure
            _backend = Aer.get_backend(self.backend)                                    #Simulation          qasm_simulator  ;  statevector_simulator
            job = execute(psi, _backend, shots=self.shots)
            if self.backend == 'qasm_simulator':
                distrib = dict(job.result().get_counts(psi))                            # Collect the results
                n_distrib = {key: value / self.shots for key, value in distrib.items()} # Normalization
            elif self.backend == 'statevector_simulator':
                n_distrib = dict(job.result().get_counts(psi))                          #This is working only because there is no classical register

            if self.wordiness > 1:
                print(psi.draw())
                print(n_distrib)

            eval_energies.append(self.expectation_value(n_distrib, POps))
        final_energy = sum(eval_energies)
        if self.wordiness > 0:
            print(final_energy)

        self.state = self.psi_ansatz(Measure_State=True)
        return final_energy