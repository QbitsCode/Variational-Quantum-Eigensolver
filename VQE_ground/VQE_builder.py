from abc import ABC
from math import pi
from xml.dom.minidom import Attr

from scipy import optimize
from qiskit import execute, Aer, QuantumRegister, QuantumCircuit

from qiskit.opflow.converters.pauli_basis_change import PauliBasisChange
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper

from VQE_utils import build_Ham, build_UCCops, findpostrot, build_HF, UCC_gate, init_gate, ent_gate, postrot_gate
from myutils import getStrFinalRes, getDictFinalRes


class VQE_g(ABC):

    def __init__(
            self,
            molecule,
            ansatz='UCC',
            refstate='HF',
            optimizer="SLSQP",
            mapper=JordanWignerMapper(),
            nlayers=1,
            backend='statevector_simulator',
            shots=1,
            optimize_with='scipy',
            wordiness=0,
            useUent=False,
            device='CPU',
    ):
        self.molecule = molecule
        self.ansatz = ansatz
        self.refstate = refstate
        self.optimizer = optimizer                                              # COBYLA fastest , SLSQP , SPSA noise tolerant
        self.mapper = mapper
        setattr(self.mapper, 'name', 'JordanWigner')                            #TODO change if other mapper used
        self.backend = backend                                                  # 'qasm_simulator' or 'statevector_simulator'
        self.shots = shots
        self.optimize_with = optimize_with                                      # 'scipy' or 'qiskit'
        self.wordiness = wordiness
        self.useUent = useUent
        self.device = device
        self.nlayers = nlayers
        self.qbit_converter = QubitConverter(self.mapper)
        self.FullqHam, self.OneBodyqHam, self.TwoBodyqHam, self.fHam, self.nspinorb, self.nα, self.nβ = build_Ham(self)
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

        if self.ansatz == 'UCC':
            self.qUCCops = build_UCCops(self)                                   # list of SparsePauliOp with  attribute .paulis = PauliList of Paulis and attribute .coeffs = nparray of complexes.
            self.nexc = len(self.qUCCops)
        else:
            raise NotImplementedError('ansatz ' + self.ansatz)

        self.molecule.hfstate = build_HF(self)

    def psi_ansatz(self, angles, Measure_State=False):
        psiQR = QuantumRegister(self.nqbits)
        psi = QuantumCircuit(psiQR)
        psi.append(init_gate(self), psiQR)
        for na in range(self.nlayers):
            if self.useUent:
                psi.append(ent_gate(self), psiQR)                               # Entangle
            psi.append(UCC_gate(self, angles, na), psiQR)                       # UCC ansatz

        if Measure_State:
            _backend = Aer.get_backend(self.backend)
            _backend.set_options(device=self.device)
            result = execute(psi, _backend).result().data(0)
            return getStrFinalRes(result), getDictFinalRes(result)

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

    def measure_energy(self, angles):
        eval_energies = []
        self.eval_dict = {}
        for POps in self.commute_groups:                                        # Index to span all excitations of commute_groups
            psi, psiQR = self.psi_ansatz(angles)
            psi.append(postrot_gate(self, POps), psiQR)                         # Post Rotations

            if self.backend == 'qasm_simulator':
                psi.measure_all()                                                       # Measure
            _backend = Aer.get_backend(self.backend)                                    #Simulation
            _backend.set_options(device=self.device)
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
        if self.nlayers == 0:
            self.init_state = init_gate(self, Measure_State=True)
            self.final_state = self.psi_ansatz(None, Measure_State=True)[0]
        if self.wordiness > 0:
            print('***energy measure***')
            print('geometry : ', self.molecule.geometry)
            print('angles : ', angles)
            print('electronic energy : ', final_energy)
            print('total energy : ', final_energy + self.molecule.nuclear_energy)
            print('************ \n\n')
        return final_energy

    def minimize_energy(self, init_angles, maxiter=1000, tol=1e-6):             # Optimization
        if self.optimize_with.casefold() == 'scipy'.casefold():                 #with scipy
            result = optimize.minimize(self.measure_energy, init_angles, method=self.optimizer, options={'maxiter': maxiter}, tol=tol, bounds=[(-2 * pi, 2 * pi) for _ in range(self.nexc * self.nlayers)])
        elif self.optimize_with.casefold() == 'qiskit'.casefold():              #with qiskit
            algorithm = self.optimizer(maxiter=maxiter)
            result = algorithm.minimize(self.measure_energy, init_angles)
        else:
            print('Unknown library for optimization')

        try:
            self.success = result.success
        except AttributeError:
            self.success = 'Undefined'
        try:
            self.optmessage = result.message
        except AttributeError:
            self.optmessage = 'Undefined'
        try:
            self.niter = result.nit
        except AttributeError:
            self.niter = 'Undefined'

        self.opt_angles = result.x
        self.opt_state = self.psi_ansatz(self.opt_angles)
        self.opt_energy = self.measure_energy(self.opt_angles)
        self.init_state, self.init_state_dict = self.psi_ansatz(init_angles, Measure_State=True)
        self.final_state, self.final_state_dict = self.psi_ansatz(self.opt_angles, Measure_State=True)
        return self.opt_energy
