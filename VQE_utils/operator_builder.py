from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp
from qiskit.quantum_info.operators.symplectic.sparse_pauli_op import SparsePauliOp
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit_nature.drivers import PySCFDriver, UnitsType
from qiskit_nature.circuit.library import UCC


def build_Ham(instance, ):
    driver = PySCFDriver(molecule=instance.molecule.molecule, unit=UnitsType.ANGSTROM, basis=instance.molecule.basis)

    es_problem = ElectronicStructureProblem(driver)
    second_q_op = es_problem.second_q_ops()
    second_q_op[0].set_truncation(0)

    OneBodyfOps = FermionicOp.zero(register_length=2)
    TwoBodyfOps = FermionicOp.zero(register_length=4)
    for op in range(len(second_q_op[0])):
        if len(second_q_op[0]._data[op][0]) == 2:
            OneBodyfOps = OneBodyfOps + FermionicOp(second_q_op[0].to_list(display_format='sparse')[op], display_format='sparse')
        elif len(second_q_op[0]._data[op][0]) == 4:
            TwoBodyfOps = TwoBodyfOps + FermionicOp(second_q_op[0].to_list(display_format='sparse')[op], display_format='sparse')

    nspinorb = es_problem.num_spin_orbitals
    (nα, nβ) = es_problem.num_particles
    fHam = second_q_op[0]                                                       #Fermionic Hamiltonian in Hartree

    OneBodyqHam = instance.qbit_converter.convert(OneBodyfOps.reduce()).primitive
    TwoBodyqHam = instance.qbit_converter.convert(TwoBodyfOps.reduce()).primitive #TODO this one has useless qbits for small systems
    qHam = instance.qbit_converter.convert(fHam).primitive                        #Qubit Hamiltonian in Hartree
    return qHam, OneBodyqHam, TwoBodyqHam, fHam, nspinorb, nα, nβ


def build_UCCops(instance):                                                     #for later TODO : turn the "sd" into a parameter...

    #tuple (nalphaspin, nbetaspin) , num_spin_orbs , generalized = True
    uccsd = UCC(instance.qbit_converter, (instance.nα, instance.nβ), instance.nspinorb, excitations='sd', alpha_spin=True, beta_spin=True, max_spin_excitation=None)
    fUCCops = uccsd.excitation_ops()                                            #Fermionic Operators
    qUCCops = []
    for op in fUCCops:
        qCCop = instance.qbit_converter.convert(op)
        qUCCops.append(qCCop.primitive)                                         #SparsePauliOp
    return qUCCops


def findpostrot(p):
    prstr = ''
    for P in p:
        if P == Pauli('I') or P == Pauli('Z'):
            prstr += 'i'
        elif P == Pauli('X'):
            prstr += 'h'
        elif P == Pauli('Y'):
            prstr += 'r'
    return prstr[::-1]
