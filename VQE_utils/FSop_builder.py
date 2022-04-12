from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit_nature.drivers import PySCFDriver, UnitsType


def build_FSop(instance, ):
    driver = PySCFDriver(molecule=instance.molecule.molecule, unit=UnitsType.ANGSTROM, basis=instance.molecule.basis)
    es_problem = ElectronicStructureProblem(driver)
    second_q_op = es_problem.second_q_ops()
    fOp = second_q_op[0]
    omegaOp = FermionicOp.one(instance.nqbits) * instance.omegael
    somme = fOp - omegaOp
    fsOp = (somme @ somme).reduce()
    qOp = instance.qbit_converter.convert(fsOp).primitive                       #Qubit Hamiltonian in Hartree
    return qOp


def build_FSop_test(instance, ):
    driver = PySCFDriver(molecule=instance.molecule.molecule, unit=UnitsType.ANGSTROM, basis=instance.basis)

    es_problem = ElectronicStructureProblem(driver)
    second_q_op = es_problem.second_q_ops()
    fOp = second_q_op[0]
    qOp = instance.qbit_converter.convert(fOp).primitive
    qOp2 = (qOp.power(2)).simplify()
    return qOp2