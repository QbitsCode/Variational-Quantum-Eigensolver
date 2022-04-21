from VQE_utils import build_molecule
from qc_HF import HF_g

H2 = build_molecule([('H', [0, 0, 0]), ('H', [0, 0, 1.5])], 'sto-3g', 0, 1, 'H2')
He = build_molecule([('He', [0, 0, 0])], '6-31g', 0, 1, 'He')
H2O = build_molecule([('H', [-2.59626, 1.77639, 0.00000]), ('H', [-3.88959, 1.36040, -0.81444]), ('O', [-3.56626, 1.77639, 0])], 'sto3g', 0, 1, 'H2O')

myvqeHF = HF_g(
    H2,
    backend='statevector_simulator',                                            #'qasm_simulator' or 'statevector_simulator'
    shots=1,
)

vqe_energy = myvqeHF.measure_energy()
print('RESULTS FOR ', myvqeHF.molecule.name, 'HF STATE CALCULATION')
print(myvqeHF.molecule.geometry)
print(myvqeHF.shots, 'shot(s)', ';', myvqeHF.backend)
print('electronic energy (computed) is ', vqe_energy, 'Ha')
print('total energy : ', vqe_energy + myvqeHF.molecule.nuclear_energy, 'Ha')
print('state = \n', myvqeHF.state)
