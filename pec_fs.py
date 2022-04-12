from time import time
from qiskit import Aer
from numpy import arange

from VQE_FS import VQE_fs
from VQE_utils import build_molecule

He = build_molecule([('He', [0, 0, 0])], '6-31g', 0, 1, 'He')
H2O = build_molecule([('O', [-3.56626, 1.77639, 0]), ('H', [-2.59626, 1.77639, 0.00000]), ('H', [-3.88959, 1.36040, -0.81444])], 'sto3g', 0, 1, 'H2O')

lengths = []
energies = []

for length in arange(0.9, 3, 0.2):
    H2 = build_molecule([('H', [0, 0, 0]), ('H', [0, 0, length])], 'sto-3g', 0, 1, 'H2')

    myvqe1 = VQE_fs(
        H2,
        nlayers=1,
        backend='statevector_simulator',                                        #'qasm_simulator' or 'statevector_simulator'
        shots=1,
        optimizer='SLSQP',
        optimize_with='scipy',
        wordiness=False,
        useUent=False,
        omegatot=-0.9,                                                          #omega is the target zone of total energy, in Ha
    )

    initAngles = [0.0 for _ in range(myvqe1.nexc * myvqe1.nlayers)]

    start = time()
    vqe_energy = myvqe1.minimize_energy(initAngles, maxiter=1000)
    end = time()
    t = end - start
    print('init', initAngles, '; ', myvqe1.nlayers, 'layer(s)', '; ', myvqe1.shots, 'shot(s)', ';', myvqe1.backend, '; algo', myvqe1.optimizer)
    print('Converged after', myvqe1.niter, 'iterations ; final angles', myvqe1.opt_angles)
    print('electronic energy (computed) is ', vqe_energy, 'Ha')
    print('total energy : ', vqe_energy + myvqe1.molecule.nuclear_energy, 'Ha')
    print('time of execution : ' + str(t // 3600) + ' h ' + str((t % 3600) // 60) + ' min ' + str((t % 3600) % 60) + ' sec' + '\n')

    lengths.append(length)
    energies.append(vqe_energy + myvqe1.molecule.nuclear_energy)
    print(lengths)
    print(energies)

print(lengths)
print(energies)
