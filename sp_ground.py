from time import time
from pyscf import gto
from VQE_ground import VQE_g
from VQE_utils import build_molecule, build_basis

from qiskit.algorithms.optimizers import COBYLA, SLSQP, SPSA

custom_basis = build_basis({
    'H': gto.load("./basisets/small_custom_basis.nw", 'H'),
    'Li': gto.load("./basisets/small_custom_basis.nw", 'Li'),
}, 'Li-1,2s-ccpvdz & H-1s-sto3g')

# H2 = build_molecule([('H', [0, 0, 0]), ('H', [0, 0, 1.5])], custom_basis, 0, 1, 'H2')
He = build_molecule([('He', [0, 0, 0])], '6-31g', 0, 1, 'He')
# H2O = build_molecule([('O', [-3.56626, 1.77639, 0]),('H', [-2.59626, 1.77639, 0.00000]),('H', [-3.88959, 1.36040, -0.81444])], 'sto3g', 0, 1,'H2O')
# N2 = build_molecule([('N',[0,0,0]),('N',[0,0,1.0975])], 'sto-3g', 0, 1,'N2')
# LiH = build_molecule([('Li', [0, 0, 0]), ('H', [0, 0, 1.5949])], custom_basis, 0, 1, 'LiH')

mymlc = He

myvqeG = VQE_g(
    mymlc,
    nlayers=1,
    wordiness=0,
)

initzeros = [0.0 for _ in range(myvqeG.nexc * myvqeG.nlayers)]
initAngles = initzeros

if myvqeG.nlayers == 0:
    start = time()
    vqe_energy = myvqeG.measure_energy(initAngles)
    end = time()

    print('***********************')
    print('RESULTS FOR ', myvqeG.molecule.name, 'HF STATE CALCULATION')
    print('state = \n', myvqeG.final_state)

else:
    start = time()
    vqe_energy = myvqeG.minimize_energy(initAngles, maxiter=1000)
    end = time()

    print('***********************')
    print('RESULTS FOR ', myvqeG.molecule.name, 'GROUND STATE CALCULATION')
    print('init angles', initAngles, '; ')
    print('init state : ', myvqeG.init_state)
    print('final angles', myvqeG.opt_angles)
    print('final state = ', myvqeG.final_state)

    print('niterations = ', myvqeG.niter)
    print('optimization success ', myvqeG.success)
    print('ExitFlag ', myvqeG.optmessage)

t = end - start
print(myvqeG.molecule.geometry)
print('basis :', myvqeG.molecule.basis_name)
print(myvqeG.nlayers, 'layer(s)', '; ', myvqeG.shots, 'shot(s)', ';', myvqeG.backend, '; algo', myvqeG.optimizer, '\n')
print('electronic energy is ', vqe_energy, 'Ha')
print('total energy : ', vqe_energy + myvqeG.molecule.nuclear_energy, 'Ha')
print('runtime : ' + str(t // 3600) + ' h ' + str((t % 3600) // 60) + ' min ' + str((t % 3600) % 60) + ' sec' + '\n')
print('***********************')
