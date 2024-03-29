from numpy import random, pi
from time import time

from pyscf import gto
from regex import P
from VQE_FS import VQE_fs_test, VQE_fs
from VQE_utils import build_molecule, build_basis

mybasis = build_basis({
    'H': gto.load("./basisets/small_custom_basis.nw", 'H'),
    'Li': gto.load("./basisets/small_custom_basis.nw", 'Li'),
    'O': gto.load("./basisets/small_custom_basis.nw", 'O'),
}, {
    'H': 'H-1s-sto3g',
    'Li': 'Li-1,2s-ccpvdz',
    'O': 'O-2s,p-sto3g'
})

# H2 = build_molecule([('H', [0, 0, 0]), ('H', [0, 0, 1.5])], mybasis, 0, 1, 'H2')
# He = build_molecule([('He', [0, 0, 0])], '6-31g', 0, 1, 'He')
# H2O = build_molecule([('O', [-3.56626, 1.77639, 0]), ('H', [-2.59626, 1.77639, 0.00000]), ('H', [-3.88959, 1.36040, -0.81444])], 'sto3g', 0, 1, 'H2O')
# N2 = build_molecule([('N',[0,0,0]),('N',[0,0,1.0975])], 'sto-3g', 0, 1,'N2')
LiH = build_molecule([('Li', [0, 0, 0]), ('H', [0, 0, 0.6])], mybasis, 0, 1, 'LiH')

mymlc = LiH
algo = VQE_fs

myvqeFS = algo(
    mymlc,
    nlayers=1,
    wordiness=0,
    ω=-7,                                                                       #omega is the target zone of total energy, in Ha
    refstate='HF',
)

print(myvqeFS.nqbits)

initzeros = [0.0 for _ in range(myvqeFS.nexc * myvqeFS.nlayers)]
initrdm = [random.uniform(0, 2 * pi) for _ in range(myvqeFS.nexc * myvqeFS.nlayers)]
init_angles = initzeros

if myvqeFS.nlayers == 0:
    start = time()
    vqe_energy = myvqeFS.measure_expval(init_angles)                            #debug
    end = time()

else:
    start = time()
    vqe_energy = myvqeFS.minimize_expval(init_angles, maxiter=1000)
    end = time()
    print('init angles', init_angles)
    print('final angles', myvqeFS.opt_angles)
    print('# iterations = ', myvqeFS.niter)
    print('optimization success ', myvqeFS.success)
    print('exit flag ', myvqeFS.optmessage)

t = end - start
print('\n***********************')
print('FS RESULTS FOR ', myvqeFS.molecule.name, '; ω = ', str(myvqeFS.omegatot))
print(myvqeFS.molecule.geometry)
print('basis :', myvqeFS.molecule.basis_name)
print('Electronic Hamiltonian : (H-' + str(myvqeFS.omegael) + ')² \n')
print(myvqeFS.nlayers, 'layer(s)', '; ', myvqeFS.shots, 'shot(s)', ';', myvqeFS.backend, '; algo', myvqeFS.optimizer)
print('init state : ', myvqeFS.init_state)
print('final state = ', myvqeFS.final_state)
print('electronic energy = ', vqe_energy, 'Ha')
print('total energy : ', vqe_energy + myvqeFS.molecule.nuclear_energy, 'Ha')
print('runtime : ' + str(t // 3600) + ' h ' + str((t % 3600) // 60) + ' min ' + str((t % 3600) % 60) + ' sec')
print('***********************')
# print('final expectation value of qOp = ', myvqeFS.loss)
