from time import time

from arg_parser import build_default_arg_parser
from pyscf import gto

from VQE_ground import VQE_g
from VQE_FS import VQE_fs, VQE_fs_test
from VQE_utils import build_molecule, build_basis

args = build_default_arg_parser().parse_args()

if args.basis == 'custom':
    mybasis = build_basis({
        'H': gto.load("./basisets/custom_basis.nw", 'H'),
        'Li': gto.load("./basisets/custom_basis.nw", 'Li'),
    }, {
        'H': 'H-1s-sto3g',
        'Li': 'Li-1,2,3s-ccpvdz',
    })
elif args.basis == 'small_custom':
    mybasis = build_basis({
        'H': gto.load("./basisets/small_custom_basis.nw", 'H'),
        'Li': gto.load("./basisets/small_custom_basis.nw", 'Li'),
        'O': gto.load("./basisets/small_custom_basis.nw", 'O'),
    }, {
        'H': 'H-1s-sto3g',
        'Li': 'Li-1,2s-ccpvdz',
        'O': 'O-2s,p-sto3g',
    })
elif args.basis == 'sto-3g':
    mybasis = 'sto-3g'
elif args.basis == '6-31g':
    mybasis = '6-31g'

if args.molecule == 'H2':
    if args.bondlen == 0:
        args.bondlen = 0.735
    mymlc = build_molecule([('H', [0, 0, 0]), ('H', [0, 0, args.bondlen])], mybasis, 0, 1, 'H2')
elif args.molecule == 'He':
    mymlc = build_molecule([('He', [0, 0, 0])], mybasis, 0, 1, 'He')
elif args.molecule == 'H2O':
    mymlc = build_molecule([('H', [-2.59626, 1.77639, 0.00000]), ('H', [-3.88959, 1.36040, -0.81444]), ('O', [-3.56626, 1.77639, 0])], mybasis, 0, 1, 'H2O')
elif args.molecule == 'LiH':
    if args.bondlen == 0:
        args.bondlen = 1.5949
    mymlc = build_molecule([('H', [0, 0, args.bondlen]), ('Li', [0, 0, 0])], mybasis, 0, 1, 'LiH')

id_job = args.jobID

if args.algo == 'VQE_g':
    myvqe = VQE_g(
        mymlc,
        nlayers=args.nlayer,
        wordiness=0,
        device=args.device,
    )
elif args.algo == 'VQE_fs':
    myvqe = VQE_fs(
        mymlc,
        nlayers=args.nlayer,
        wordiness=0,
        ω=args.omega,
        refstate=args.refstate,
        device=args.device,
    )
elif args.algo == 'VQE_fs_test':
    myvqe = VQE_fs_test(
        mymlc,
        nlayers=args.nlayer,
        wordiness=0,
        ω=args.omega,
        refstate=args.refstate,
        device=args.device,
    )

initzeros = [0.0 for _ in range(myvqe.nexc * myvqe.nlayers)]
init_angles = initzeros

if myvqe.nlayers == 0:
    outputfile = open("./outputs/" + myvqe.molecule.name + '_' + args.algo + "_HF_" + str(id_job) + ".txt", "a+")
    outputfile.write('**********************************************' + '\n\n')
    outputfile.write('RESULTS FOR ' + myvqe.molecule.name + ' VQE ENERGY CALCULATION' + '\n\n')
    outputfile.write('**********************************************' + '\n\n\n\n')
    outputfile.write(
        str(myvqe.molecule.geometry) + '\n' + args.algo + '\n' + str(args.nlayer) + 'layer(s)' + '\n' + 'ω= ' + str(args.omega) + '\n' + args.basis + ' basis' + '\n' + 'refstate = ' + args.refstate + '\n\n\n\n')
    outputfile.flush()
    if args.algo == 'VQE_g':
        start = time()
        vqe_energy = myvqe.measure_energy(init_angles)
        end = time()
    else:
        start = time()
        vqe_energy = myvqe.measure_expval(init_angles)
        end = time()

else:

    outputfile = open("./outputs/" + myvqe.molecule.name + '_' + args.algo + "_SP_" + str(id_job) + ".txt", "a+")
    outputfile.write('**********************************************' + '\n\n')
    outputfile.write('RESULTS FOR ' + myvqe.molecule.name + ' VQE ENERGY CALCULATION' + '\n\n')
    outputfile.write('**********************************************' + '\n\n\n\n')
    outputfile.write(
        str(myvqe.molecule.geometry) + '\n' + args.algo + '\n' + str(args.nlayer) + 'layer(s)' + '\n' + 'ω= ' + str(args.omega) + '\n' + args.basis + ' basis' + '\n' + 'refstate = ' + args.refstate + '\n\n\n\n')
    outputfile.flush()
    if args.algo == 'VQE_g':
        start = time()
        vqe_energy = myvqe.minimize_energy(init_angles)
        end = time()
    else:
        start = time()
        vqe_energy = myvqe.minimize_expval(init_angles)
        end = time()

    outputfile.write('init angles' + str(init_angles) + '\n')

    outputfile.write('final angles = ' + str(myvqe.opt_angles) + '\n')

t = end - start
outputfile.write('init state : ' + myvqe.init_state + '\n')
outputfile.write('final state = ' + myvqe.final_state + '\n')
outputfile.write('VQE electronic energy = ' + str(vqe_energy) + ' Ha' + '\n')
outputfile.write('total energy = ' + str(vqe_energy + myvqe.molecule.nuclear_energy) + ' Ha' + '\n')
outputfile.write('runtime : ' + str(t // 3600) + ' h ' + str((t % 3600) // 60) + ' min ' + str((t % 3600) % 60) + ' sec' + '\n')
outputfile.write("-------------------------------------------------------" + '\n')
outputfile.write('init angles : ' + str(init_angles) + '\n' + str(myvqe.nlayers) + ' layer(s)' + '\n' + 'Backend : ' + str(myvqe.backend) + '\n' + str(myvqe.shots) + ' shot(s) ' + '\nalgo ' + str(myvqe.optimizer) + '\n')
outputfile.write('Nuclear Repulsion = ' + str(myvqe.molecule.nuclear_energy) + ' Ha\n')
outputfile.write('Basis : ' + str(myvqe.molecule.basis_name) + '\n')
outputfile.write('Use Uent gate : ' + str(myvqe.useUent) + '\n')
outputfile.write('Mapper : ' + str(myvqe.mapper.name) + '\n')
outputfile.write('qHam : ' + str(myvqe.qHam_dict) + '\n')

outputfile.write('******VQE terminated normally******')

outputfile.close()