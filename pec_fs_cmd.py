from time import time
from numpy import arange
from pyscf import gto

from VQE_FS import VQE_fs
from VQE_utils import build_molecule, build_basis
from myutils import build_h5file_onestate, build_ket_coeff_dict

from arg_parser import build_default_arg_parser

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
        'O': 'O-2s,p-sto3g'
    })
elif args.basis == 'sto-3g':
    mybasis = 'sto-3g'
elif args.basis == '6-31g':
    mybasis = '6-31g'
else:
    mybasis = args.basis

id_job = args.jobID

lengths = []
energies = []

namefile = args.outdir + args.molecule + "_pec_FS_" + str(id_job)

outputfile = open(namefile + ".txt", "a+")
outputfile.write('**********************************************' + '\n\n')
outputfile.write('FOLDED SPECTRUM VQE POTENTIAL ENERGY CURVE' + '\n\n')
outputfile.write('**********************************************' + '\n')

outputfile.write("Input : \n")
outputfile.write(str(args.nlayer) + "layer(s) \n")
outputfile.write('Device : ' + args.device + '\n')
outputfile.write('Range from ' + str(args.lb) + ' to ' + str(args.ub) + ' Å, step ' + str(args.step) + ' Å \n\n\n\n')

outputfile.flush()

for length in arange(args.lb, args.ub, args.step):

    if args.molecule == 'H2':
        mymlc = build_molecule([('H', [0, 0, 0]), ('H', [0, 0, length])], mybasis, 0, 1, 'H2')
    elif args.molecule == 'LiH':
        mymlc = build_molecule([('H', [0, 0, length]), ('Li', [0, 0, 0])], mybasis, 0, 1, 'LiH')
    elif args.molecule == 'H2O':
        mymlc = build_molecule([('H', [-2.59626, 1.77639, 0.00000]), ('H', [-3.88959, 1.36040, -0.81444]), ('O', [-3.56626, 1.77639, 0])], mybasis, 0, 1, 'H2O') #TODO make the hoh angle vary.
    else:
        raise NotImplementedError('Molecule ' + args.molecule)

    if length == args.lb:
        ω = args.omega
    else:
        init_angles = last_angles
        ω = last_energy

    myvqeFS = VQE_fs(
        mymlc,
        nlayers=args.nlayer,
        wordiness=0,
        ω=ω,                                                                    #omega is the target zone of total energy, in Ha
        refstate=args.refstate,
    )

    if length == args.lb:
        init_angles = [0.0 for _ in range(myvqeFS.nexc * myvqeFS.nlayers)]
        ket_coeff_dict, ket_list = build_ket_coeff_dict(myvqeFS.nα + myvqeFS.nβ, myvqeFS.nqbits)

    init_angles = [0.0 for _ in range(myvqeFS.nexc * myvqeFS.nlayers)]

    start = time()
    vqe_energy = myvqeFS.minimize_expval(init_angles, maxiter=1000)
    end = time()
    t = end - start

    lengths.append(length)
    energies.append(vqe_energy + myvqeFS.molecule.nuclear_energy)

    final_state_dict = myvqeFS.final_state_dict
    last_angles = myvqeFS.opt_angles
    last_energy = vqe_energy + myvqeFS.molecule.nuclear_energy

    for ket in ket_coeff_dict:
        if ket in final_state_dict:
            ket_coeff_dict[ket].append(final_state_dict[ket].real)
        else:
            ket_coeff_dict[ket].append(0)

    outputfile.write('Bond Length : ' + str(length) + '\n')
    outputfile.write('Omega (Total Energy Shift) = ' + str(myvqeFS.omegatot) + ' Ha \n')
    outputfile.write('refstate = ' + args.refstate + '\n')
    outputfile.write('final state = ' + myvqeFS.final_state + '\n')
    outputfile.write('Electronic Hamiltonian : (H-' + str(myvqeFS.omegael) + ')² \n')
    outputfile.write('electronic energy = ' + str(vqe_energy) + ' Ha' + '\n')
    outputfile.write('total energy = ' + str(vqe_energy + myvqeFS.molecule.nuclear_energy) + ' Ha' + '\n')
    outputfile.write('runtime : ' + str(t // 3600) + ' h ' + str((t % 3600) // 60) + ' min ' + str((t % 3600) % 60) + ' sec' + '\n')
    outputfile.write("---------------------------------" + '\n')
    outputfile.write('init angles : ' + str(init_angles) + '\n' + str(myvqeFS.nlayers) + ' layer(s)' + '\n' + 'Backend : ' + str(myvqeFS.backend) + '\n' + str(myvqeFS.shots) + ' shot(s) ' + '\nalgo ' +
                     str(myvqeFS.optimizer) + '\n')
    outputfile.write('Molecule : ' + str(myvqeFS.molecule.name) + '\n')
    outputfile.write(str(myvqeFS.molecule.geometry) + '\n')
    outputfile.write('Nuclear Repulsion = ' + str(myvqeFS.molecule.nuclear_energy) + ' Ha\n')
    outputfile.write('# iterations = ' + str(myvqeFS.niter) + '\n')
    outputfile.write('optimization success ' + str(myvqeFS.success) + '\n')
    outputfile.write('ExitFlag ' + str(myvqeFS.optmessage) + '\n')
    outputfile.write("-------------------------------" + '\n\n\n\n')
    outputfile.write('\n\n LENGTHS \n\n')
    outputfile.write(str(lengths) + '\n')
    outputfile.write('\n\n ENERGIES \n\n')
    outputfile.write(str(energies) + '\n')
    outputfile.write("-------------------------------------------------------" + '\n\n\n')
    outputfile.flush()

outputfile.write('Basis : ' + str(myvqeFS.molecule.basis_name) + '\n')
outputfile.write('Use Uent gate : ' + str(myvqeFS.useUent) + '\n')
outputfile.write('Mapper : ' + str(myvqeFS.mapper.name) + '\n')
outputfile.write("-------------------------------------------------------" + '\n')
outputfile.write('\n\n LENGTHS \n\n')
outputfile.write(str(lengths) + '\n')
outputfile.write('\n\n ENERGIES \n\n')
outputfile.write(str(energies) + '\n')
outputfile.write("-------------------------------------------------------" + '\n\n\n')

outputfile.write("******************************" + '\n')
build_h5file_onestate(namefile, lengths, energies, ket_coeff_dict, ket_list)
outputfile.write('Datafile ' + namefile + '.hdf5' + ' created' + '\n')
outputfile.write("******************************" + '\n')

outputfile.write('*****pec terminated normally*****')
outputfile.close()