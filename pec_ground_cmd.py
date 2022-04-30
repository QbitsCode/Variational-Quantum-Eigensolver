from time import time
from numpy import arange
from pyscf import gto

from VQE_ground import VQE_g
from VQE_utils import build_molecule, build_basis
from myutils import build_h5file_onestate, build_ket_coeff_dict

from arg_parser import build_default_arg_parser

args = build_default_arg_parser().parse_args()

if args.basis == 'small_custom':
    mybasis = build_basis(
        {
            'H': gto.load("./basisets/small_custom_basis.nw", 'H'),
            'Li': gto.load("./basisets/small_custom_basis.nw", 'Li'),
            'O': gto.load("./basisets/small_custom_basis.nw", 'O'),
            'Be': gto.load("./basisets/small_custom_basis.nw", 'Be'),
        }, {
            'H': 'H-1s-sto3g',
            'Li': 'Li-1,2s-ccpvdz',
            'O': 'O-2s,p-sto3g',
            'Be': 'Be-1,2s-sto3g'
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

if args.nlayer == 0:
    namefile = args.outdir + args.molecule + "_pec_HF_" + str(id_job)
    outputfile = open(namefile + ".txt", "a+")
    outputfile.write('**********************************************' + '\n\n')
    outputfile.write('RESULTS FOR ' + args.molecule + ' Hartree Fock ENERGY VQE CALCULATION' + '\n\n')
    outputfile.write('**********************************************' + '\n\n\n\n')
    outputfile.flush()
else:
    namefile = args.outdir + args.molecule + "_pec_ground_" + str(id_job)
    outputfile = open(namefile + ".txt", "a+")
    outputfile.write('**********************************************' + '\n\n')
    outputfile.write('RESULTS FOR ' + args.molecule + ' VQE ENERGY CALCULATION' + '\n\n')
    outputfile.write('**********************************************' + '\n')
    outputfile.flush()

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
    elif args.molecule == 'BeH2':
        mymlc = build_molecule([('H', [0, 0, -length]), ('H', [0, 0, length]), ('Be', [0, 0, 0])], mybasis, 0, 1, 'BeH2')
    else:
        raise NotImplementedError('Molecule ' + args.molecule)

    myvqe = VQE_g(
        mymlc,
        nlayers=args.nlayer,
        device=args.device,
        wordiness=args.wordiness,
    )

    if length == args.lb:
        init_angles = [0.0 for _ in range(myvqe.nexc * myvqe.nlayers)]
        ket_coeff_dict, ket_list = build_ket_coeff_dict(myvqe.nα + myvqe.nβ, myvqe.nqbits)
    else:
        init_angles = last_angles

    if myvqe.nlayers == 0:
        start = time()
        vqe_energy = myvqe.measure_energy(init_angles)
        end = time()
    else:
        start = time()
        vqe_energy = myvqe.minimize_energy(init_angles, maxiter=1000)
        end = time()
        outputfile.write('init angles' + str(init_angles) + '\n')
        outputfile.write('final angles = ' + str(myvqe.opt_angles) + '\n')

    t = end - start
    lengths.append(length)
    energies.append(vqe_energy + myvqe.molecule.nuclear_energy)

    final_state_dict = myvqe.final_state_dict
    last_angles = myvqe.opt_angles
    last_energy = vqe_energy + myvqe.molecule.nuclear_energy

    for ket in ket_coeff_dict:
        if ket in final_state_dict:
            ket_coeff_dict[ket].append(final_state_dict[ket].real)
        else:
            ket_coeff_dict[ket].append(0)

    outputfile.write('Bond Length : ' + str(length) + '\n')
    outputfile.write('init state = ' + myvqe.init_state + '\n')
    outputfile.write('final state = ' + myvqe.final_state + '\n')
    outputfile.write('electronic energy = ' + str(vqe_energy) + ' Ha' + '\n')
    outputfile.write('total energy = ' + str(vqe_energy + myvqe.molecule.nuclear_energy) + ' Ha' + '\n')
    outputfile.write('runtime : ' + str(t // 3600) + ' h ' + str((t % 3600) // 60) + ' min ' + str((t % 3600) % 60) + ' sec' + '\n')
    outputfile.write("------------------" + '\n')
    outputfile.write('init angles : ' + str(init_angles) + '\n' + str(myvqe.nlayers) + ' layer(s)' + '\n' + 'Backend : ' + str(myvqe.backend) + '\n' + str(myvqe.shots) + ' shot(s) ' + '\nalgo ' + str(myvqe.optimizer) +
                     '\n')
    outputfile.write('Molecule : ' + str(myvqe.molecule.name) + '\n')
    outputfile.write(str(myvqe.molecule.geometry) + '\n')
    outputfile.write('Nuclear Repulsion = ' + str(myvqe.molecule.nuclear_energy) + ' Ha\n')
    outputfile.write('# iterations = ' + str(myvqe.niter) + '\n')
    outputfile.write('optimization success ' + str(myvqe.success) + '\n')
    outputfile.write('ExitFlag ' + str(myvqe.optmessage) + '\n')
    outputfile.write("-------------------------------" + '\n\n\n\n')
    outputfile.flush()

outputfile.write('Basis : ' + str(myvqe.molecule.basis_name) + '\n')
outputfile.write('Use Uent gate : ' + str(myvqe.useUent) + '\n')
outputfile.write('Mapper : ' + str(myvqe.mapper.name) + '\n')
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