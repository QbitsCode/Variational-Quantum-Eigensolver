from numpy import linalg, sort, power, subtract, arange
from time import time
from VQE_utils import build_molecule, build_basis
from pyscf import gto
from VQE_ground import VQE_g
from myutils import build_h5file_multiplestates, getDictFinalRes, getStrFinalRes, build_state_kets_dict

from arg_parser import build_default_arg_parser

args = build_default_arg_parser().parse_args()

if args.basis == 'custom':
    mybasis = build_basis({
        'H': gto.load("./basisets/custom_basis.nw", 'H'),
        'Li': gto.load("./basisets/custom_basis.nw", 'Li'),
    }, 'Li-1,2,3s-ccpvdz & H-1s-sto3g')
elif args.basis == 'small_custom':
    mybasis = build_basis({
        'H': gto.load("./basisets/small_custom_basis.nw", 'H'),
        'Li': gto.load("./basisets/small_custom_basis.nw", 'Li'),
    }, 'Li-1,2s-ccpvdz & H-1s-sto3g')
elif args.basis == 'sto-3g':
    mybasis = 'sto-3g'
elif args.basis == '6-31g':
    mybasis = '6-31g'
else:
    mybasis = args.basis

namefile = args.outdir + args.molecule + "_pec_eigen"

outputfile = open(namefile + ".txt", "a+")
outputfile.write('**********************************************' + '\n\n')
outputfile.write('EXACT POTENTIAL ENERGY CURVE' + '\n\n')
outputfile.write('**********************************************' + '\n')
outputfile.write("Input : \n")
outputfile.write('Range from ' + str(args.lb) + ' to ' + str(args.ub) + ' Å, step ' + str(args.step) + ' Å \n\n\n\n')
outputfile.flush()

lengths = []

for length in arange(args.lb, args.ub, args.step):

    if args.molecule == 'H2':
        mymlc = build_molecule([('H', [0, 0, 0]), ('H', [0, 0, length])], mybasis, 0, 1, 'H2')
    elif args.molecule == 'LiH':
        mymlc = build_molecule([('Li', [0, 0, 0]), ('H', [0, 0, length])], mybasis, 0, 1, 'LiH')
    else:
        raise NotImplementedError('Molecule ' + args.molecule)

    outputfile.write(mymlc.name + ' : ' + str(mymlc.geometry) + '\n\n')

    myvqeG = VQE_g(
        mymlc,
        nlayers=0,
    )

    if length == args.lb:
        outputfile.write('States with ' + str(myvqeG.nα + myvqeG.nβ) + ' electrons only. \n\n\n\n')
        outputfile.flush()
        state_kets_dict, wrong_index, ket_list = build_state_kets_dict(myvqeG.nα + myvqeG.nβ, myvqeG.nqbits)
        energies = [[] for _ in range(2**myvqeG.nqbits - len(wrong_index))]

    mathamG = myvqeG.qHam.to_operator()
    start = time()
    eigen = linalg.eig(mathamG)
    end = time()
    t = end - start

    eigenstates = eigen[1]
    eigenval = eigen[0]
    phy_eigenval = []
    statestr = []
    totenergy_kets_dict = {}
    counterval = 0

    for k in range(len(eigenval)):
        estate = eigenstates[:, k]
        takeit = True
        for i in wrong_index:
            if takeit and estate[i] < 1e-6:
                pass
            else:
                takeit = False
                break
        if takeit:
            ket_coeff_dict = getDictFinalRes(estate)
            sorted_ket_coeff_dict = {key: v for key, v in sorted(ket_coeff_dict.items(), key=lambda item: abs(item[1]), reverse=True)}
            statestr.append(getStrFinalRes(sorted_ket_coeff_dict).replace('\n', ' '))
            phy_eigenval.append(eigenval[k])
            for ket in state_kets_dict['e' + str(counterval)]:
                if ket in sorted_ket_coeff_dict:
                    state_kets_dict['e' + str(counterval)][ket].append(sorted_ket_coeff_dict[ket].real)
                else:
                    state_kets_dict['e' + str(counterval)][ket].append(0)
            counterval += 1

    totvalues = phy_eigenval + mymlc.nuclear_energy

    for k in range(len(totvalues)):
        if totvalues[k] in totenergy_kets_dict.keys():
            totenergy_kets_dict[totvalues[k]].append(statestr[k])
        else:
            totenergy_kets_dict.update({totvalues[k]: [statestr[k]]})

    phy_eigenval = sort(phy_eigenval)
    totvalues = sort(totvalues)
    totenergy_kets_dict = dict(sorted(totenergy_kets_dict.items(), reverse=True))

    outputfile.write('Bond Length : ' + str(length) + '\n\n')
    outputfile.write('vp of H in Eel= ' + str(phy_eigenval) + '\n\n')
    outputfile.write('vp of H in Etot= ' + str(totvalues) + '\n\n')
    outputfile.write('Eigenstates : \n')
    for key in totenergy_kets_dict:
        for state in totenergy_kets_dict[key]:
            outputfile.write(str(key.real) + '\t\t' + str(state).replace('\n', '') + '\n\n')
    outputfile.write('runtime : ' + str(t // 3600) + ' h ' + str((t % 3600) // 60) + ' min ' + str((t % 3600) % 60) + ' sec' + '\n')
    outputfile.write('------------------------------------------------------- \n\n')
    outputfile.flush()

    lengths.append(length)
    for k in range(len(phy_eigenval)):
        energies[k].append(totvalues[k])

outputfile.write("******************************" + '\n')
outputfile.write('\n\n ENERGIES \n\n')
outputfile.write(str(energies) + '\n')
outputfile.write('\n\n LENGTHS \n\n')
outputfile.write(str(lengths) + '\n')

outputfile.write('\n\n SEPARATED ENERGIES \n\n')
for k in range(len(energies)):
    outputfile.write(str(energies[k]) + '\n\n\n')
    outputfile.write('------------------\n\n\n')
outputfile.write("-------------------------------------------------------" + '\n\n\n')
outputfile.flush()

outputfile.write("******************************" + '\n')
build_h5file_multiplestates(namefile, lengths, energies, state_kets_dict, ket_list)
outputfile.write('Datafile ' + namefile + '.hdf5' + ' created' + '\n')
outputfile.write("******************************" + '\n')

outputfile.write("Terminated Normally")
outputfile.close()
