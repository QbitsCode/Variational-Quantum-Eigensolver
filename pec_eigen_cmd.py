from numpy import linalg, sort, power, subtract, arange
from time import time
from VQE_utils import build_molecule, build_basis
from pyscf import gto
from VQE_FS import VQE_fs
from VQE_ground import VQE_g
from myutils import build_h5file

namefile = "./outputs/" + "LiH_pec_eigenvalues_4e_smallbasis_test"

nel = 2
nqbits = 4

outputfile = open(namefile + ".txt", "a+")
outputfile.write('**********************************************' + '\n\n')
outputfile.write('EXACT POTENTIAL ENERGY CURVE' + '\n\n')
outputfile.write('**********************************************' + '\n')
outputfile.write('States with ' + str(nel) + ' electrons only. \n\n\n\n')
outputfile.flush()

mybasis = build_basis({
    'H': gto.load("./basisets/small_custom_basis.nw", 'H'),
    'Li': gto.load("./basisets/small_custom_basis.nw", 'Li'),
}, 'Li-1,2s-ccpvdz & H-1s-sto3g')

wrong_index = []

for k in range(0, 2**nqbits):
    kbin = bin(k)[2:]
    count = 0
    for j in range(len(kbin)):
        if kbin[j] == '1':
            count += 1
    if count != nel:
        wrong_index.append(k)

lengths = []
energies = [[] for _ in range(2**nqbits - len(wrong_index))]

for length in arange(0.2, 6, 0.2):

    LiH = build_molecule([('Li', [0, 0, 0]), ('H', [0, 0, length])], mybasis, 0, 1, 'LiH')
    # H2 = build_molecule([('H', [0, 0, 0]), ('H', [0, 0, length])], 'sto-3g', 0, 1, 'H2')

    mymlc = LiH

    outputfile.write(mymlc.name + ' : ' + str(mymlc.geometry) + '\n\n')

    myvqeG = VQE_g(
        mymlc,
        nlayers=0,
    )

    mathamG = myvqeG.qHam.to_operator()
    start = time()
    eigen = linalg.eig(mathamG)
    end = time()

    t = end - start
    eigenstates = eigen[1]
    eigenval = eigen[0]
    phyvalues = []

    for k in range(len(eigenval)):
        estate = eigenstates[:, k]
        takeit = True
        for i in wrong_index:
            if takeit and estate[i] < 0.000001:
                pass
            else:
                takeit = False
                break
        if takeit:
            phyvalues.append(eigenval[k])

    outputfile.write('Bond Length : ' + str(length) + '\n\n')
    outputfile.write('vp of H in Eel= ' + str(phyvalues) + '\n\n')
    outputfile.write('vp of H in Etot= ' + str(phyvalues + myvqeG.molecule.nuclear_energy) + '\n\n')
    outputfile.write('runtime : ' + str(t // 3600) + ' h ' + str((t % 3600) // 60) + ' min ' + str((t % 3600) % 60) + ' sec' + '\n')
    outputfile.write('------------------------------------------------------- \n\n')
    outputfile.flush()
    lengths.append(length)
    phyvalues = sort(phyvalues)
    for k in range(len(phyvalues)):
        energies[k].append(phyvalues[k] + myvqeG.molecule.nuclear_energy)

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
build_h5file(namefile, lengths, energies)
outputfile.write('Datafile ' + namefile + '.hdf5' + ' created' + '\n')
outputfile.write("******************************" + '\n')

outputfile.write("Terminated Normally")
outputfile.close()
