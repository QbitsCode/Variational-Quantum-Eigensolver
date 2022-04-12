import h5py
import numpy
from numpy import arange


def build_h5file(namefile, lengths, energies):
    h5out = h5py.File(namefile + ".hdf5", "w")
    h5out.create_dataset("LENGTHS", data=lengths)
    h5out.create_dataset("ENERGY", data=energies)
    h5out.close()


#Generate a file :
leng = arange(0.4,2.1,0.1)

ener = [-6.639450618459496,-7.070970729993077,-7.356270617665642,-7.145037981788242 ,-7.140853241084525,-7.205531763252141 ,-7.250534114704643,-7.2839118139490004 ,-7.310208090567725,-7.332153393188733, -7.3514285429508295, -7.369037524128959 ,-7.38551243054649,-7.401151005898077,-7.415994847109036, -7.430013711853318,-7.443133939224849 ]

build_h5file('./results/LiH_pec_FS_20171_temp', leng, ener)
