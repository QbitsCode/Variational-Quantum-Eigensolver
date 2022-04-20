import h5py
import numpy
from numpy import arange


def build_h5file_multiplestates(namefile, lengths, energies, state_dict={}, right_kets=[]):
    h5out = h5py.File(namefile + ".hdf5", "w")
    h5out.create_dataset("LENGTHS", data=lengths)
    h5out.create_dataset("ENERGY", data=energies)
    h5out.create_dataset("KETS", data=right_kets)
    for j in range(len(state_dict)):
        dict_group = h5out.create_group('state' + str(j))
        for k, v in state_dict['e' + str(j)].items():
            dict_group[k] = v
    h5out.close()


def build_h5file_onestate(namefile, lengths, energies, state_dict={}, right_kets=[]):
    h5out = h5py.File(namefile + ".hdf5", "w")
    h5out.create_dataset("LENGTHS", data=lengths)
    h5out.create_dataset("ENERGY", data=energies)
    h5out.create_dataset("KETS", data=right_kets)
    dict_group = h5out.create_group('state0')
    for k, v in state_dict.items():
        dict_group[k] = v

    h5out.close()


#Generate a file :
# leng =

# ener =

# build_h5file('./results/LiH_pec_FS_20171_temp', leng, ener)
