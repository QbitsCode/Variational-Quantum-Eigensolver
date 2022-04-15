import h5py
import matplotlib.pyplot as plt

#TODO

states_dict = {}
file = h5py.File('/home/qlila/Cambridge/ExcitedQC/debugging/dict_data.hdf5', 'r')
dict_group_load = file['dict_data']
dict_group_keys = dict_group_load.keys()
for k in dict_group_keys:
    states_dict[k] = list(dict_group_load[k][()])

lengths = file['LENGTHS'][()]
print(lengths)
print(states_dict)

for key in states_dict:
    plt.plot(lengths, states_dict[key], label=key)
plt.legend()
plt.show()
