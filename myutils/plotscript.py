import h5py
import matplotlib.pyplot as plt

#TODO

namefile = "../outputs/testH2.hdf5"

datafile = h5py.File(namefile, 'r')

mylen = datafile['LENGTHS'][()]
myy = datafile['ENERGY2'][()]

plt.plot(mylen, myy)

plt.show()

datafile.close()