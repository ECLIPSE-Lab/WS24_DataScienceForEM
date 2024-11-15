#%%#
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
import fusion_utils as utils
from tqdm import tqdm 
import numpy as np
import h5py
# import sys
# raise RuntimeError(sys.executable)
# %%
fname = 'CoSX_maps.h5'
mapNum = 'map7/'

# Parse Chemical Maps
elementList = ['Co', 'O', 'S']

# Load Raw Data and Reshape
file = h5py.File(fname, 'r')

print('Available EDX Maps: ', list(file))

xx = np.array([], dtype=np.float32)
for ee in elementList:

	# Read Chemical Map for Element "ee"
	edsMap = file[mapNum+ee][:, :]

	# Set Noise Floor to Zero and Normalize Chemical Maps
	edsMap -= np.min(edsMap)
	edsMap /= np.max(edsMap)

	# Concatenate Chemical Map to Variable of Interest
	xx = np.concatenate([xx, edsMap.flatten()])

# Make Copy of Raw Measurements for Poisson Maximum Likelihood Term
xx0 = xx.copy()
