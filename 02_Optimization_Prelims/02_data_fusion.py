#%%

import cv2
import kornia
import torch as th
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

# %%
# Image Dimensions
(nx, ny) = edsMap.shape
nPix = nx * ny
nz = len(elementList)
lambdaHAADF = 1/nz


class TVDenoise(th.nn.Module):
    def __init__(self, noisy_image, lambdaTV):
        super(TVDenoise, self).__init__()
        self.lambdaTV = lambdaTV
        self.l2_term = th.nn.MSELoss(reduction='mean')
        self.regularization_term = kornia.losses.TotalVariation()
        # create the variable which will be optimized to produce the noise free image
        self.clean_image = th.nn.Parameter(
            data=noisy_image.clone(), requires_grad=True)
        self.noisy_image = noisy_image

    def forward(self):
        return self.l2_term(self.clean_image, self.noisy_image) + lambdaTV * self.regularization_term(self.clean_image)

    def get_clean_image(self):
        return self.clean_image


def reg(x, lambda_TV, ng):
  # read the image with OpenCV
  img = x
  # convert to torch tensor
  noisy_image: th.tensor = kornia.image_to_tensor(img).squeeze()  # CxHxW
  # define the total variation denoising network
  tv_denoiser = TVDenoise(noisy_image, lambda_TV)
  # define the optimizer to optimize the 1 parameter of tv_denoiser
  optimizer = th.optim.SGD(tv_denoiser.parameters(), lr=0.1, momentum=0.9)
  # run the optimization loop
  for i in range(ng):
      optimizer.zero_grad()
      loss = tv_denoiser()
      # if i % 25 == 0:
      #     print("Loss in iteration {} of {}: {:.3f}".format(i, ng, loss.item()))
      loss.backward()
      optimizer.step()
  # convert back to numpy
  img_clean: np.ndarray = kornia.tensor_to_image(tv_denoiser.get_clean_image())
  return img_clean, loss.item()


# HAADF Signal (Measurements)
b = file[mapNum+'HAADF'][:].flatten()

# Data Subtraction and Normalization
b -= np.min(b)
b /= np.max(b)

# Create Summation Matrix
A = utils.create_measurement_matrix(nx, ny,nz)

# Show Raw Data
utils.plot_elemental_images(xx, b, elementList, nx, ny, 2, 2)  # %%

# %%
regularize = True
ng = 15
lambdaTV = 0.1
r, cost = reg(edsMap, lambdaTV, ng)

fig, ax = plt.subplots()
ax.imshow(r)
plt.show()

# %%
# Convergence Parameters
gamma = 1.6
lambdaEDS = 5e-6
nIter = 30
bkg = 1e-1

# TV Min Parameters
regularize = True
ng = 15
lambdaTV = 0.1

# Auxiliary Functions


def lsqFun(inData): return 0.5 * np.linalg.norm(A.dot(inData**gamma) - b) ** 2
def poissonFun(inData): return np.sum(xx0 * np.log(inData + 1e-8) - inData)


# Main Loop
costHAADF = np.zeros(nIter, dtype=np.float32)
costEDS = np.zeros(nIter, dtype=np.float32)
costTV = np.zeros(nIter, dtype=np.float32)
for kk in tqdm(range(nIter)):

	# HAADF Update
	xx -= gamma * spdiags(xx**(gamma - 1), [0], nz*nx*ny, nz*nx*ny) * lambdaHAADF * A.transpose() * (A.dot(xx**gamma) - b) \
            + lambdaEDS * (1 - xx0 / (xx + bkg))
	xx[xx < 0] = 0

	# Regularization
	if regularize:
		for zz in range(nz):
			w, cost = reg(xx[zz*nPix:(zz+1)*nPix].reshape(ny, nx), lambdaTV, ng)
			xx[zz*nPix:(zz+1)*nPix] = w.reshape((nPix,))

	# Measure Cost Function
	costHAADF[kk] = lsqFun(xx)
	costEDS[kk] = poissonFun(xx)

	# Measure Isotropic TV
	if regularize:
		for zz in range(nz):
			w, cost = reg(xx[zz*nPix:(zz+1)*nPix].reshape(ny, nx), lambdaTV, ng)
			costTV[kk] += cost

# %%
utils.plot_elemental_images(xx, A.dot(xx**gamma),elementList,nx,ny,2,2)
# %%
utils.plot_convergence(costHAADF, lambdaHAADF, costEDS,
                       lambdaEDS, costTV, lambdaTV)

# %%
