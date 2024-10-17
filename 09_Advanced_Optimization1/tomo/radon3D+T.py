# %%
from utils import ray_transform
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam
from skimage.filters import gaussian
import torch as th
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = th.device('cuda:0')


def mosaic(data):
    n, w, h = data.shape
    diff = np.sqrt(n) - int(np.sqrt(n))
    s = np.sqrt(n)
    m = int(s)
    # print 'm', m
    if diff > 1e-6:
        m += 1
    mosaic = np.zeros((m * w, m * h)).astype(data.dtype)
    for i in range(m):
        for j in range(m):
            if (i * m + j) < n:
                mosaic[i * w:(i + 1) * w, j * h:(j + 1) * h] = data[i * m + j]
    return mosaic


def plotmosaic(img, title='Image', savePath=None, cmap='hot', show=True, dpi=150, vmax=None):
    fig, ax = plt.subplots(dpi=dpi)
    mos = mosaic(img)
    cax = ax.imshow(mos, interpolation='nearest',
                    cmap=plt.cm.get_cmap(cmap), vmax=vmax)
    cbar = fig.colorbar(cax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.grid(False)
    plt.show()
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=dpi)


img = np.load(r'.\\shepp_phantom.npy')
img = gaussian(img, 1)
target = th.as_tensor(img).unsqueeze(0).unsqueeze(0).to(device)
# %%
fig, ax = plt.subplots(1, 3)
ax[0].imshow(img.sum(0))
ax[1].imshow(img.sum(1))
ax[2].imshow(img.sum(2))
plt.show()
# %%
n_theta = 50
phi_deg = th.linspace(0, 180, n_theta)
theta_deg = th.linspace(5, 5, n_theta)
psi_deg = th.linspace(7, 7, n_theta)

phi_rad_target = th.deg2rad(phi_deg).to(device)
theta_rad_target = th.deg2rad(theta_deg).to(device)
psi_rad_target = th.deg2rad(psi_deg).to(device)

translation_target = th.zeros((2, n_theta), device=device)
theta_rad_model = theta_rad_target.clone()
psi_rad_model = psi_rad_target.clone()

sino_target = ray_transform(
    target, phi_rad_target, theta_rad_target, psi_rad_target, translation_target)

plotmosaic(sino_target.squeeze().cpu().numpy())
# %%

model = th.zeros_like(target).to(device)
phi_rad_model = th.deg2rad(phi_deg).to(device)
theta_rad_model = th.deg2rad(theta_deg).to(device)
psi_rad_model = th.deg2rad(psi_deg).to(device)
# translation_model = th.zeros_like(translation_target).to(device)
translation_model = translation_target.clone()

phi_rad_model += th.as_tensor(np.random.uniform(-3.14, 3.14, phi_rad_target.shape[0]) / 50,
                              device=device)
fig, ax = plt.subplots()
ax.scatter(np.arange(len(phi_rad_model)),
           phi_rad_model.detach().cpu().numpy().squeeze())
ax.scatter(np.arange(len(phi_rad_model)),
           phi_rad_target.detach().cpu().numpy().squeeze())
plt.show()
# %%
model.requires_grad = True
phi_rad_model.requires_grad = True
theta_rad_model.requires_grad = False
psi_rad_model.requires_grad = False

lr_angles = 2e-3

optimizer_model = Adam([model], 50)
optimizer_phi = Adam([phi_rad_model], lr_angles)
lam = 5e-3
loss_fn = mse_loss
scheduler = ExponentialLR(optimizer_model, gamma=1 - 1e-4)
i = 0
start_refine = 100

for epoch in range(200):
    if epoch % 2 == 0 or epoch < start_refine:
        optimizer_model.zero_grad()
    else:
        optimizer_phi.zero_grad()
    sino_model = ray_transform(
        model, phi_rad_model, theta_rad_model, psi_rad_model, translation_model)
    # + lam * regularization_term(model)
    loss = loss_fn(sino_model, sino_target)
    loss.backward()

    if epoch % 2 == 0 or epoch < start_refine:
        if epoch % 10 == 0:
            print(f'{epoch}: loss = {loss.item()}')
        optimizer_model.step()
        model.requires_grad = False
        model[model < 0] = 0
        model.requires_grad = True
    else:
        if epoch > start_refine:
            optimizer_phi.step()
            if i % 10 == 0:
                print(th.max(th.abs(phi_rad_model.grad)) * lr_angles)
            i += 1
# %%
fig, ax = plt.subplots()
ax.scatter(np.arange(len(phi_rad_model)),
           phi_rad_target.detach().cpu().numpy().squeeze())
ax.scatter(np.arange(len(phi_rad_model)),
           phi_rad_model.detach().cpu().numpy().squeeze(), marker='x')

plt.show()
# %%
m = model.detach().cpu().numpy().squeeze()
t = target.detach().cpu().numpy().squeeze()
fig, ax = plt.subplots()
# imax = ax.imshow(np.abs(m - t)/t)
imax = ax.imshow(m[32])
plt.colorbar(imax)
plt.show()
