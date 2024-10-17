from torch.utils.data import BatchSampler, SequentialSampler
# from ccpi.filters.regularisers import ROF_TV, FGP_TV, PD_TV, SB_TV, TGV, LLT_ROF, FGP_dTV, NDF, Diff4th
import torch
from torch import nn
import torch.nn.functional as F
# from pytorch_radon.utils import PI, SQRT2, deg2rad, affine_grid, grid_sample
import skimage.data as d
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import gaussian
# from fastatomography.util import sector_mask
import os
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.nn.functional import mse_loss


class ComplexAbs(th.autograd.Function):
    '''Absolute value class for autograd'''

    @staticmethod
    def forward(ctx, tensor_in):
        output = th.abs(tensor_in)
        ctx.save_for_backward(tensor_in)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        psi_old, = ctx.saved_tensors
        psi_updated = grad_output * th.exp(1j * th.angle(psi_old))
        return psi_updated


class ComplexMul(th.autograd.Function):
    @staticmethod
    def forward(ctx, a: th.Tensor, b: th.tensor) -> th.Tensor:
        """
        :param S_split: B x K x M1 x M2 tensor
        :param psi: B x M1 x M2

        :return: B x K x M1 x M2
        """
        ctx.save_for_backward(a, b)

        # print(f'RegularizedComplexMul.forward: psi.shape: {psi.shape}')

        return a * b

    def backward(ctx, grad_output):
        # psi:        B x M1 x M2
        # S_split:    B x K x M1 x M2 tensor
        # grad_output B x K x M1 x M2

        a, b = ctx.saved_tensors
        return grad_output * b.conj(), grad_output * a.conj()


complex_abs = ComplexAbs.apply
complex_mul = ComplexMul.apply


def affine_matrix_3D(phi, theta, psi, translation):
    """Rotation matrix in 2 and 3 dimensions.
    Its rows represent the canonical unit vectors as seen from the
    rotated system while the columns are the rotated unit vectors as
    seen from the canonical system.
    Parameters
    ----------
    phi : `array-like`
        Either 2D counter-clockwise rotation angle (in radians) or first
        Euler angle.
    theta, psi : `array-like`
        Second and third Euler angles in radians. If both are ``None``, a
        2D rotation matrix is computed. Otherwise a 3D rotation is computed,
        where the default ``None`` is equivalent to ``0.0``.
        The rotation is performed in "ZXZ" rotation order, see the
        Wikipedia article `Euler angles`_.
    translation : `array-like`, shape (2, n_angles) 2D translation vector of the projection
    Returns
    -------
    mat : `numpy.ndarray`
        Rotation matrix corresponding to the given angles. The
        returned array has shape ``(ndim, ndim)`` if all angles represent
        single parameters, with ``ndim == 2`` for ``phi`` only and
        ``ndim == 3`` for 2 or 3 Euler angles.
        If any of the angle parameters is an array, the shape of the
        returned array is ``broadcast(phi, theta, psi).shape + (ndim, ndim)``.
    References
    ----------
    .. _Euler angles:
        https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    """
    cph = th.cos(phi)
    sph = th.sin(phi)
    cth = th.cos(theta)
    sth = th.sin(theta)
    cps = th.cos(psi)
    sps = th.sin(psi)
    line1 = th.stack([cph * cps - sph * cth * sps, -cph *
                     sps - sph * cth * cps, sph * sth, translation[0]], 1)
    line2 = th.stack([sph * cps + cph * cth * sps, -sph * sps +
                     cph * cth * cps, -cph * sth, th.zeros_like(translation[1])], 1)
    line3 = th.stack([sth * sps + 0 * cph, sth * cps + 0 * cph,
                     cth + 0 * (cph + cps), translation[1]], 1)
    R = th.stack([line1, line2, line3], 1)
    return R


def ray_transform_complex(vol, phi_rad, theta_rad, psi_rad, translation):
    n_theta = phi_rad.shape[0]
    R = affine_matrix_3D(phi_rad, theta_rad, psi_rad, translation)
    out_size = (n_theta, 1, vol.shape[2], vol.shape[3], vol.shape[4])
    grid = F.affine_grid(R, out_size)
    out_real = F.grid_sample(vol.real.expand(
        n_theta, 1, vol.shape[2], vol.shape[3], vol.shape[4]), grid)
    out_imag = F.grid_sample(vol.imag.expand(
        n_theta, 1, vol.shape[2], vol.shape[3], vol.shape[4]), grid)
    # print(out.shape)
    # out is (N_batch, channels, Z, Y, X)
    sino_real = th.sum(out_real, 3)
    sino_imag = th.sum(out_imag, 3)
    return sino_real + 1j * sino_imag


def ray_transform(vol, phi_rad, theta_rad, psi_rad, translation):
    n_theta = phi_rad.shape[0]
    R = affine_matrix_3D(phi_rad, theta_rad, psi_rad, translation)
    out_size = (n_theta, 1, vol.shape[2], vol.shape[3], vol.shape[4])
    grid = F.affine_grid(R, out_size)
    out = F.grid_sample(vol.expand(
        n_theta, 1, vol.shape[2], vol.shape[3], vol.shape[4]), grid)
    # print(out.shape)
    # out is (N_batch, channels, Z, Y, X)
    sino = th.sum(out, 3)
    return sino


def ray_transform_partial(vol, phi_rad, theta_rad, psi_rad, translation, projection_patches):
    n_theta = phi_rad.shape[0]
    device = vol.get_device()
    R = affine_matrix_3D(phi_rad, theta_rad, psi_rad, translation)
    out_size = (n_theta, 1, vol.shape[2], vol.shape[3], vol.shape[4])
    sino_out_shape = (n_theta, 1, vol.shape[2], vol.shape[4])
    # sino_full is (N_batch, channels, Z, X)
    sino_full = th.zeros(sino_out_shape).to(device)
    for (zs, ze, xs, xe) in projection_patches:
        grid = F.affine_grid(R, out_size)
        out = F.grid_sample(vol.expand(
            n_theta, 1, vol.shape[2], vol.shape[3], vol.shape[4]), grid)
        # print(out.shape)
        # out is (N_batch, channels, Z, Y, X)
        sino = th.sum(out, 3)
        sino_full[:, :, zs:ze, xs:xe] = sino
    return sino_full


def optimize_sino(num_iterations, sino_target, phi_rad_target, theta_rad_target, psi_rad_target, translation_target,
                  model_shape):
    device = sino_target.device
    model = th.zeros(model_shape).to(device)
    losses = []
    model.requires_grad = True
    lr_model = 40

    pars = {'algorithm': FGP_TV,
            'regularisation_parameter': 3e-3,
            'number_of_iterations': 50,
            'tolerance_constant': 1e-06,
            'methodTV': 0,
            'nonneg': 1}

    optimizer_model = Adam([model], lr_model)
    loss_fn = mse_loss
    # sampler = BatchSampler(SequentialSampler(range(sino_target.shape[0])),
    #                        batch_size=4, drop_last=False)
    for epoch in range(num_iterations):
        optimizer_model.zero_grad()
        sino_model = ray_transform(model, phi_rad_target, theta_rad_target, psi_rad_target,
                                   translation_target)
        loss = loss_fn(sino_model, sino_target)
        losses.append(loss.item())
        loss.backward()

        optimizer_model.step()
        model.requires_grad = False
        model[model < 0] = 0
        # m = model.detach().cpu().numpy().squeeze()
        # (fgp_gpu3D, info_vec_gpu) = FGP_TV(m,
        #                                    pars['regularisation_parameter'],
        #                                    pars['number_of_iterations'],
        #                                    pars['tolerance_constant'],
        #                                    pars['methodTV'],
        #                                    pars['nonneg'], 'gpu')
        # model[0, 0, :, :, :] = th.as_tensor(fgp_gpu3D, device=device)
        model.requires_grad = True

    return model, np.array(losses)

# def optimize_sino(num_iterations, sino_target, phi_rad_target, theta_rad_target, psi_rad_target, translation_target,
#                   model_shape):
#     device = sino_target.device
#     model = th.zeros(model_shape).to(device)
#     losses = []
#     model.requires_grad = True
#     lr_model = 40
#
#     optimizer_model = Adam([model], lr_model)
#     loss_fn = mse_loss
#     sampler = BatchSampler(SequentialSampler(range(sino_target.shape[0])),
#                            batch_size=4, drop_last=False)
#     for epoch in range(num_iterations):
#         for i in sampler:
#             optimizer_model.zero_grad()
#             sino_model = ray_transform(model, phi_rad_target[i], theta_rad_target[i], psi_rad_target[i],
#                                        translation_target[:,i])
#             loss = loss_fn(sino_model[i], sino_target[i])
#             losses.append(loss.item())
#             loss.backward()
#
#             optimizer_model.step()
#             model.requires_grad = False
#             model[model < 0] = 0
#             model.requires_grad = True
#
#     return model, np.array(losses)
