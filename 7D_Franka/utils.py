import numpy as np
import torch
from torch.func import jacrev, vmap, hessian, functional_call

def prismatic(xyz, rpy, axis, qi):
    # qi is a batch x 1 tensor
    batch = qi.shape[0]
    # Origin rotation from RPY ZYX convention
    cr = np.cos(rpy[0])
    sr = np.sin(rpy[0])
    cp = np.cos(rpy[1])
    sp = np.sin(rpy[1])
    cy = np.cos(rpy[2])
    sy = np.sin(rpy[2])
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr
    p0 = r00 * axis[0] * qi + r01 * axis[1] * qi + r02 * axis[2] * qi
    p1 = r10 * axis[0] * qi + r11 * axis[1] * qi + r12 * axis[2] * qi
    p2 = r20 * axis[0] * qi + r21 * axis[1] * qi + r22 * axis[2] * qi

    # Homogeneous transformation matrix
    R = torch.tensor([[r00, r01, r02],
                      [r10, r11, r12],
                      [r20, r21, r22]], dtype=torch.float32)

    R_batch = torch.tile(R[None, :, :], (batch, 1, 1))
    R_matrix = R_batch
    T_matrix = torch.tile(torch.tensor(xyz)[None, :], (batch, 1, 3)) + torch.concatenate([p0, p1, p2], dim=1)[:, :, None]
    One_matrix = torch.concatenate([torch.zeros([batch, 1, 3]), torch.ones([batch, 1, 1])], dim=2)
    T = torch.concatenate([R_matrix, T_matrix], dim=2)
    T = torch.concatenate([T, One_matrix], dim=1)

    return T


def revolute(xyz, rpy, axis, qi):
    batch = qi.shape[0]
    # Origin rotation from RPY ZYX convention
    cr = np.cos(rpy[0])
    sr = np.sin(rpy[0])
    cp = np.cos(rpy[1])
    sp = np.sin(rpy[1])
    cy = np.cos(rpy[2])
    sy = np.sin(rpy[2])
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    # joint rotation from skew sym axis angle
    cqi = torch.cos(qi)
    sqi = torch.sin(qi)
    s00 = (torch.ones(qi.shape) - cqi) * axis[0] * axis[0] + cqi
    s11 = (torch.ones(qi.shape) - cqi) * axis[1] * axis[1] + cqi
    s22 = (torch.ones(qi.shape) - cqi) * axis[2] * axis[2] + cqi
    s01 = (torch.ones(qi.shape) - cqi) * axis[0] * axis[1] - axis[2] * sqi
    s10 = (torch.ones(qi.shape) - cqi) * axis[0] * axis[1] + axis[2] * sqi
    s12 = (torch.ones(qi.shape) - cqi) * axis[1] * axis[2] - axis[0] * sqi
    s21 = (torch.ones(qi.shape) - cqi) * axis[1] * axis[2] + axis[0] * sqi
    s20 = (torch.ones(qi.shape) - cqi) * axis[0] * axis[2] - axis[1] * sqi
    s02 = (torch.ones(qi.shape) - cqi) * axis[0] * axis[2] + axis[1] * sqi

    # Homogeneous transformation matrix
    R = torch.tensor([[r00, r01, r02],
                      [r10, r11, r12],
                      [r20, r21, r22]], dtype=torch.float32)

    S_0 = torch.concatenate([s00, s10, s20], dim=1)[:, :, None]
    S_1 = torch.concatenate([s01, s11, s21], dim=1)[:, :, None]
    S_2 = torch.concatenate([s02, s12, s22], dim=1)[:, :, None]
    S = torch.concatenate([S_0, S_1, S_2], dim=2)

    R_matrix = torch.einsum('ij, kjh -> kih', R, S)
    T_matrix = torch.tile(torch.tensor(xyz)[None, :, None], (batch, 1, 1))
    One_matrix = torch.concatenate([torch.zeros([batch, 1, 3]), torch.ones([batch, 1, 1])], dim=2)
    T = torch.concatenate([R_matrix, T_matrix], dim=2)
    T = torch.concatenate([T, One_matrix], dim=1)

    return T

def torch_rpy(batch, displacement, roll, pitch, yaw):
    """Homogeneous transformation matrix with roll pitch yaw."""
    T = torch.zeros([4, 4])
    T[:3, :3] = torch_rotation_rpy(roll, pitch, yaw)
    T[:3, 3] = displacement
    T[3, 3] = 1.0
    T.reshape(1, 4, 4).repeat(batch, 1, 1)
    return T

def torch_skew_symmetric(v):
    return torch.tensor([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])


def torch_batch_skew_symmetric(v):
    zeros = torch.zeros(v.shape[0])
    batch_skew = torch.stack([torch.stack([zeros, -v[:, 2], v[:, 1]], dim=1),
                              torch.stack([v[:, 2], zeros, -v[:, 0]], dim=1),
                              torch.stack([-v[:, 1], v[:, 0], zeros], dim=1)], dim=1)
    return batch_skew



def spatial_inertia_matrix_IO(ixx, ixy, ixz, iyy, iyz, izz, mass, c, batch):
    # Expressed in the origin frame
    IO = torch.zeros([6, 6])
    cx = torch_skew_symmetric(c)
    inertia_matrix = torch.tensor([[ixx, ixy, ixz],
                                   [ixy, iyy, iyz],
                                   [ixz, iyz, izz]])

    IO[:3, :3] = inertia_matrix + mass * (cx @ cx.T)
    IO[:3, 3:] = mass * cx
    IO[3:, :3] = mass * cx.T

    IO[3, 3] = mass
    IO[4, 4] = mass
    IO[5, 5] = mass

    return torch.tile(IO[None, :, :], (batch, 1, 1))


def torch_rotation_rpy(roll, pitch, yaw):
    ## Returns a rotation matrix from roll pitch yaw. ZYX convention
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return torch.tensor([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                         [-sp, cp * sr, cp * cr]])


def batch_spatial_transform(R, r):
    ## Returns the spatial motion transform from a 3x3 rotation matrix and a 3x1 displacement vector
    X_upper = torch.concatenate([R, torch.zeros(R.shape)], dim=2)
    X_lower = torch.concatenate([torch.bmm(-R, torch_batch_skew_symmetric(r)), R], dim=2)
    X = torch.concatenate([X_upper, X_lower], dim=1)

    return X


def XJT_revolute(xyz, rpy, axis, qi):
    ## Returns the spatial transform from child link to parent link with a revolute connecting joint
    ## T has the size batch x 4 x 4
    T = revolute(xyz, rpy, axis, qi)
    rotation_matrix = T[:, :3, :3]
    displacement = T[:, :3, 3]
    return batch_spatial_transform(rotation_matrix.permute(0, 2, 1), displacement)


def XJT_prismatic(xyz, rpy, axis, qi):
    ## Returns the spatial transform from child link to parent link with a prismatic connecting joint
    T = prismatic(xyz, rpy, axis, qi)
    rotation_matrix = T[:, :3, :3]
    displacement = T[:, :3, 3]
    return batch_spatial_transform(rotation_matrix.T, displacement)


def XT(xyz, rpy, batch):
    ## Returns a general spatial transformation matrix matrix
    ## change to batch but not necessary
    rotation_matrix = torch_rotation_rpy(rpy[0], rpy[1], rpy[2])
    rotation_matrix_batch = torch.tile(rotation_matrix[None, :, :], (batch, 1, 1))
    xyz_batch = torch.tile(torch.tensor(xyz)[None, :], (batch, 1))
    return batch_spatial_transform(rotation_matrix_batch.permute(0, 2, 1), xyz_batch)


## The following part is relevant to the rotation calculation
def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def _copysign(a, b):
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    # w,x,y,z
    return torch.stack((o0, o1, o2, o3), -1)

# Logarithmic map for R^3 x S^3 manifold (with e in tangent space)
def logmap(q, q0):
    H = dQuatToDxJac(q0)  #(N, 3, 4)
    log_s3 = logmap_S3(q, q0)  #(N, 4)
    e = 2 * torch.einsum('ijk,ik->ij', H, log_s3)
    return e

# Logarithmic map for S^3 manifold (with e in ambient space)
def logmap_S3(q, q0):
    dot_product = torch.einsum('ij,ij->i', q0, q)  # (N,)
    th = acoslog(dot_product)

    u = q - dot_product.unsqueeze(-1) * q0
    u = (th.unsqueeze(-1) * u) / (torch.norm(u, dim=-1) + 1e-4).unsqueeze(-1)
    return u

# Arcosine redefinition to ensure distance between antipodal quaternions is zero
def acoslog(q):
    y = torch.acos(q)
    mask = (q >= -1.0) & (q < 0)
    y[mask] = y[mask] - np.pi
    return y

def dQuatToDxJac(q):
    # Create the Jacobian matrix for each quaternion in the batch
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]  # (N,)
    H = torch.stack([
        -q1, q0, -q3, q2,
        -q2, q3, q0, -q1,
        -q3, -q2, q1, q0
    ], dim=1).reshape(-1, 3, 4)  # (N, 3, 4)
    return H


## The following part is relevant to the training part

def factored_gradient(q_e, q_s, phi):
    grad_phi = torch.autograd.grad(outputs=phi, inputs=q_e, grad_outputs=torch.ones(phi.size()),
                                   only_inputs=True, create_graph=True, retain_graph=True)[0]
    d_e = torch.linalg.norm(q_e - q_s, dim=1, keepdim=True)
    grad = (q_e - q_s) / (d_e + 1e-12) * phi + d_e * grad_phi

    return grad[:, :, None]

def weighted_norm(gradient, weight):

    return torch.bmm(torch.bmm(gradient.permute(0, 2, 1), weight), gradient)

def fcall(net, params, x1, x2, permute=False):
    if permute:
        x = torch.concatenate([x2, x1], dim=-1)
    else:
        x = torch.concatenate([x1, x2], dim=-1)
    return functional_call(net, params, x)

# get Hessian matrix
def compute_hessian(net, params, x1, x2, permute=False):
    H = hessian(fcall, argnums=2)(net, params, x1, x2, permute)
    return H

#  get Jacobian matrix
def compute_jacobian(net, params, x1, x2, permute=False):
    J = jacrev(fcall, argnums=2)(net, params, x1, x2, permute)
    return J

def christoffel_symbols(x, metric, inv_M):  ## batch x 7 x 7
    dM_dq = vmap(jacrev(metric))(x[:, None, :]).squeeze([1, 4])

    term1 = torch.einsum('bkm, bjmi->bijk', inv_M, dM_dq)
    term2 = torch.einsum('bkm, bmij->bijk', inv_M, dM_dq)
    term3 = torch.einsum('bkm, bijm->bijk', inv_M, dM_dq)

    return 0.5 * (term1 + term2 - term3)


def laplace_beltrami_operator(net, params, x1, x2, metric, inv_M, permute=False):
    J = vmap(compute_jacobian, in_dims=(None, None, 0, 0, None))(net, params, x1, x2, permute).squeeze()  # b x 7
    H = vmap(compute_hessian, in_dims=(None, None, 0, 0, None))(net, params, x1, x2, permute).squeeze()  # b x 7 x 7

    cs_gamma = christoffel_symbols(x1, metric, inv_M)  # b x 7 x 7 x 7

    inv_M_H = torch.einsum('bij, bij -> b', inv_M, H)
    inv_M_gamma = torch.einsum('bjk, bjki -> bi', inv_M, cs_gamma)
    inv_M_gamma_J = torch.einsum('bi, bi -> b', inv_M_gamma, J)

    return inv_M_H - inv_M_gamma_J

def scale_field(q_e, upper_bound, lower_bound):
    scale = torch.ones([q_e.shape[0]])

    distance_upper = torch.linalg.norm(q_e-upper_bound, dim=1)
    distance_lower = torch.linalg.norm(q_e-lower_bound, dim=1)

    mask_upper = distance_upper < 4
    mask_lower = distance_lower < 4

    scale[mask_upper] = torch.clamp(distance_upper[mask_upper] / 4, min=0.1, max=1)
    scale[mask_lower] = torch.clamp(distance_lower[mask_lower] / 4, min=0.1, max=1)
    return scale

def scale_matrix(M):
    eigvals, eigvecs = torch.linalg.eig(M)
    clamped_eigvals = torch.clamp(eigvals.real, min=0.5)
    eigvecs = eigvecs.float()
    D = torch.diag_embed(clamped_eigvals)
    M = torch.matmul(torch.matmul(eigvecs, D), eigvecs.inverse())
    return M