import torch
from torch.func import jacrev

def compute_inertial_matrix(x, config):
    link_length = torch.tensor(config.link_length, dtype=torch.float32).to(config.device).unsqueeze(0)
    link_mass = torch.tensor(config.link_mass, dtype=torch.float32).to(config.device).unsqueeze(0).expand(2, -1)
    B = x.shape[0]
    L = torch.tensor([[1, 0], [1, 1]], dtype=torch.float32).to(config.device)
    G_l = torch.matmul(link_length.transpose(0, 1), link_length).unsqueeze(0).repeat(B, 1, 1)
    G_cos = torch.cos(torch.matmul(x, L.transpose(0, 1)).unsqueeze(2) - torch.matmul(x, L.transpose(0, 1)).unsqueeze(1))
    U = torch.multiply(torch.tensor([[1, 1], [0, 1]], dtype=torch.float32).to(config.device), link_mass)
    G_u = torch.matmul(U, U.transpose(0, 1)).unsqueeze(0).repeat(B, 1, 1)
    G = G_l * G_cos * G_u
    M = torch.matmul(torch.matmul(G.transpose(1, 2), L).transpose(1, 2), L)

    return M

def compute_Jacobi_metric(x, config):
    M = compute_inertial_matrix(x, config)
    P = get_potential_energy(x, config)
    E = config.total_energy * torch.ones_like(P)
    M = 1 / (2 * (E - P)).unsqueeze(-1) * M

    return M 

def get_potential_energy(x, config):
    link_length = torch.tensor(config.link_length, dtype=torch.float32).to(config.device).unsqueeze(0)
    link_mass = torch.tensor(config.link_mass, dtype=torch.float32).to(config.device).unsqueeze(0)
    h_1 = link_length[:, 0] * torch.sin(x[:, 0])
    h_2 = h_1 + link_length[:, 1] * torch.sin(x[:, 0] + x[:, 1])
    P = (h_1 * link_mass[:, 0] + h_2 * link_mass[:, 1]) * config.gravity
    return P.unsqueeze(1)

def compute_Jacobian(x,config):
    B = x.size(0)
    L = torch.tril(torch.ones([2, 2])).expand(B, -1, -1).float().to(config.device)
    x = x.unsqueeze(2)
    link_length = torch.tensor(config.link_length, dtype=torch.float32).to(config.device)
    diag_length = torch.diag(link_length).unsqueeze(0)
    J = torch.stack([
        torch.matmul(torch.matmul(-torch.sin(torch.matmul(L, x)).transpose(1, 2), diag_length), L),
        torch.matmul(torch.matmul(torch.cos(torch.matmul(L, x)).transpose(1, 2), diag_length), L),
    ], dim=1).squeeze()
    if B == 1:
        J = J.unsqueeze(0)
    return J


def compute_AD_Jacobian(x, config):
    return jacrev(compute_forward_kinematics)(x, config)

def compute_forward_kinematics(x,config):
    B = x.size(0)
    L = torch.tril(torch.ones([2, 2])).expand(B,- 1, -1).float().to(config.device)
    x = x.unsqueeze(2)
    link_length = torch.tensor(config.link_length, dtype=torch.float32).to(config.device)
    f = torch.stack([
        torch.matmul(link_length, torch.cos(torch.matmul(L, x))),
        torch.matmul(link_length, torch.sin(torch.matmul(L, x)))], dim=0).transpose(0, 1).squeeze()
    if B == 1:
        f = f.unsqueeze(0)
    return f 



# def quiver(ax,X,Y,U,V,subsampling=tuple(),**kwargs):
# 	if np.ndim(subsampling)==0: subsampling = (subsampling,)*2
# 	where = tuple(slice(None,None,s) for s in subsampling)
# 	def f(Z): return Z.__getitem__(where)
# 	return ax.quiver(f(X),f(Y),f(U),f(V),**kwargs)
#
# def set_uniform_pts(tips = [6,6]):
#     X,Y = torch.meshgrid(torch.linspace(-3.0, 3.0, tips[0]), torch.linspace(-3.0, 3.0, tips[1]))
#     pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
#     return pts
#
#
# def check_metric(M):
#     eigvals, eigvecs = torch.linalg.eig(M.float())
#     values = eigvals.real
#     # print(values)
#     print('min:',values.min(dim=0)[0],'\n','max:', values.max(dim=0)[0],'\n','mean:', values.mean(dim=0))
#     # clamped_eigvals = torch.clamp(eigvals.real, min=1.0,max=100.0).float()
#     # # print(clamped_eigvals.min(), clamped_eigvals.max())
#     # eigvecs = eigvecs.float()
#     # clamped_eigvals = clamped_eigvals.float()
#     # D = torch.diag_embed(clamped_eigvals)
#     # # print(D.shape)
#     # M = torch.matmul(torch.matmul(eigvecs, D), eigvecs.inverse())
#     return M
