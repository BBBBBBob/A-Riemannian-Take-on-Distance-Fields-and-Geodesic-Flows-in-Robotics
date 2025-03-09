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

def compute_forward_kinematics(x, config):
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


def factored_gradient(q_c, x_s, phi, config): ## grad size batch * 2
    # grad_phi = torch.autograd.grad(outputs=phi, inputs=q_c, grad_outputs=torch.ones_like(phi).to(self.config.device),
    #                         only_inputs=True,create_graph=True, retain_graph=True)[0]
    x_c = compute_forward_kinematics(q_c, config)
    d_e = torch.linalg.norm(x_c-x_s, dim=1, keepdim=True)
    grad = torch.autograd.grad(outputs=phi*d_e, inputs=q_c, grad_outputs=torch.ones_like(phi).to(self.config.device),
                                only_inputs=True,create_graph=True, retain_graph=True)[0]
    return grad[:, :, None]

def weighted_norm(gradient, weight): ## gradient size batch * 2 * 1, weight batch 2 * 2

    return torch.bmm(torch.bmm(gradient.permute(0, 2, 1), weight), gradient)