import torch
from torch.func import vmap, jacrev

## If target function acquires other inputs, please modified the relevant inputs in jacrev.
## dH_dq_auto and dH_dp_auto directly apply the automatic differentiation.
## dH_dq and dH_dp partially use the automatic differentiation.

class LeapFrog(object):
    def __init__(self, metric, target, n_dim, fix_iter, eps, q, p):
        self.metric = metric
        self.target = target

        self.n_dim = n_dim
        self.fix_iter = fix_iter
        self.eps = eps

        self.q = q
        self.p = p

    def H(self, q, p):
        M = self.metric(q)
        inv_M = torch.linalg.inv(M)
        det_M = torch.linalg.det(M)

        L_term = self.target(q)
        log_term = 0.5 * torch.log(((2*torch.pi)**self.n_dim)*det_M)
        kinetic_term = 0.5 * torch.sum(torch.einsum('bi, bij -> bj', p, inv_M) * p, dim=1)

        return -L_term + log_term + kinetic_term

    def dH_dq_auto(self, q, p):
        dH_dq_auto = vmap(jacrev(self.H, argnums=0))(q[:, None, :], p[:, None, :]).squeeze([1, 2])

        return dH_dq_auto

    def dH_dp_auto(self, q, p):
        dH_dp_auto = vmap(jacrev(self.H, argnums=1))(q[:, None, :], p[:, None, :]).squeeze([1, 2])

        return dH_dp_auto

    def dH_dq(self, q, p):
        M = self.metric(q)
        M_inv = torch.linalg.inv(M)

        dM_dq = vmap(jacrev(self.metric))(q[:, None, :]).squeeze([1, 4])
        dL_dq = vmap(jacrev(self.target, argnums=0), in_dims=(0, None, 0))(q[:, None, :]).squeeze([1, 2])

        M_inv_dM_dq = torch.einsum('bij, bjkl -> bikl', M_inv, dM_dq)

        mask = torch.zeros([q.shape[0], self.n_dim, self.n_dim, self.n_dim])
        mask[:, torch.arange(0, self.n_dim), torch.arange(0, self.n_dim), :] = 1
        trace = torch.sum(M_inv_dM_dq*mask, dim=(1, 2))

        p_M_inv = torch.einsum('bi, bij -> bj', p, M_inv)
        p_M_inv_dM_dq = torch.einsum('bi, bijk -> bjk', p_M_inv, dM_dq)
        dKin_dq = torch.einsum('bij, bi -> bj', p_M_inv_dM_dq, p_M_inv)

        return -dL_dq + 0.5*trace - 0.5 * dKin_dq

    def dH_dp(self, q, p):
        M = self.metric(q)
        M_inv = torch.linalg.inv(M)

        return torch.einsum('bij, bj -> bi', M_inv, p)  # bx7

    def update_p_half(self):
        p_n = self.p.clone()
        for _ in range(self.fix_iter):
            self.p = p_n - self.eps * 0.5 * self.dH_dq_auto(self.q, self.p)

    def update_q(self):
        q_n = self.q.clone()
        for _ in range(self.fix_iter):
            self.q = q_n + self.eps * 0.5 * (self.dH_dp(q_n, self.p) + self.dH_dp(self.q, self.p))

    def update_p(self):
        self.p = self.p - self.eps * 0.5 * self.dH_dq_auto(self.q, self.p)

    def update_all(self):
        self.update_p_half()
        self.update_q()
        self.update_p()
