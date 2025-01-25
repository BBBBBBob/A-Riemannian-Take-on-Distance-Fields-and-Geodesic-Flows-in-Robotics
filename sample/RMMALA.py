## For the detailed mathematical derivation,
## please check the paper https://statmodeling.stat.columbia.edu/wp-content/uploads/2010/04/RMHMC_MG_BC_SC_REV_08_04_10.pdf#page=6.40

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.func import vmap, jacrev
from tqdm import tqdm

## Target and metric are functions which take the q as the only argument.
## Target function returns the log-likelihood.
## If target function acquires other inputs, please modified the relevant inputs in jacrev.

class RMMALA(object):
    def __init__(self, q_init, n_dim, chain, metric, target, eps=0.1, burn_in=10, num_samples=10):
        assert q_init.shape[0] != chain or q_init.shape[1] != n_dim, "Please check the dimensionality of init_q"
        self.q = q_init
        self.metric = metric
        self.target = target

        self.n_dim = n_dim
        self.chain = chain
        self.eps = eps
        self.burn_in = burn_in
        self.num_samples = num_samples

    def mean(self, q):
        M = self.metric(q)
        M_inv = torch.linalg.inv(M)
        
        dL_dq = vmap(jacrev(self.target, argnums=0), in_dims=(0, None, 0))(q[:, None, :]).squeeze([1, 2])
        dM_inv_dL_dq = torch.einsum('bij, bj -> bi', M_inv, dL_dq)
        dM_dq = vmap(jacrev(self.metric))(q[:, None, :]).squeeze([1, 4])

        M_inv_dM_dq = torch.einsum('bij, bjkl -> bikl', M_inv, dM_dq)
        M_inv_dM_dq_M_inv = torch.einsum('bijk, bjl -> bilk', M_inv_dM_dq, M_inv)
        mask = torch.zeros([self.chain, self.n_dim, self.n_dim, self.n_dim])
        mask[:, :, torch.arange(0, self.n_dim), torch.arange(0, self.n_dim)] = 1
        sum_M_inv_dM_dq_M_inv = torch.sum(M_inv_dM_dq_M_inv * mask, dim=(2, 3))

        mask = torch.zeros([self.chain, self.n_dim, self.n_dim, self.n_dim])
        mask[:, torch.arange(0, self.n_dim), torch.arange(0, self.n_dim), :] = 1
        trace = torch.sum(M_inv_dM_dq * mask, dim=(1, 2))
        G_inv_trace = torch.einsum('bij, bj -> bi', M_inv, trace)

        return (0.5 * dM_inv_dL_dq - sum_M_inv_dM_dq_M_inv + 0.5 * G_inv_trace) * self.eps**2 + q

    def cov(self, q):
        M = self.metric(q)
        M_inv = torch.linalg.inv(M)

        return M_inv * self.eps ** 2

    def update_q(self, q):
        MultiDist_q = MultivariateNormal(loc=self.mean(q), covariance_matrix=self.cov(q))
        q_propose = MultiDist_q.sample()
        log_prob_q_pro_q = MultiDist_q.log_prob(q_propose)

        MultiDist_q_pro = MultivariateNormal(loc=self.mean(q_propose), covariance_matrix=self.cov(q_propose))
        log_prob_q_q_pro = MultiDist_q_pro.log_prob(q)

        return q_propose, log_prob_q_pro_q, log_prob_q_q_pro

    def sample(self):
        samples = []
        for i in tqdm(range(self.burn_in + self.num_samples)):

            q_propose, log_prob_q_pro_q, log_prob_q_q_pro = self.update_q(self.q)
            log_prob_q = self.target(self.q)
            log_prob_q_pro = self.target(q_propose)

            log_diff = log_prob_q_pro + log_prob_q_q_pro - log_prob_q - log_prob_q_pro_q
            threshold = torch.rand(self.chain)
            mask = (threshold <= torch.exp(log_diff))

            q_star = torch.empty(self.q.shape)
            q_star[mask] = q_propose[mask]
            q_star[~mask] = self.q[~mask]

            self.q = q_star

            if i >= self.burn_in:
                samples.append(q_star.detach().cpu().numpy())

        return samples
