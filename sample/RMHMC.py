## For the detailed mathematical derivation,
## please check the paper https://statmodeling.stat.columbia.edu/wp-content/uploads/2010/04/RMHMC_MG_BC_SC_REV_08_04_10.pdf#page=6.40

import torch
import random
from torch.distributions.multivariate_normal import MultivariateNormal
from LeapFrog import LeapFrog
from tqdm import tqdm

## Target and metric are functions which take the q as the only argument.
## Target function returns the log-likelihood.
## q is the variable and p is the momentum

class RMHMC(object):
    def __init__(self, q_init, n_dim, chain, metric, target, eps=0.1, fix_iter=8, frog_iter=8, burn_in=10, num_samples=10):
        assert q_init.shape[0] != chain or q_init.shape[1] != n_dim, "Please check the dimensionality of init_q"
        self.chain = chain
        self.n_dim = n_dim
        MultiDist = MultivariateNormal(loc=torch.zeros([chain, n_dim]),
                                       covariance_matrix=metric(q_init))
        p_init = MultiDist.sample()

        self.integrator = LeapFrog(metric, target, n_dim, fix_iter, eps, q_init, p_init)
        self.frog_iter = frog_iter
        self.burn_in = burn_in
        self.num_samples = num_samples

    def sample(self):
        samples = []
        for i in tqdm(range(self.burn_in + self.num_samples)):

            q_start = self.integrator.q.clone()
            p_start = self.integrator.p.clone()

            rand_frog_iter = random.randint(self.frog_iter, self.frog_iter+10)
            for _ in range(rand_frog_iter):
                self.integrator.update_all()

            H_diff = self.integrator.H(q_start, p_start) - self.integrator.H(self.integrator.q, self.integrator.p)
            accept = torch.minimum(torch.ones(self.chain), torch.exp(H_diff))
            threshold = torch.rand(self.chain)
            mask = (threshold <= accept)

            q_star = torch.empty(q_start.shape)
            q_star[mask] = self.integrator.q[mask]
            q_star[~mask] = q_start[~mask]

            self.integrator.q = q_star
            MultiDist = MultivariateNormal(loc=torch.zeros([q_star.shape[0], self.n_dim]),
                                           covariance_matrix=self.integrator.metric(q_star))
            self.integrator.p = MultiDist.sample()

            if i >= self.burn_in:
                samples.append(q_star.detach().cpu().numpy())

        return samples

