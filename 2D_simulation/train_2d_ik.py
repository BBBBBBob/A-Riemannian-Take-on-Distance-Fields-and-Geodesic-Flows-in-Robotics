import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ReLU6, ELU, Dropout, BatchNorm1d as BN, LayerNorm as LN, Tanh
import numpy as np
import math
import yaml
import utils
import os
from types import SimpleNamespace
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(CUR_PATH, '..'))
from mlp import MLPRegression_IK
from tqdm import tqdm
import random

random_seed = 11
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

config_path = os.path.join(CUR_PATH, 'config.yaml')
config_dict = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
config = SimpleNamespace(**config_dict)
metric = 'KE'   # 'KE' or 'EC'

class aniEikonalNN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.model = MLPRegression_IK(input_dims=4, 
                                   output_dims=1, 
                                   mlp_layers=[256, 256, 128, 128, 128],
                                   skips=[], 
                                   act_fn=torch.nn.Tanh,
                                   nerf=True,
                                   normalize=True)

    def train(self,batchsize,epoches=500):
        # Define model
        self.model.to(config.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                        threshold=0.01, threshold_mode='rel',
                                                        cooldown=0, min_lr=0, eps=1e-04, verbose=True)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        MSEloss = torch.nn.MSELoss()
        model_dict = {}
        for iter in tqdm(range(epoches)):
            self.model.train()
            with torch.cuda.amp.autocast():
                x_s = torch.rand(batchsize, 2).to(self.config.device) * 2 * 3.0 - 3.0
                q_c = torch.rand(batchsize, 2).to(self.config.device) * 2 * math.pi - math.pi
                q_c.requires_grad = True
                if metric == 'KE':
                    inertia = utils.compute_inertial_matrix(q_c, config)
                if metric == 'EC':
                    inertia = utils.compute_Jacobi_metric(q_c, config)
                inertia_inv = torch.linalg.inv(inertia.float())

                # # add soft margin
                # inertia_inv_copy = inertia_inv.clone()
                # mask = (q_c > self.upper_bound) | (q_c < self.lower_bound)
                # mask = mask.sum(dim=1).bool()
                # inertia_inv_copy[mask] = inertia_inv[mask]*0.01

                input_q = torch.cat([q_c, x_s], dim=1)
                tilda_phi = torch.abs(self.model(input_q))
                factored_grad = self.factored_gradient(q_c, x_s, tilda_phi, config)
                norm = self.weighted_norm(factored_grad, inertia_inv).squeeze()
                loss = MSEloss(norm, torch.ones_like(norm).to(self.config.device))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step(loss)

            if iter % 200 == 0:
                print('Iter: {}, Loss: {}'.format(iter, loss))
                model_dict[iter] = self.model.state_dict()
                torch.save(model_dict, os.path.join('./unify_model/IK_KE/', 'model_dict.pt'))

if __name__ == "__main__":
    anieik = aniEikonalNN(config)
    anieik.train(10000, epoches=50000)
