from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable



class LinearCombinator(nn.Module):
    def __init__(self, input_shape):
        super(LinearCombinator, self).__init__()

        self.l1 = nn.Linear(input_shape, input_shape)
        self.l2 = nn.Linear(input_shape, input_shape)

    def forward(self, x_recon, x_short):
        return self.l1(x_recon) + self.l2(x_short)

class MLPCombinator(nn.Module):
    def __init__(self, input_shape):
        super(MLPCombinator, self).__init__()

        self.l1 = nn.Linear(3*input_shape, input_shape)
        self.l2 = nn.Linear(4*input_shape, input_shape)

        self.sig = nn.Sigmoid()

    def forward(self, u, z):
        x = torch.cat((u, z, u*z), dim=1)
        s = self.sig(self.l1(x))
        return self.l2(torch.cat((x,s), dim=1))


class Encoder(nn.Module):
    def __init__(self, layers, acts, cuda, noise_std=0.):
        super().__init__()
        self.cuda = cuda
        self.acts = acts
        self.layers = layers
        self.noise_std = noise_std

    def noise(self, x):
        noise = Variable(torch.randn(x.size()) * self.noise_std)
        if self.cuda:
            noise = noise.cuda()
        return x + noise

    def encode(self, layer, act, x):
        if self.noise_std > 0:
            y = self.noise(layer(x))
        else:
            y = layer(x)
        y_act = act(y)
        return y, y_act

    def forward(self, e0):
        if self.noise_std > 0:
            e0 = self.noise(e0)

        e_act = e0
        pres = [e0]
        acts = []

        for (layer,act) in zip(self.layers, self.acts):
            e_pre, e_act = self.encode(layer, act, e_act)
            pres += [e_pre]
            acts += [e_act]

        return pres, acts

class LN(nn.Module):
    def __init__(self, parallel_encoder=False, noise_std=0.3, cuda=False):
        super(LN, self).__init__()

        self.parallel_encoder = parallel_encoder

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()

        self.e1 = nn.Linear(784, 300)
        self.e2 = nn.Linear(300, 20)
        self.e3 = nn.Linear(20, 10)

        self.d3 = nn.Linear(10, 20)
        self.d2 = nn.Linear(20, 300)
        self.d1 = nn.Linear(300, 784)

        self.g3 = MLPCombinator(10)
        self.g2 = MLPCombinator(20)
        self.g1 = MLPCombinator(300)
        self.g0 = MLPCombinator(784)

        acts = [self.relu, self.relu, self.softmax]
        layers = [self.e1, self.e2, self.e3]

        self.e_noisy = Encoder(layers=layers, acts=acts, cuda=cuda, noise_std=noise_std)
        self.e_clean = Encoder(layers=layers, acts=acts, cuda=cuda, noise_std=0.0)

        if self.parallel_encoder:
            self.e_noisy = torch.nn.DataParallel(self.e_noisy, device_ids=[0])
            self.e_clean = torch.nn.DataParallel(self.e_clean, device_ids=[0])


    def forward(self, x):
        x = x.view(-1, 784)

        (e0_clean,e1_clean,e2_clean,e3_clean), (e1_clean_act,e2_clean_act,e3_clean_act) = self.e_clean(x)
        (e0_noisy,e1_noisy,e2_noisy,e3_noisy), (e1_noisy_act,e2_noisy_act,e3_noisy_act) = self.e_noisy(x)

        l3_recon = self.g3(e3_noisy_act, e3_noisy)
        u2 = self.d3(l3_recon)
        l2_recon = self.g2(u2, e2_noisy)
        u1 = self.d2(l2_recon)
        l1_recon = self.g1(u1, e1_noisy)
        u0 = self.d1(l1_recon)
        l0_recon = self.g0(u0, e0_noisy)

        refs = [e0_clean, e1_clean, e2_clean, e3_clean]
        recs = [l0_recon, l1_recon, l2_recon, l3_recon]
        sup_noisy = e3_noisy_act
        sup_clean = e3_clean_act

        return sup_noisy, sup_clean, (refs, recs)


    """

          |   e3_rc
          o---------o
          |   e3_sc |   u3
         (f)   .-->(g)  g3
          o----'    |
    e3  [ooo]     [ooo] d3
          |         |   u2
         (f)   .-->(g)  g2
          o----'    |
    e2  [ooo]     [ooo] d2
          |         |   u1
         (f)   .-->(g)  g1
          o----'    |
    e1  [ooo]     [ooo] d1
          |         |   u0
          o------->(g)  g0
          |         |
    e0  [ooo]     [ooo]

    """

    def sample_by_example(self, x, k=1):
        samples = []

        for _ in range(k):
            (e0_noisy,e1_noisy,e2_noisy,e3_noisy), (e1_noisy_act,e2_noisy_act,e3_noisy_act) = self.e_noisy(x)
            l3_recon = self.g3(e3_noisy_act, e3_noisy)
            u2 = self.d3(l3_recon)
            l2_recon = self.g2(u2, e2_noisy)
            u1 = self.d2(l2_recon)
            l1_recon = self.g1(u1, e1_noisy)
            u0 = self.d1(l1_recon)
            l0_recon = self.g0(u0, e0_noisy)

            x = l0_recon

            samples.append(x)

        return samples


    def sample(self, z, k=1):
        """
        - we sample from \hat{P}(z) several times before using the reconstruction
        - we do this by repeatedly using the reconstruction as input

        References:
        - Improving Sampling from Generative Autoencoders with Markov Chains
        """
        e3_rc = e3_sc = z

        g3 = self.g3(e3_rc, e3_sc)
        u2 = self.d3(g3)
        #e3_sc = self.e3(self.relu(u2))

        # TODO

