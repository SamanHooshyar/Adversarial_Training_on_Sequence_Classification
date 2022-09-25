#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.autograd as ag
import numpy
from scipy import optimize
from scipy.optimize import check_grad


# In[ ]:


class ProximalMapping(ag.Function):
    @classmethod
    def __init__(cls, hidden_size, epsilon):
        cls.epsilon = epsilon
        cls.I = torch.eye(hidden_size)

    @classmethod
    def forward(cls, ctx, G, c):

        N, hidden_size, D = G.size()

        Gt = G.permute(0, 2, 1).contiguous()

        I = cls.I.unsqueeze(0).repeat(N, 1, 1)

        assert I.size() == (N, hidden_size, hidden_size)

        prox_int = torch.inverse(I + cls.epsilon*torch.matmul(G, Gt))

        assert prox_int.size() == (N, hidden_size, hidden_size)

        c = c.view(N, hidden_size, 1)

        prox_c = torch.matmul(prox_int, c).squeeze(-1)

        ctx.save_for_backward(prox_int, G, prox_c)

        return prox_c

    @classmethod
    def backward(cls, ctx, grad_c):

        N, hidden_size = grad_c.size()

        prox_int, G, prox_c = ctx.saved_tensors

        prox_c = prox_c.view(N, hidden_size, 1)
        prox_ct = prox_c.view(N, 1, hidden_size)

        grad_c = grad_c.view(N, hidden_size, 1)
        a = torch.matmul(prox_int, grad_c).view(N, hidden_size, 1)
        at = a.permute(0, 2, 1).contiguous()

        subcomp = (torch.matmul(a, prox_ct).view(N, hidden_size, hidden_size) +
                   torch.matmul(prox_c, at).view(N, hidden_size, hidden_size))

        dG = -torch.matmul(subcomp, G)  
        ds = torch.matmul(grad_c.view(N, 1, hidden_size), prox_int)  
        ds = ds.view(N, hidden_size)

        return dG, ds


# In[ ]:


class ProximalLSTMCell(nn.Module):
    def __init__(self, lstm, hidden_size, epsilon):  # feel free to add more input arguments as needed
        super(ProximalLSTMCell, self).__init__()
        self.lstm = lstm   # use LSTMCell as blackbox
        self.proxmap = ProximalMapping(hidden_size, epsilon)
        self.ones = torch.ones(1)

    def compute_jacobian(self, v, w):
        N, Dv = v.size()
        Dw = w.size(-1)
        vector = self.ones.view(1).repeat(N)
        J = [torch.autograd.grad(v[:, i], [w],
                             grad_outputs=vector,
                             retain_graph=True,
                             create_graph=True)[0].view(N, 1, Dw)
             for i in range(Dv)]
        J = torch.cat(J, dim=1)

        return J

    def forward(self, input, hx, cx, G_final=False):

        hx, cx = self.lstm(input, (hx, cx))

        if not G_final:
            G = self.compute_jacobian(cx, input)
            self.G = G
            cx = self.proxmap.apply(G, cx)

        return hx, cx

