import torch
torch.set_printoptions(profile="full", precision=16)
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import torch.optim as optim

from tnn import *
from integration import *
from quadrature import *

import itertools
import os
import sys
import time
import math

pi = 3.14159265358979323846
dtype = torch.double
device = 'cuda:3'
name = 'x_free_y_per_shift'

m=1
dim=1
a, b = 0, 1

p_A = 2
p_chi1 = 20
p_eta = 20
p_u0 = 18
p_w = 20

x1, w1 = composite_quadrature_1d(8, 0, 1, 200, device=device)
x2, w2 = composite_quadrature_1d(8, 0, 1, 200, device=device)

dummy = torch.ones((1, 1), dtype=dtype, device=device)

x_error_0 = torch.zeros(1, dtype=dtype, device=device)

N = len(x1)

A = torch.ones((2, 2, N), dtype=dtype, device=device)
A[0, 0, :] = 2 + x1
A[1, 1, :] = torch.sin(2*pi * x2)
alpha_A = torch.ones(2, dtype=dtype, device=device)

A_partial_x = torch.ones((2, 2, N), dtype=dtype, device=device)
A_partial_x[:,1,:] = 0
alpha_A_partial_x = torch.ones(2, dtype=dtype, device=device)


def bd_u0(x):
    # return torch.sin(pi * x)
    return (x - a) * (b - x)

def grad_bd_u0(x):
    # return pi * torch.cos(pi * x)
    return a + b - 2 * x

def grad_grad_bd_u0(x):
    # return -pi * pi * torch.sin(pi * x)
    return -2 * torch.ones_like(x)

def grad_grad_grad_bd_u0(x):
    # return -pi * pi * pi * torch.cos(pi * x)
    return 0 * torch.ones_like(x)
    
model_chi1_x = torch.load('./model_chi1/x_free_y_per_final_chi1_x.pkl', map_location=device)
model_chi1_y1 = torch.load('./model_chi1/x_free_y_per_final_chi1_y1.pkl', map_location=device)

model_eta1_x = torch.load('./model_eta1/x_free_y_per_shift_final_eta1_x.pkl', map_location=device)
model_eta1_y1 = torch.load('./model_eta1/x_free_y_per_shift_final_eta1_y1.pkl', map_location=device)

model_eta2_x = torch.load('./model_eta2/x_free_y_per_shift_final_eta2_x.pkl', map_location=device)
model_eta2_y1 = torch.load('./model_eta2/x_free_y_per_shift_final_eta2_y1.pkl', map_location=device)

model_u0 = torch.load('./model_u0/x_free_y_per_final_u0.pkl', map_location=device)

with torch.no_grad():
    chi1 = torch.ones((2, p_chi1, N), dtype=dtype, device=device)
    chi1_grad_phi = torch.zeros((2, p_chi1, N), dtype=dtype, device=device)
    chi1[0:1, :, :], chi1_grad_phi[0:1, :, :] = model_chi1_x(w1, x1, need_grad=1)
    chi1[1:2, :, :], chi1_grad_phi[1:2, :, :] = model_chi1_y1(w2, x2, need_grad=1)
    alpha_chi1 = model_chi1_x.scaling_par()

    A0 = torch.ones((1, p_A + p_A * p_chi1, N), dtype=dtype, device=device)
    A0[0:1, 0:p_A, :] = A[0:1, :, :]
    A0[0:1, p_A:p_A + p_A * p_chi1, :] = torch.einsum('dpn,dqn->dpqn', A[0:1, :, :], chi1[0:1, :, :]).flatten(start_dim=1, end_dim=2)

    alpha_A0 = torch.ones(p_A + p_A * p_chi1, dtype=dtype, device=device)
    alpha_A0[0:p_A] = torch.sum(w2 * A[1, :, :], dim=-1).squeeze(dim=0) * alpha_A
    alpha_A0[p_A:p_A + p_A * p_chi1] = (torch.einsum('pn,qn->pq', w2 * A[1, :, :], chi1_grad_phi[1, :, :]) * torch.outer(alpha_A, alpha_chi1)).view(-1)

    A0_grad_x = torch.ones((1, p_A + p_A * p_chi1, N), dtype=dtype, device=device)
    A0_grad_x[0:1, 0:p_A, :] = A_partial_x[0:1, :, :]
    A0_grad_x[0:1, p_A:p_A + p_A * p_chi1, :] = torch.einsum('dpn,dqn->dpqn', A_partial_x[0:1, :, :], chi1[0:1, :, :]).flatten(start_dim=1, end_dim=2) \
                                                        + torch.einsum('dpn,dqn->dpqn', A[0:1, :, :], chi1_grad_phi[0:1, :, :]).flatten(start_dim=1, end_dim=2)

    alpha_A0_grad_x = alpha_A0
    


def criterion_0(model_u0):
    u0_phi, u0_grad_phi = model_u0(w1, x1, need_grad=1)
    
    alpha = torch.ones(p_u0, device=device, dtype=dtype)
    A = Int2TNN_amend_1d_with_coe(w1, w1, alpha, u0_phi, alpha, u0_phi, u0_grad_phi, u0_grad_phi, alpha_A0, A0,
                                      if_sum=False)
    M = Int2TNN(w1, alpha, u0_phi, alpha, u0_phi, if_sum=False)

    L = torch.linalg.cholesky(M)  # Cholesky分解：M = L L^T
    C = torch.linalg.solve(L, A.t()).t()  # C = A L^{-T}
    D = torch.linalg.solve(L, C)  # D = L^{-1} A L^{-T}
    E, V = torch.linalg.eigh(D)  # 解 D 的特征值问题
    ind = torch.argsort(E)
    lam1 = E[ind]
    u1 = torch.linalg.solve(L.t(),V[:,ind])

    return lam1, u1

lam, u = criterion_0(model_u0)
print('lam[:m]=', lam[:m])

with torch.no_grad():
    chi1_x, chi1_x_grad, chi1_x_grad_grad = model_chi1_x(w1, x1, need_grad=2)
    chi1_y = model_chi1_y1(w2, x2, need_grad=0)
    alpha_chi1 = model_chi1_x.scaling_par()
    
    eta1_x, eta1_x_grad = model_eta1_x(w1, x1, need_grad=1)
    eta1_y, eta1_y_grad = model_eta1_y1(w2, x2, need_grad=1)
    alpha_eta1 = model_eta1_x.scaling_par()
    
    eta2_x, eta2_x_grad = model_eta2_x(w1, x1, need_grad=1)
    eta2_y, eta2_y_grad= model_eta2_y1(w2, x2, need_grad=1)
    alpha_eta2 = model_eta2_x.scaling_par()

    u0_x, u0_x_grad, u0_x_grad_grad, u0_x_grad_grad_grad = model_u0(w1, x1, need_grad=3)
    alpha_u0 = u[:,0]

    chi1_y1_norm_phi = model_chi1_y1(w2, x2, normed=False)
    chi1_y1_norm = torch.sqrt(torch.sum(w2 * chi1_y1_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    chi1_y1_shift = model_chi1_y1(dummy, x_error_0, need_grad=0, normed=False) #[d,p,1]
    chi1_y1_shift = chi1_y1_shift / chi1_y1_norm  # [d,p,1]

    # u0_x < A chi1_xx >
    part1 = torch.einsum('don,dpn,dqn->dopqn', u0_x_grad, A[0:1,:,:], chi1_x_grad_grad).flatten(start_dim=1, end_dim=3)
    part1_int = torch.einsum('pn,qn,n->pq', A[1,:,:], (chi1_y - chi1_y1_shift)[0,:,:], w2)
    alpha_part1 = (torch.einsum('o,p,q->opq', alpha_u0, alpha_A, alpha_chi1) * part1_int).view(-1)
    
    # u0_x < A eta1_xy >
    part2 = torch.einsum('don,dpn,dqn->dopqn', u0_x_grad, A[0:1,:,:], eta1_x_grad).flatten(start_dim=1, end_dim=3)
    part2_int = torch.einsum('pn,qn,n->pq', A[1,:,:], (eta1_y_grad)[0,:,:], w2)
    alpha_part2 = (torch.einsum('o,p,q->opq', alpha_u0, alpha_A, alpha_eta1) * part2_int).view(-1)

    # u0_x < A_x chi1_x >
    part3 = torch.einsum('don,dpn,dqn->dopqn', u0_x_grad, A_partial_x[0:1,:,:], chi1_x_grad).flatten(start_dim=1, end_dim=3)
    part3_int = torch.einsum('pn,qn,n->pq', A_partial_x[1,:,:], (chi1_y - chi1_y1_shift)[0,:,:], w2)
    alpha_part3 = (torch.einsum('o,p,q->opq', alpha_u0, alpha_A_partial_x, alpha_chi1) * part3_int).view(-1)

    # u0_x < A_x eta1_y >
    part4 = torch.einsum('don,dpn,dqn->dopqn', u0_x_grad, A_partial_x[0:1,:,:], eta1_x).flatten(start_dim=1, end_dim=3)
    part4_int = torch.einsum('pn,qn,n->pq', A_partial_x[1,:,:], (eta1_y_grad)[0,:,:], w2)
    alpha_part4 = (torch.einsum('o,p,q->opq', alpha_u0, alpha_A_partial_x, alpha_eta1) * part4_int).view(-1)

    # 2 u0_xx < A chi1_x >
    part5 = torch.einsum('don,dpn,dqn->dopqn', u0_x_grad_grad, A[0:1,:,:], chi1_x_grad).flatten(start_dim=1, end_dim=3)
    part5_int = 2 * torch.einsum('pn,qn,n->pq', A[1,:,:], (chi1_y - chi1_y1_shift)[0,:,:], w2)
    alpha_part5 = (torch.einsum('o,p,q->opq', alpha_u0, alpha_A, alpha_chi1) * part5_int).view(-1)

    # u0_xx < A eta1_y >
    part6 = torch.einsum('don,dpn,dqn->dopqn', u0_x_grad_grad, A[0:1,:,:], eta1_x).flatten(start_dim=1, end_dim=3)
    part6_int = torch.einsum('pn,qn,n->pq', A[1,:,:], (eta1_y_grad)[0,:,:], w2)
    alpha_part6 = (torch.einsum('o,p,q->opq', alpha_u0, alpha_A, alpha_eta1) * part6_int).view(-1)

    # u0_xx < A eta2_xy >
    part7 = torch.einsum('don,dpn,dqn->dopqn', u0_x_grad_grad, A[0:1,:,:], eta2_x_grad).flatten(start_dim=1, end_dim=3)
    part7_int = torch.einsum('pn,qn,n->pq', A[1,:,:], (eta2_y_grad)[0,:,:], w2)
    alpha_part7 = (torch.einsum('o,p,q->opq', alpha_u0, alpha_A, alpha_eta2) * part7_int).view(-1)

    # u0_xx < A_x chi1 >
    part8 = torch.einsum('don,dpn,dqn->dopqn', u0_x_grad_grad, A_partial_x[0:1,:,:], chi1_x).flatten(start_dim=1, end_dim=3)
    part8_int = torch.einsum('pn,qn,n->pq', A_partial_x[1,:,:], (chi1_y - chi1_y1_shift)[0,:,:], w2)
    alpha_part8 = (torch.einsum('o,p,q->opq', alpha_u0, alpha_A_partial_x, alpha_chi1) * part8_int).view(-1)

    # u0_xx < A_x eta2_y >
    part9 = torch.einsum('don,dpn,dqn->dopqn', u0_x_grad_grad, A_partial_x[0:1,:,:], eta2_x).flatten(start_dim=1, end_dim=3)
    part9_int = torch.einsum('pn,qn,n->pq', A_partial_x[1,:,:], (eta2_y_grad)[0,:,:], w2)
    alpha_part9 = (torch.einsum('o,p,q->opq', alpha_u0, alpha_A_partial_x, alpha_eta2) * part9_int).view(-1)

    # u0_xxx < A chi1 >
    part10 = torch.einsum('don,dpn,dqn->dopqn', u0_x_grad_grad_grad, A[0:1,:,:], chi1_x).flatten(start_dim=1, end_dim=3)
    part10_int = torch.einsum('pn,qn,n->pq', A[1,:,:], (chi1_y - chi1_y1_shift)[0,:,:], w2)
    alpha_part10 = (torch.einsum('o,p,q->opq', alpha_u0, alpha_A, alpha_chi1) * part10_int).view(-1)

    # u0_xxx < A eta2_y >
    part11 = torch.einsum('don,dpn,dqn->dopqn', u0_x_grad_grad_grad, A[0:1,:,:], eta2_x).flatten(start_dim=1, end_dim=3)
    part11_int = torch.einsum('pn,qn,n->pq', A[1,:,:], (eta2_y_grad)[0,:,:], w2)
    alpha_part11 = (torch.einsum('o,p,q->opq', alpha_u0, alpha_A, alpha_eta2) * part11_int).view(-1)
    
    hat_F = torch.cat((part1, part2, part3, part4, part5, part6, part7, part8, part9, part10, part11), dim=1)
    alpha_hat_F = torch.cat((alpha_part1, alpha_part2, alpha_part3, alpha_part4, alpha_part5, alpha_part6, alpha_part7, alpha_part8, alpha_part9, alpha_part10, alpha_part11))

    norm_square_F = Int2TNN(w1, alpha_hat_F, hat_F, alpha_hat_F, hat_F)

    hat_F_u0 = Int2TNN(w1, alpha_u0, u0_x, alpha_hat_F, hat_F)
    print('hat_F_u0', hat_F_u0)
    print('lam[0]', lam[0])

    
    u0_u0grad_chix = torch.einsum('don,dpn,dqn->dopqn', u0_x, u0_x_grad, chi1_x).flatten(start_dim=1, end_dim=3)
    int_chi1_y = torch.einsum('qn,n->q', (chi1_y - chi1_y1_shift).squeeze(dim=0), w2)
    alpha_u0_u0grad_chix = torch.einsum('o,p,q->opq', alpha_u0, alpha_u0, alpha_chi1 * int_chi1_y).view(-1)
    u0_u0grad_chix_int = Int1TNN(w1, alpha_u0_u0grad_chix, u0_u0grad_chix)
    lam0_u0_u0grad_chix_int = lam[0] * u0_u0grad_chix_int
    print('lam0_u0_u0grad_chix_int', lam0_u0_u0grad_chix_int)

    print(-(hat_F_u0+lam0_u0_u0grad_chix_int))

