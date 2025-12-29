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


# def bd_u0(x):
#     return torch.sin(pi * x)

# def grad_bd_u0(x):
#     return pi * torch.cos(pi * x)

# def grad_grad_bd_u0(x):
#     return -pi * pi * torch.sin(pi * x)

# def grad_grad_grad_bd_u0(x):
#     return -pi * pi * pi * torch.cos(pi * x)

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

def criterion_1(model_w):
    w_phi, w_grad_phi, w_grad_grad_phi = model_w(w1, x1, need_grad=2)

    # # 强形式
    # operand_A0_grad_x_w_grad_x = torch.einsum('dpn, dqn->dpqn', A0_grad_x, w_grad_phi)
    # operand_A0_w_grad_grad_x = torch.einsum('dpn, dqn->dpqn', A0, w_grad_grad_phi)

    # my_operand_phi = torch.cat((operand_A0_grad_x_w_grad_x, operand_A0_w_grad_grad_x), dim=1)
    # my_operand_alpha = torch.cat((alpha_A0_grad_x, alpha_A0))

    # # p * p matrix
    # my_matrix = torch.sum(
    #     torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(
    #         torch.einsum('n, dpqn, djkn->dpjqk', w1, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    # # p vector
    # my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_hat_F).unsqueeze(dim=-1) * torch.prod(
    #     torch.einsum('n, dpqn, dkn->dpkq', w1, my_operand_phi, hat_F), dim=0), dim=(0, 1))

    # my_alpha = torch.linalg.solve(my_matrix, my_vector)
    # model_w.scaling_par().data.copy_(my_alpha)
    # loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_square_F)

    # 弱形式
    my_matrix = torch.sum(alpha_A0 * torch.einsum('n, pn, qn, kn->pqk', w1, w_grad_phi.squeeze(dim=0), w_grad_phi.squeeze(dim=0), A0.squeeze(dim=0)), dim=-1)
    my_vector = torch.sum(alpha_hat_F * torch.einsum('n, pn, qn->pq', w1, w_phi.squeeze(dim=0), hat_F.squeeze(dim=0)), dim=-1)

    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_w.scaling_par().data.copy_(my_alpha)

    loss = -0.5 * torch.sum(my_alpha * my_vector)

    return loss


def train_1(model_w, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_1(model_w)
            if e == 0:
                print('\n Step 5 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                print('lr: ', scheduler.get_last_lr()[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_1(model_w)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])
                    torch.save(model_w, "./model_w/{}_final_w_x_Adam.pkl".format(name))

        

    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_1(model_w)
            optimizer.zero_grad()
            loss.backward()
            return loss

        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_1(model_w)
                    print('\n Step 1 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_1(model_w)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
    

if __name__ == '__main__':
    sizes = [1, 50, 50, 50, 50, p_w]
    activation = TNN_Sin

    # model_w = torch.load('./model_w/x_free_y_per_shift_final_w_x.pkl', map_location=device)
    model_w = TNN(1, sizes, activation, bd=bd_u0, grad_bd=grad_bd_u0, grad_grad_bd=grad_grad_bd_u0, \
        grad_grad_grad_bd=grad_grad_grad_bd_u0, scaling=True).to(dtype).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, itertools.chain(model_w.parameters())), lr=1e-2)

    epochs = 40000
    begin_time = time.time()
    train_1(model_w, optimizer, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_w, "./model_w/{}_final_w_x_Adam.pkl".format(name))

    
    epochs = 10000
    save = True 
    # optimizer used
    begin_time = time.time()
    train_1(model_w, optimizer, 'LBFGS', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_w, "./model_w/{}_final_w_x.pkl".format(name))

    
