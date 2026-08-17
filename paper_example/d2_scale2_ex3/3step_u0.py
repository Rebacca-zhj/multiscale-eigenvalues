import torch
torch.set_printoptions(profile="full", precision=16)
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import torch.optim as optim
import importlib.util

from tnn import *
from integration import *
from quadrature import *

import itertools
import os
import sys
import time

pi = 3.14159265358979323846
dtype = torch.double
device = 'cuda:6'
name = 'd2_ex3_test2'

dim = 2

p_A = 1
p_chi = 20
p = 20
p_A0 = p_A + p_A * p_chi

a, b = 0, 1

x1, w1 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
N = len(x1)
x2, w2 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
x3, w3 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
x4, w4 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)

w12 = torch.stack((w1, w2), dim=0)
w34 = torch.stack((w3, w4), dim=0)
w1234 = torch.stack((w1, w2, w3, w4), dim=0)

dummy = torch.ones((1, 1), dtype=dtype, device=device)
x_error_0 = torch.zeros(1, dtype=dtype, device=device)
F = torch.ones((2, 2, N), dtype=dtype, device=device)
F[0, 0, :] = torch.sin(2 * pi * x1)
F[1, 1, :] = torch.cos(2 * pi * x2)
alpha_F = torch.tensor([30, 20], dtype=dtype, device=device)
norm_F = Int2TNN(w12, alpha_F, F, alpha_F, F, if_sum=True)

# 从0step_A.py导入A矩阵信息（使用importlib因为模块名不能以数字开头）
spec = importlib.util.spec_from_file_location("step_0_A", os.path.join(os.path.dirname(__file__), "0step_A.py"))
if spec is None or spec.loader is None:
    raise ImportError("Could not load 0step_A.py for module import.")
step_0_A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step_0_A)
init_A_matrices = step_0_A.init_A_matrices

A_dict = init_A_matrices(x1, x2, x3, x4, pi=pi, dtype=dtype, device=device)
A = A_dict['A']
alpha_A = A_dict['alpha_A']
A_grad = A_dict['A_grad']

model_chi1_x = torch.load('./model_chi1/d2_ex3_final_chi1_x.pkl', map_location=device)
model_chi1_y = torch.load('./model_chi1/d2_ex3_final_chi1_y.pkl', map_location=device)

model_chi2_x = torch.load('./model_chi2/d2_ex3_final_chi2_x.pkl', map_location=device)
model_chi2_y = torch.load('./model_chi2/d2_ex3_final_chi2_y.pkl', map_location=device)

with torch.no_grad():
    chi1_phi_x, chi1_grad_phi_x = model_chi1_x(w1, x1, need_grad=1)
    chi1_phi_y, chi1_grad_phi_y = model_chi1_y(w3, x3, need_grad=1)
    alpha_chi1 = model_chi1_x.scaling_par()

    chi2_phi_x, chi2_grad_phi_x = model_chi2_x(w1, x1, need_grad=1)
    chi2_phi_y, chi2_grad_phi_y = model_chi2_y(w3, x3, need_grad=1)
    alpha_chi2 = model_chi2_x.scaling_par()

    # a_{ij}^{*}=\int_{Y}\left(a \delta_{ij} + a \frac{\grad\chi_j}{\grad y_i}\right) dy
    A0_11 = torch.ones((2, p_A + p_A * p_chi, N), dtype=dtype, device=device)
    A0_11[0:2, 0:p_A, :] = A[0:2, :, :]
    A0_11[0:2, p_A:p_A + p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[0:2, :, :], chi1_phi_x).flatten(start_dim=1, end_dim=2)

    A0_11_grad_x1 = torch.ones((2, p_A + p_A * p_chi, N), dtype=dtype, device=device)
    A0_11_grad_x1[0:1, 0:p_A, :] = A_grad[0:1, :, :]
    A0_11_grad_x1[1:2, 0:p_A, :] = A[1:2, :, :]
    A0_11_grad_x1[0:1, p_A:p_A + p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A_grad[0:1, :, :], chi1_phi_x[0:1, :, :]).flatten(start_dim=1, end_dim=2) \
                                                        + torch.einsum('dpn, dqn->dpqn', A[0:1, :, :], chi1_grad_phi_x[0:1, :, :]).flatten(start_dim=1, end_dim=2)
    A0_11_grad_x1[1:2, p_A:p_A + p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[1:2, :, :], chi1_phi_x[1:2, :, :]).flatten(start_dim=1, end_dim=2)

    alpha_A0_11 = torch.ones(p_A + p_A * p_chi, dtype=dtype, device=device)
    alpha_A0_11[0:p_A] = torch.prod(torch.einsum('dn, dpn->dp', w34, A[2:4]), dim=0) * alpha_A
    alpha_A0_11[p_A:p_A + p_A * p_chi] = (torch.einsum('n, pn, qn->pq', w3, A[2, :, :], chi1_grad_phi_y[0, :, :]) * torch.einsum('n, pn, qn->pq', w4, A[3, :, :], chi1_phi_y[1, :, :]) * torch.outer(alpha_A, alpha_chi1)).view(-1)
    
    A0_12 = torch.ones((2, p_A * p_chi, N), dtype=dtype, device=device)
    A0_12[0:2, 0:p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[0:2, :, :], chi2_phi_x).flatten(start_dim=1, end_dim=2)

    A0_12_grad_x1 = torch.ones((2, p_A * p_chi, N), dtype=dtype, device=device)
    A0_12_grad_x1[0:1, 0:p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A_grad[0:1, :, :], chi2_phi_x[0:1, :, :]).flatten(start_dim=1, end_dim=2) \
                                                 + torch.einsum('dpn, dqn->dpqn', A[0:1, :, :], chi2_grad_phi_x[0:1, :, :]).flatten(start_dim=1, end_dim=2)
    A0_12_grad_x1[1:2, 0:p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[1:2, :, :], chi2_phi_x[1:2, :, :]).flatten(start_dim=1, end_dim=2)

    alpha_A0_12 = torch.ones(p_A * p_chi, dtype=dtype, device=device)
    alpha_A0_12[0:p_A * p_chi] = (torch.einsum('n, pn, qn->pq', w3, A[2, :, :], chi2_grad_phi_y[0, :, :]) * torch.einsum('n, pn, qn->pq', w4, A[3, :, :], chi2_phi_y[1, :, :]) * torch.outer(alpha_A, alpha_chi2)).view(-1)

    A0_21 = torch.ones((2, p_A * p_chi, N), dtype=dtype, device=device)
    A0_21[0:2, 0:p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[0:2, :, :], chi1_phi_x).flatten(start_dim=1, end_dim=2)

    A0_21_grad_x2 = torch.ones((2, p_A * p_chi, N), dtype=dtype, device=device)
    A0_21_grad_x2[0:1, 0:p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[0:1, :, :], chi1_phi_x[0:1, :, :]).flatten(start_dim=1, end_dim=2)
    A0_21_grad_x2[1:2, 0:p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A_grad[1:2, :, :], chi1_phi_x)[1:2, :, :].flatten(start_dim=1, end_dim=2) \
                                                + torch.einsum('dpn, dqn->dpqn', A[1:2, :, :], chi1_grad_phi_x)[1:2, :, :].flatten(start_dim=1, end_dim=2)

    alpha_A0_21 = torch.ones(p_A * p_chi, dtype=dtype, device=device)
    alpha_A0_21[0:p_A * p_chi] = (torch.einsum('n, pn, qn->pq', w3, A[2, :, :], chi1_phi_y[0, :, :]) * torch.einsum('n, pn, qn->pq', w4, A[3, :, :], chi1_grad_phi_y[1, :, :]) * torch.outer(alpha_A, alpha_chi1)).view(-1)

    A0_22 = torch.ones((2, p_A + p_A * p_chi, N), dtype=dtype, device=device)
    A0_22[0:2, 0:p_A, :] = A[0:2, :, :]
    A0_22[0:2, p_A:p_A + p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[0:2, :, :], chi2_phi_x).flatten(start_dim=1, end_dim=2)

    A0_22_grad_x2 = torch.ones((2, p_A + p_A * p_chi, N), dtype=dtype, device=device)
    A0_22_grad_x2[0:1, 0:p_A, :] = A[0:1, :, :]
    A0_22_grad_x2[1:2, 0:p_A, :] = A_grad[1:2, :, :]
    A0_22_grad_x2[0:1, p_A:p_A + p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[0:1, :, :], chi2_phi_x[0:1, :, :]).flatten(start_dim=1, end_dim=2)
    A0_22_grad_x2[1:2, p_A:p_A + p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A_grad[1:2, :, :], chi2_phi_x[1:2, :, :]).flatten(start_dim=1, end_dim=2)\
                                                         + torch.einsum('dpn, dqn->dpqn', A[1:2, :, :], chi2_grad_phi_x[1:2, :, :]).flatten(start_dim=1, end_dim=2)

    alpha_A0_22 = torch.ones(p_A + p_A * p_chi, dtype=dtype, device=device)
    alpha_A0_22[0:p_A] = torch.prod(torch.einsum('dn, dpn->dp', w34, A[2:4]), dim=0) * alpha_A
    alpha_A0_22[p_A:p_A + p_A * p_chi] = (torch.einsum('n, pn, qn->pq', w3, A[2, :, :], chi2_phi_y[0, :, :]) * torch.einsum('n, pn, qn->pq', w4, A[3, :, :], chi2_grad_phi_y[1, :, :]) * torch.outer(alpha_A, alpha_chi2)).view(-1)

# 弱形式
def criterion_u0_weak(model_u0):
    u0_phi, u0_grad_phi, u0_grad_grad_phi = model_u0(w1, x1, need_grad=2)
    # alpha = model_u0.scaling_par()

    my_matrix = torch.sum(alpha_A0_11 * torch.einsum('n, pn, qn, kn->pqk', w1, u0_grad_phi[0], u0_grad_phi[0], A0_11[0]) * torch.einsum('n, pn, qn, kn->pqk', w2, u0_phi[1], u0_phi[1], A0_11[1]), dim=-1) + \
        torch.sum(alpha_A0_12 * torch.einsum('n, pn, qn, kn->pqk', w1, u0_phi[0], u0_grad_phi[0], A0_12[0]) * torch.einsum('n, pn, qn, kn->pqk', w2, u0_grad_phi[1], u0_phi[1], A0_12[1]), dim=-1) + \
        torch.sum(alpha_A0_21 * torch.einsum('n, pn, qn, kn->pqk', w1, u0_grad_phi[0], u0_phi[0], A0_21[0]) * torch.einsum('n, pn, qn, kn->pqk', w2, u0_phi[1], u0_grad_phi[1], A0_21[1]), dim=-1) + \
        torch.sum(alpha_A0_22 * torch.einsum('n, pn, qn, kn->pqk', w1, u0_phi[0], u0_phi[0], A0_22[0]) * torch.einsum('n, pn, qn, kn->pqk', w2, u0_grad_phi[1], u0_grad_phi[1], A0_22[1]), dim=-1)
    
    # print(torch.norm(my_matrix - my_matrix.t()))

    my_vector = torch.sum(alpha_F * torch.prod(torch.einsum('dn, dpn, dqn->dpq', w12, u0_phi, F), dim=0), dim=-1)

    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_u0.scaling_par().data.copy_(my_alpha)

    # 弱形式
    loss = -0.5 * torch.sum(my_alpha * my_vector)

    return loss

# 强形式
def criterion_u0(model_u0):
    u0_phi, u0_grad_phi, u0_grad_grad_phi = model_u0(w1, x1, need_grad=2)
    
    # 强形式
    u_grad_x1 = torch.cat((u0_grad_phi[0:1, :, :], u0_phi[1:2, :, :]),dim=0)
    u_grad_x2 = torch.cat((u0_phi[0:1, :, :], u0_grad_phi[1:2, :, :]),dim=0)
    u_grad2_x1 = torch.cat((u0_grad_grad_phi[0:1, :, :], u0_phi[1:2, :, :]),dim=0)
    u_grad2_x2 = torch.cat((u0_phi[0:1, :, :], u0_grad_grad_phi[1:2, :, :]),dim=0)

    operand_A0_11_grad_x1_u_grad_x1 = torch.einsum('dpn,dqn->dpqn', A0_11_grad_x1, u_grad_x1)
    operand_A0_11_u_grad2_x1 = torch.einsum('dpn,dqn->dpqn', A0_11, u_grad2_x1)
    operand_A0_12_grad_x1_u_grad_x2 = torch.einsum('dpn,dqn->dpqn', A0_12_grad_x1, u_grad_x2)
    operand_A0_12_u_grad2_x1x2 = torch.einsum('dpn,dqn->dpqn', A0_12, u0_grad_phi)
    operand_A0_21_grad_x2_u_grad_x1 = torch.einsum('dpn,dqn->dpqn', A0_21_grad_x2, u_grad_x1)
    operand_A0_21_u_grad2_x1x2 = torch.einsum('dpn,dqn->dpqn', A0_21, u0_grad_phi)
    operand_A0_22_grad_x2_u_grad_x2 = torch.einsum('dpn,dqn->dpqn', A0_22_grad_x2, u_grad_x2)
    operand_A0_22_u_grad2_x2 = torch.einsum('dpn,dqn->dpqn', A0_22, u_grad2_x2)

    my_operand_phi = torch.cat((operand_A0_11_grad_x1_u_grad_x1,operand_A0_11_u_grad2_x1,
                                        operand_A0_12_grad_x1_u_grad_x2,operand_A0_12_u_grad2_x1x2,
                                        operand_A0_21_grad_x2_u_grad_x1,operand_A0_21_u_grad2_x1x2,
                                        operand_A0_22_grad_x2_u_grad_x2,operand_A0_22_u_grad2_x2), dim=1)
    my_operand_alpha = torch.cat((alpha_A0_11, alpha_A0_11, alpha_A0_12, alpha_A0_12, alpha_A0_21, alpha_A0_21, alpha_A0_22, alpha_A0_22), dim=0)
    my_matrix = torch.sum(
        torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(
            torch.einsum('dn, dpqn, djkn->dpjqk', w12, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    
    # 计算向量：my_vector = ∫ (my_operand_phi, rhs)
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_F).unsqueeze(dim=-1) * torch.prod(
        torch.einsum('dn, dpqn, dkn->dpkq', w12, my_operand_phi, F), dim=0), dim=(0, 1))
    
    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    
    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_F)
    
    model_u0.scaling_par().data.copy_(my_alpha.detach())

    return loss


def train_0(model_u0, optimizer, criterion, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion(model_u0)
            if e == 0:
                print('\n Step 3 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                print('lr: ', scheduler.get_last_lr()[0])
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion(model_u0)
                    print('\n [ epoch %d ]' % (e+1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])
                    torch.save(model_u0, "model_u0/{}_final_u0.pkl".format(name))

        # if not os.path.exists("model_u0"):
        #     os.mkdir("model_u0")
        # torch.save(model_u0, "model_u0/{}_Adam_u0.pkl".format(name))
                
    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion(model_u0)
            optimizer.zero_grad()
            loss.backward()
            return loss
        
        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion(model_u0)
                    print('\n Step 3 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion(model_u0)
                    print('\n [ epoch %d ]' % (e+1))
                    print('loss: ', loss.item())

            # if (e + 1) == 1000:
            #     torch.save(model_u0, "model_u0/{}_LBFGS_1000_u0.pkl".format(name))
        
        # torch.save(model_u0, "model_u0/{}_final_u0.pkl".format(name))
 
if __name__ == '__main__':
    sizes_periodic = [1, 5, 100, 100, 100, 100, p]
    activation = TNN_Sin
    print(sizes_periodic)

    if not os.path.exists("model_u0"):
        os.mkdir("model_u0")

    # model_u0 = torch.load("model_u0/d2_ex3_test_final_u0.pkl", map_location=device)
    # loss = criterion_u0(model_u0)
    # print('loss',loss)
    
    # # model_u0 = TNN(dim, sizes, activation, scaling=True).to(dtype).to(device)
    model_u0 = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=True).to(dtype).to(device)

    # 弱形式学习
    optimizer = optim.Adam(filter(lambda p:p.requires_grad, itertools.chain(model_u0.parameters())), lr=1e-2)
    epochs = 20000
    begin_time = time.time()
    train_0(model_u0, optimizer, criterion_u0_weak, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))
    torch.save(model_u0, "model_u0/{}_weak_final_u0.pkl".format(name))

    # 强形式学习
    optimizer = optim.Adam(filter(lambda p:p.requires_grad, itertools.chain(model_u0.parameters())), lr=1e-2)
    epochs = 50000

    begin_time = time.time()
    train_0(model_u0, optimizer, criterion_u0, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_u0, "model_u0/{}_final_u0.pkl".format(name))

    # 第二次训练 LBFGS
    epochs = 10000
    optimizer = torch.optim.LBFGS(
        filter(lambda p: p.requires_grad, itertools.chain(model_u0.parameters())),
        lr=0.1)
    train_0(model_u0, optimizer, criterion_u0, 'LBFGS', epochs)
    torch.save(model_u0, "model_u0/{}_final_u0.pkl".format(name))
