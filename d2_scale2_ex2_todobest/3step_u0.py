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
device = 'cuda:2'
name = 'd2_ex2_test4'

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
F = torch.ones((2, 1, N), dtype=dtype, device=device)
F[0, 0, :] = torch.sin(2 * pi * x1)
F[1, 0, :] = torch.cos(2 * pi * x2)
alpha_F = 30*torch.ones(1, dtype=dtype, device=device)
norm_F = torch.sqrt(Int2TNN(w12, alpha_F, F, alpha_F, F, if_sum=True))

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

model_chi1_x = torch.load('./model_chi1/d2_ex2_final_chi1_x.pkl', map_location=device)
model_chi1_y = torch.load('./model_chi1/d2_ex2_final_chi1_y.pkl', map_location=device)

model_chi2_x = torch.load('./model_chi2/d2_ex2_final_chi2_x.pkl', map_location=device)
model_chi2_y = torch.load('./model_chi2/d2_ex2_final_chi2_y.pkl', map_location=device)

def einsum_flatten(a, b):
    return torch.einsum('dpn,dqn->dpqn', a, b).reshape(a.shape[0], -1, a.shape[2])

def batch_outer(alpha, my_alpha):
    return torch.outer(alpha, my_alpha).view(-1)

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


def criterion_0(model_u0):
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

    # # 弱形式
    # loss = -0.5 * torch.sum(my_alpha * my_vector)

    # 后验误差loss
    u_grad_x1 = torch.cat((u0_grad_phi[0:1, :, :], u0_phi[1:2, :, :]),dim=0)
    u_grad_x2 = torch.cat((u0_phi[0:1, :, :], u0_grad_phi[1:2, :, :]),dim=0)
    u_grad2_x1 = torch.cat((u0_grad_grad_phi[0:1, :, :], u0_phi[1:2, :, :]),dim=0)
    u_grad2_x2 = torch.cat((u0_phi[0:1, :, :], u0_grad_grad_phi[1:2, :, :]),dim=0)
    u_grad_x1_grad_x2 = torch.cat((u0_grad_phi[0:1, :, :], u0_grad_phi[1:2, :, :]),dim=0)

    operand_phi_list = [
        einsum_flatten(A0_11_grad_x1, u_grad_x1),
        einsum_flatten(A0_11, u_grad2_x1),
        einsum_flatten(A0_12_grad_x1, u_grad_x2),
        einsum_flatten(A0_12, u_grad_x1_grad_x2),
        einsum_flatten(A0_21_grad_x2, u_grad_x1),
        einsum_flatten(A0_21, u_grad_x1_grad_x2),
        einsum_flatten(A0_22_grad_x2, u_grad_x2),
        einsum_flatten(A0_22, u_grad2_x2),
        F]

    my_operand_phi = torch.cat(operand_phi_list, dim=1)
    
    operand_alpha_list = [
            batch_outer(alpha_A0_11, my_alpha),  # alpha_A0_11_grad_x1_u_grad_x1
            batch_outer(alpha_A0_11, my_alpha),  # alpha_A0_11_u_grad2_x1 (与上一个相同)
            batch_outer(alpha_A0_12, my_alpha),  # alpha_A0_12_grad_x1_u_grad_x2
            batch_outer(alpha_A0_12, my_alpha),  # alpha_A0_12_u_grad_x1_grad_x2 (与上一个相同)
            batch_outer(alpha_A0_21, my_alpha),  # alpha_A0_21_grad_x2_u_grad_x1
            batch_outer(alpha_A0_21, my_alpha),  # alpha_A0_21_u_grad_x1_grad_x2 (与上一个相同)
            batch_outer(alpha_A0_22, my_alpha),  # alpha_A0_22_grad_x2_u_grad_x2
            batch_outer(alpha_A0_22, my_alpha),  # alpha_A0_22_u_grad2_x2 (与上一个相同)
            alpha_F]
    
    my_operand_alpha = torch.cat(operand_alpha_list, dim=0).T

    # 计算批量结果并检查是否有0
    loss = torch.sqrt(torch.abs(Int2TNN(w1, my_operand_alpha, my_operand_phi, my_operand_alpha, my_operand_phi, if_sum=True)))
    
    return loss

def criterion_1(model_u0):
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
    
    loss = torch.sqrt(torch.abs(torch.sum(-my_alpha * my_vector) + norm_F))
    
    model_u0.scaling_par().data.copy_(my_alpha.detach())

    return loss


def train_0(model_u0, optimizer,criterion, optimizer_flag, epochs):
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
                    loss1 = criterion_0(model_u0)
                    print('loss_1: ', loss1.item())

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
if __name__ == '__main__':
    # sizes_periodic = [1, 2, 20, 20, 20, p] # test1 0.00039871638656340224
    sizes_periodic = [1, 2, 50, 50, 50, p] # test2 8.67760111908854e-05
    # sizes_periodic = [1, 5, 80, 80, 80, p] # test3 no
    sizes_periodic = [1, 2, 50, 50, p] # test4 0.0001719728597632713
    activation = TNN_Sin
    print(sizes_periodic)

    if not os.path.exists("model_u0"):
        os.mkdir("model_u0")

    model_u0 = torch.load("model_u0/d2_ex2_test2_final_u0.pkl", map_location=device)
    loss = criterion_0(model_u0)
    print('loss',loss)
    
    # model_u0 = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=True).to(dtype).to(device)

    # # optimizer = optim.Adam(filter(lambda p:p.requires_grad, itertools.chain(model_u0.parameters())), lr=1e-2)

    # # epochs = 20000

    # # begin_time = time.time()
    # # train_0(model_u0, optimizer,criterion_0, 'Adam', epochs)
    # # end_time = time.time()
    # # print('\n')
    # # print('*' * 40)
    # # print('Training done!')
    # # print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    # # torch.save(model_u0, "model_u0/{}_final_u0.pkl".format(name))

    # optimizer = optim.Adam(filter(lambda p:p.requires_grad, itertools.chain(model_u0.parameters())), lr=1e-2)

    # epochs = 50000

    # begin_time = time.time()
    # train_0(model_u0, optimizer,criterion_0, 'Adam', epochs)
    # end_time = time.time()
    # print('\n')
    # print('*' * 40)
    # print('Training done!')
    # print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    # torch.save(model_u0, "model_u0/{}_final_u0.pkl".format(name))

    # # 第二次训练 LBFGS
    # epochs = 5000
    # optimizer = torch.optim.LBFGS(
    #     filter(lambda p: p.requires_grad, itertools.chain(model_u0.parameters())),
    #     lr=0.1)
    # train_0(model_u0, optimizer, criterion_0, 'LBFGS', epochs)
    # torch.save(model_u0, "model_u0/{}_final_u0.pkl".format(name))

    # # ========== 画chi2关于x1, x2的三维图像 ==========
    # import matplotlib.pyplot as plt

    # with torch.no_grad():
    #     # 均匀网格
    #     x1_plot = torch.linspace(a, b, 100, dtype=dtype, device=device)
    #     x2_plot = torch.linspace(a, b, 100, dtype=dtype, device=device)
    #     X1, X2 = torch.meshgrid(x1_plot, x2_plot, indexing='ij')
    #     X1_flat = X1.flatten()
    #     X2_flat = X2.flatten()

    #     xy_grid = torch.stack([X1_flat, X2_flat], dim=0)  # [2, n_points]
    #     n_points = xy_grid.shape[1]
    #     dummy_plot = torch.ones((1, n_points), dtype=dtype, device=device)

    #     #归一化
    #     u0_x_norm_phi = model_u0(w1, x1, need_grad=0, normed=False)
    #     u0_x_norm = torch.sqrt(torch.sum(w1 * u0_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    #     # u0_phi_x (大尺度), u0_phi_y (小尺度)
    #     u0_phi_x_plot = model_u0(dummy_plot, xy_grid, need_grad=0, normed=False)  # [d, p, n_points]
    #     u0_phi_x_plot = u0_phi_x_plot / u0_x_norm


    #     # 获取alpha
    #     alpha_u0 = model_u0.scaling_par()  # [p]

    #     # u0 = sum(alpha * prod(u0_phi_x))
    #     u0_plot = torch.sum(
    #         alpha_u0.unsqueeze(dim=-1) *
    #         torch.prod(u0_phi_x_plot, dim=0),
    #         dim=0
    #     )  # [n_points]
    #     # 转成二维形状
    #     u0_plot_2d = u0_plot.reshape(100, 100)
    #     u0_value_2d = u0_plot_2d.detach().cpu().numpy()
    #     X1_np = X1.detach().cpu().numpy()
    #     X2_np = X2.detach().cpu().numpy()

    #     # 绘制二维图像（等高线图）
    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     contour = ax.contourf(X1_np, X2_np, u0_value_2d, levels=100, cmap='viridis')
    #     cbar = fig.colorbar(contour, shrink=0.75, aspect=20, pad=0.02)
    #     cbar.set_label('u0', fontsize=12)
    #     ax.set_xlabel('x1', fontsize=12)
    #     ax.set_ylabel('x2', fontsize=12)
    #     ax.set_title(r'$u_0(x_1, x_2)$ Contour Plot', fontsize=14)
    #     plt.tight_layout()
    #     plt.savefig('png/u0_x1_x2_contour.png', dpi=300, bbox_inches='tight')
    #     plt.close(fig)
    #     print("2D contour plot saved to: png/u0_x1_x2_contour.png")
 


