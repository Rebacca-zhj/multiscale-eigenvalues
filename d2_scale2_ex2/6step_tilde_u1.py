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
data_dir = os.environ.get('DATA_DIR', '/home/zhanghuaijia')
import sys
import time
import math
import importlib.util

pi = 3.14159265358979323846
dtype = torch.double
device = 'cuda:3'
name = 'd2_ex2_test3'

dim=2
a, b = 0, 1
p_A = 1
p_chi = 20
p_tilde_u1 = 20



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
A_grad_x1 = A_dict['A_grad_x1']
alpha_A_grad_x1 = A_dict['alpha_A_grad_x1']
A_grad_x2 = A_dict['A_grad_x2']
alpha_A_grad_x2 = A_dict['alpha_A_grad_x2']
A_grad_y1 = A_dict['A_grad_y1']
alpha_A_grad_y1 = A_dict['alpha_A_grad_y1']
A_grad_y2 = A_dict['A_grad_y2']
alpha_A_grad_y2 = A_dict['alpha_A_grad_y2']

model_u0 = torch.load('./model_u0/d2_ex2_final_u0.pkl', map_location=device)
model_u1_x = torch.load('./model_u1/d2_ex2_final_u1_x.pkl', map_location=device)
model_u1_y = torch.load('./model_u1/d2_ex2_final_u1_y.pkl', map_location=device)
model_u2_x = torch.load('./model_u2/d2_ex2_final_u2_x.pkl', map_location=device)
model_u2_y = torch.load('./model_u2/d2_ex2_final_u2_y.pkl', map_location=device)
model_chi1_x = torch.load('./model_chi1/d2_ex2_final_chi1_x.pkl', map_location=device)
model_chi1_y = torch.load('./model_chi1/d2_ex2_final_chi1_y.pkl', map_location=device)
model_chi2_x = torch.load('./model_chi2/d2_ex2_final_chi2_x.pkl', map_location=device)
model_chi2_y = torch.load('./model_chi2/d2_ex2_final_chi2_y.pkl', map_location=device)

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


# 计算特征值，不需要梯度
with torch.no_grad():
    u1_x, u1_grad_x, u1_grad_grad_x = model_u1_x(w1, x1, need_grad=2)
    u1_y = model_u1_y(w3, x3, need_grad=0)
    alpha_u1 = model_u1_x.scaling_par()
    u1_norm_phi_y = model_u1_y(w3, x3, normed=False)
    u1_norm = torch.sqrt(torch.sum(w3 * u1_norm_phi_y**2, dim=-1)).unsqueeze(dim=-1)
    u1_y_shift = model_u1_y(dummy, x_error_0, need_grad=0, normed=False)
    u1_y_shift = u1_y_shift / u1_norm
    u1_x_grad_x1 = torch.cat((u1_grad_x[0:1,:,:], u1_x[1:2,:,:]), dim=0)
    u1_x_grad_x2 = torch.cat((u1_x[0:1,:,:], u1_grad_x[1:2,:,:]), dim=0)
    u1_x_grad_grad_x1 = torch.cat((u1_grad_grad_x[0:1,:,:], u1_x[1:2,:,:]), dim=0)
    u1_x_grad_grad_x2 = torch.cat((u1_x[0:1,:,:], u1_grad_grad_x[1:2,:,:]), dim=0)

    u2_x, u2_grad_x = model_u2_x(w1, x1, need_grad=1)
    u2_y, u2_grad_y = model_u2_y(w3, x3, need_grad=1)
    alpha_u2 = model_u2_x.scaling_par()
    u2_x_grad_x1 = torch.cat((u2_grad_x[0:1,:,:], u2_x[1:2,:,:]), dim=0)
    u2_x_grad_x2 = torch.cat((u2_x[0:1,:,:], u2_grad_x[1:2,:,:]), dim=0)
    u2_y_grad_y1 = torch.cat((u2_grad_y[0:1,:,:], u2_y[1:2,:,:]), dim=0)
    u2_y_grad_y2 = torch.cat((u2_y[0:1,:,:], u2_grad_y[1:2,:,:]), dim=0)

    part2_int = torch.prod(torch.einsum('dn,dpn,dqn->dpq', w34, A[2:4,:,:], u1_y), dim=0) - torch.prod(torch.einsum('dn,dpn,dqn->dpq', w34, A[2:4,:,:], u1_y_shift), dim=0)
    alpha_part2 = (torch.outer(alpha_A, alpha_u1) * part2_int).view(-1)
    part2 = torch.einsum('dpn,dqn->dpqn', A[0:2,:,:], u1_x_grad_grad_x1).flatten(start_dim=1, end_dim=2)
    part3 = torch.einsum('dpn,dqn->dpqn', A[0:2,:,:], u1_x_grad_grad_x2).flatten(start_dim=1, end_dim=2)
    part4 = torch.einsum('dpn,dqn->dpqn', A_grad_x1[0:2,:,:], u1_x_grad_x1).flatten(start_dim=1, end_dim=2)
    part5 = torch.einsum('dpn,dqn->dpqn', A_grad_x2[0:2,:,:], u1_x_grad_x2).flatten(start_dim=1, end_dim=2)

    part6_int = torch.prod(torch.einsum('dn,dpn,dqn->dpq', w34, A[2:4,:,:], u2_y_grad_y1), dim=0)
    alpha_part6 = (torch.outer(alpha_A, alpha_u2) * part6_int).view(-1)
    part6 = torch.einsum('dpn,dqn->dpqn', A[0:2,:,:], u2_x_grad_x1).flatten(start_dim=1, end_dim=2)
    part7 = torch.einsum('dpn,dqn->dpqn', A_grad_x1[0:2,:,:], u2_x).flatten(start_dim=1, end_dim=2)

    part8_int = torch.prod(torch.einsum('dn,dpn,dqn->dpq', w34, A[2:4,:,:], u2_y_grad_y2), dim=0)
    alpha_part8 = (torch.outer(alpha_A, alpha_u2) * part8_int).view(-1)
    part8 = torch.einsum('dpn,dqn->dpqn', A[0:2,:,:], u2_x_grad_x2).flatten(start_dim=1, end_dim=2)
    part9 = torch.einsum('dpn,dqn->dpqn', A_grad_x2[0:2,:,:], u2_x).flatten(start_dim=1, end_dim=2)

    rhs = torch.cat((part2, part3, part4, part5, part6, part7, part8, part9), dim=1)
    alpha_rhs = torch.cat((alpha_part2, alpha_part2, alpha_part2, alpha_part2, alpha_part6, alpha_part6, alpha_part8, alpha_part8))
    norm_square_rhs = Int2TNN(w12, alpha_rhs, rhs, alpha_rhs, rhs)
    
def criterion_tilde_u1(model_tilde_u1):
    tilde_u1_phi, tilde_u1_grad_phi, tilde_u1_grad_grad_phi = model_tilde_u1(w1, x1, need_grad=2)
    tilde_u1_x_grad_x1 = torch.cat((tilde_u1_grad_phi[0:1,:,:], tilde_u1_phi[1:2,:,:]), dim=0)
    tilde_u1_x_grad_x2 = torch.cat((tilde_u1_phi[0:1,:,:], tilde_u1_grad_phi[1:2,:,:]), dim=0)
    tilde_u1_x_grad2_x1 = torch.cat((tilde_u1_grad_grad_phi[0:1,:,:], tilde_u1_phi[1:2,:,:]), dim=0)
    tilde_u1_x_grad2_x2 = torch.cat((tilde_u1_phi[0:1,:,:], tilde_u1_grad_grad_phi[1:2,:,:]), dim=0)

    operand_A0_grad_x1_tilde_u1_grad_x1 = torch.einsum('dpn, dqn->dpqn', A0_11_grad_x1, tilde_u1_x_grad_x1)
    operand_A0_tilde_u1_grad2_x1 = torch.einsum('dpn, dqn->dpqn', A0_11, tilde_u1_x_grad2_x1)
    operand_A0_grad_x1_tilde_u1_grad_x2 = torch.einsum('dpn, dqn->dpqn', A0_12_grad_x1, tilde_u1_x_grad_x2)
    operand_A0_tilde_u1_grad_x1_x2 = torch.einsum('dpn, dqn->dpqn', A0_12, tilde_u1_grad_phi)
    operand_A0_grad_x2_tilde_u1_grad_x1 = torch.einsum('dpn, dqn->dpqn', A0_21_grad_x2, tilde_u1_x_grad_x1)
    operand_A0_tilde_u1_grad_x2_x1 = torch.einsum('dpn, dqn->dpqn', A0_21, tilde_u1_grad_phi)
    operand_A0_grad_x2_tilde_u1_grad_x2 = torch.einsum('dpn, dqn->dpqn', A0_22_grad_x2, tilde_u1_x_grad_x2)
    operand_A0_tilde_u1_grad2_x2 = torch.einsum('dpn, dqn->dpqn', A0_22, tilde_u1_x_grad2_x2)
 
    my_operand_phi = torch.cat((operand_A0_grad_x1_tilde_u1_grad_x1, operand_A0_tilde_u1_grad2_x1, operand_A0_grad_x1_tilde_u1_grad_x2,operand_A0_tilde_u1_grad_x1_x2,operand_A0_grad_x2_tilde_u1_grad_x1,operand_A0_tilde_u1_grad_x2_x1,operand_A0_grad_x2_tilde_u1_grad_x2,operand_A0_tilde_u1_grad2_x2), dim=1)
    my_operand_alpha = torch.cat((alpha_A0_11,alpha_A0_11,alpha_A0_12,alpha_A0_12,alpha_A0_21,alpha_A0_21,alpha_A0_22,alpha_A0_22))
    # p * p matrix
    my_matrix = torch.sum(
        torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(
            torch.einsum('n, dpqn, djkn->dpjqk', w1, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    # p vector
          
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_rhs).unsqueeze(dim=-1) * torch.prod(
        torch.einsum('n, dpqn, dkn->dpkq', w1, my_operand_phi, rhs), dim=0), dim=(0, 1))

    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_tilde_u1.scaling_par().data.copy_(my_alpha)

    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_square_rhs)
    
    return loss


def train_tilde_u1(model_tilde_u1, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_tilde_u1(model_tilde_u1)
            if e == 0:
                print('\n Step {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                print('lr: ', scheduler.get_last_lr()[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_tilde_u1(model_tilde_u1)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])
                    torch.save(model_tilde_u1, "./model_tilde_u1/{}_final_tilde_u1.pkl".format(name))

        
    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_tilde_u1(model_tilde_u1)
            optimizer.zero_grad()
            loss.backward()
            return loss

        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_tilde_u1(model_tilde_u1)
                    print('\n Step 1 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_tilde_u1(model_tilde_u1)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())

            
if __name__ == '__main__':
    # sizes_periodic = [1, 2, 50, 50, 50, p_tilde_u1] #test1 3.834463070164351e-05
    sizes_periodic = [1, 5, 80, 80, 80, p_tilde_u1] #test2 2.171476908614821e-06
    # sizes_periodic = [1, 8, 100, 100, 100, p_tilde_u1] # test3 7.499653456360247e-07
    print(sizes_periodic)
    activation = TNN_Sin

    if not os.path.exists("model_tilde_u1"):
        os.mkdir("model_tilde_u1")

    # model_tilde_u1 = torch.load('./model_tilde_u1/d2_ex2_test_final_tilde_u1.pkl',map_location=device)
    # loss = criterion_tilde_u1(model_tilde_u1)
    # print('loss',loss)

    model_tilde_u1 = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=True).to(dtype).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, itertools.chain(model_tilde_u1.parameters())),
        lr=1e-2)

    epochs = 40000
    begin_time = time.time()
    train_tilde_u1(model_tilde_u1, optimizer, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_tilde_u1, "./model_tilde_u1/{}_final_tilde_u1.pkl".format(name))

    # # INSERT_YOUR_CODE
    # # 画tilde_u1的图像
    # import numpy as np
    # from matplotlib import pyplot as plt


    # # INSERT_YOUR_CODE
    # # ============== 画tilde_u1关于x1, x2的三维图像 ==============

    # with torch.no_grad():
    #     # 均匀网格
    #     x1_plot = torch.linspace(a, b, 100, dtype=dtype, device=device)
    #     x2_plot = torch.linspace(a, b, 100, dtype=dtype, device=device)
    #     X1, X2 = torch.meshgrid(x1_plot, x2_plot, indexing='ij')
    #     X1_flat = X1.flatten()  # [n_points]
    #     X2_flat = X2.flatten()  # [n_points]

    #     # 可能的周期微元坐标
    #     # y1_flat = X1_flat * eps - (X1_flat * eps).floor()
    #     # y2_flat = X2_flat * eps - (X2_flat * eps).floor()
    #     xy_grid = torch.stack([X1_flat, X2_flat], dim=0)  # [2, n_points]
    #     n_points = xy_grid.shape[1]
    #     dummy_plot = torch.ones((1, n_points), dtype=dtype, device=device)

    #     # tilde_u1的归一化
    #     tilde_u1_norm_phi = model_tilde_u1(w1, x1, need_grad=0, normed=False)   # [d, p, N]
    #     tilde_u1_norm = torch.sqrt(torch.sum(w1 * tilde_u1_norm_phi ** 2, dim=-1)).unsqueeze(dim=-1)  # [d, p, 1]

    #     # tilde_u1 basis on full grid
    #     tilde_u1_phi_plot = model_tilde_u1(dummy_plot, xy_grid, need_grad=0, normed=False)  # [d, p, n_points]
    #     tilde_u1_phi_plot = tilde_u1_phi_plot / tilde_u1_norm

    #     # 获取alpha
    #     alpha_tilde_u1 = model_tilde_u1.scaling_par()  # [p]

    #     # tilde_u1 = sum alpha * prod(phi) (这里只考虑d=2)
    #     tilde_u1_plot = torch.sum(alpha_tilde_u1.unsqueeze(dim=-1) *
    #                              torch.prod(tilde_u1_phi_plot, dim=0),
    #                              dim=0)  # [n_points]

    #     # reshape到二维
    #     tilde_u1_plot_2d = tilde_u1_plot.reshape(100, 100)
    #     tilde_u1_value_2d = tilde_u1_plot_2d.detach().cpu().numpy()
    #     X1_np = X1.detach().cpu().numpy()
    #     X2_np = X2.detach().cpu().numpy()

    #     # 绘制三维图
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     surf = ax.plot_surface(X1_np, X2_np, tilde_u1_value_2d, cmap='viridis',
    #                            edgecolor='none', alpha=0.9, linewidth=0, antialiased=True)
    #     ax.set_xlabel('x1', fontsize=12)
    #     ax.set_ylabel('x2', fontsize=12)
    #     ax.set_zlabel('tilde_u1(x1, x2)', fontsize=12)
    #     ax.set_title(r'$\tilde{u}_1(x_1, x_2)$ 3D Plot', fontsize=14)
    #     fig.colorbar(surf, shrink=0.5, aspect=20, pad=0.1, label='tilde_u1')
    #     ax.view_init(elev=30, azim=45)
    #     plt.tight_layout()
    #     plt.savefig('png/tilde_u1_x1_x2_surface.png', dpi=300, bbox_inches='tight')
    #     plt.close(fig)
    #     print("3D plot saved to: png/tilde_u1_x1_x2_surface.png")


    # # INSERT_YOUR_CODE
    # # ========== 画tilde_u1关于x1, x2的三维图像 ==========
    # import matplotlib.pyplot as plt

    # # 设尺度参数
    # eps = 10.0

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

    #     # tilde_u1的归一化
    #     tilde_u1_norm_phi = model_tilde_u1(w1, x1, need_grad=0, normed=False)   # [d, p, N]
    #     tilde_u1_norm = torch.sqrt(torch.sum(w1 * tilde_u1_norm_phi ** 2, dim=-1)).unsqueeze(dim=-1)  # [d, p, 1]

    #     # tilde_u1 basis on the grid
    #     tilde_u1_phi_plot = model_tilde_u1(dummy_plot, xy_grid, need_grad=0, normed=False)  # [d, p, n_points]
    #     tilde_u1_phi_plot = tilde_u1_phi_plot / tilde_u1_norm

    #     # 获取alpha
    #     alpha_tilde_u1 = model_tilde_u1.scaling_par()  # [p]

    #     # tilde_u1 = sum alpha * prod(phi) (这里只考虑d=2)
    #     tilde_u1_plot = torch.sum(
    #         alpha_tilde_u1.unsqueeze(dim=-1) *
    #         torch.prod(tilde_u1_phi_plot, dim=0),
    #         dim=0
    #     )  # [n_points]

    #     # 变为2D形状
    #     tilde_u1_plot_2d = tilde_u1_plot.reshape(100, 100)
    #     tilde_u1_value_2d = tilde_u1_plot_2d.detach().cpu().numpy()
    #     X1_np = X1.detach().cpu().numpy()
    #     X2_np = X2.detach().cpu().numpy()

    #     # 绘制 tilde_u1(x1, x2) 的二维热力图
    #     fig = plt.figure(figsize=(8, 6))
    #     im = plt.imshow(
    #         tilde_u1_value_2d,
    #         extent=(a, b, a, b),
    #         origin='lower',
    #         cmap='viridis',
    #         aspect='auto'
    #     )
    #     plt.xlabel('x1', fontsize=12)
    #     plt.ylabel('x2', fontsize=12)
    #     plt.title(r'$\tilde u_1(x_1, x_2)$ 2D Heatmap ($\epsilon={}$)'.format(eps), fontsize=14)
    #     cbar = plt.colorbar(im, shrink=0.8, pad=0.03)
    #     cbar.set_label('tilde_u1', fontsize=12)
    #     plt.tight_layout()
    #     plt.savefig('png/tilde_u1_x1_x2_heatmap.png', dpi=300, bbox_inches='tight')
    #     plt.close(fig)
    #     print("2D heatmap saved to: png/tilde_u1_x1_x2_heatmap.png")

    #     # 绘制三维图
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     surf = ax.plot_surface(X1_np, X2_np, tilde_u1_value_2d, cmap='viridis',
    #                            edgecolor='none', alpha=0.9, linewidth=0, antialiased=True)
    #     ax.set_xlabel('x1', fontsize=12)
    #     ax.set_ylabel('x2', fontsize=12)
    #     ax.set_zlabel(r'$\tilde{u}_1(x_1, x_2)$', fontsize=12)
    #     ax.set_title(r'$\tilde{u}_1(x_1, x_2)$ 3D Plot', fontsize=14)
    #     fig.colorbar(surf, shrink=0.5, aspect=20, pad=0.1, label='tilde_u1')
    #     ax.view_init(elev=30, azim=45)
    #     plt.tight_layout()
    #     plt.savefig('png/tilde_u1_x1_x2_surface.png', dpi=300, bbox_inches='tight')
    #     plt.close(fig)
    #     print("3D plot saved to: png/tilde_u1_x1_x2_surface.png")