import torch
import math
import numpy as np

torch.set_printoptions(profile="full", precision=16)
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import torch.optim as optim
import gc
from tnn import *
from integration import *
from quadrature import *

import itertools
import os
import sys
import time
import importlib.util

pi = 3.14159265358979323846
dtype = torch.double
device = 'cuda:2'
name = 'd2_ex4'

p_chi = 20
p_tilde_u1 = 20
p_delta_u2 = 20

dim = 2
a, b = 0, 1

x1, w1 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
N = len(x1)
x2, w2 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
x3, w3 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
x4, w4 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)

w1234 = torch.stack((w1, w2, w3, w4), dim=0)
w12 = torch.stack((w1, w2), dim=0)
w34 = torch.stack((w3, w4), dim=0)

dummy = torch.ones((1, 1), dtype=dtype, device=device)
x_error_0 = torch.zeros(1, dtype=dtype, device=device)

# 从0step_A.py导入A矩阵信息（使用importlib因为模块名不能以数字开头）
spec = importlib.util.spec_from_file_location("step_0_A", os.path.join(os.path.dirname(__file__), "0step_A.py"))
if spec is not None and spec.loader is not None:
    step_0_A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step_0_A)
else:
    raise ImportError("Failed to import 0step_A.py: spec or spec.loader is None")
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

model_tilde_u1 = torch.load('./model_tilde_u1/d2_ex4_final_tilde_u1.pkl', map_location=device)


with torch.no_grad():
    tilde_u1, tilde_u1_grad = model_tilde_u1(w1, x1, need_grad=1)
    alpha_tilde_u1 = model_tilde_u1.scaling_par()

    expand_tilde_u1_grad_x1 = torch.cat((tilde_u1_grad[0:1, :, :], tilde_u1[1:2, :, :], torch.ones((2, p_tilde_u1, N), dtype=dtype, device=device)), dim=0)
    expand_tilde_u1_grad_x2 = torch.cat((tilde_u1[0:1, :, :], tilde_u1_grad[1:2, :, :], torch.ones((2, p_tilde_u1, N), dtype=dtype, device=device)), dim=0)

    operand_A_grad_y1_tilde_u1_grad_x1 = torch.einsum('dpn, dqn->dpqn', A_grad_y1, expand_tilde_u1_grad_x1).flatten(start_dim=1, end_dim=2)
    operand_A_grad_y2_tilde_u1_grad_x2 = torch.einsum('dpn, dqn->dpqn', A_grad_y2, expand_tilde_u1_grad_x2).flatten(start_dim=1, end_dim=2)

    alpha_operand_A_grad_y1_tilde_u1_grad_x1 = torch.outer(alpha_A_grad_y1, alpha_tilde_u1).view(-1)
    alpha_operand_A_grad_y2_tilde_u1_grad_x2 = torch.outer(alpha_A_grad_y2, alpha_tilde_u1).view(-1)

    rhs_phi = torch.cat((operand_A_grad_y1_tilde_u1_grad_x1, operand_A_grad_y2_tilde_u1_grad_x2), dim=1)
    alpha_rhs = torch.cat((alpha_operand_A_grad_y1_tilde_u1_grad_x1, alpha_operand_A_grad_y2_tilde_u1_grad_x2))

    norm_rhs = Int2TNN(w1234, alpha_rhs, rhs_phi, alpha_rhs, rhs_phi)
    print('norm_rhs =', norm_rhs.item())
    print('rhs_phi',Int1TNN(w1234, alpha_rhs, rhs_phi))

def criterion_delta_u2(model_delta_u2_x, model_delta_u2_y):
    """
    损失函数：强形式
    ||-∇_y · (A∇_y u_2) - rhs||_{L2}^2
    
    其中 RHS = ∇_{y_1}A · ∇_{x_1}tilde_u1 + ∇_{y_2}A · ∇_{x_2}tilde_u1
    """
    delta_u2_phi_x = model_delta_u2_x(w1, x1, need_grad=0)
    delta_u2_phi_y, delta_u2_grad_phi_y, delta_u2_grad_grad_phi_y = model_delta_u2_y(w3, x3, need_grad=2)
    
    # 构建 LHS 的各项
    # 1. A∇_{y_1y_1}u_2
    delta_u2_grad_grad_y1 = torch.cat((delta_u2_phi_x, delta_u2_grad_grad_phi_y[0:1], delta_u2_phi_y[1:2]), dim=0)
    operand_A_delta_u2_grad_grad_y1 = torch.einsum('dpn, dqn->dpqn', A, delta_u2_grad_grad_y1)
    
    # 2. ∇_{y_1}A · ∇_{y_1}u_2
    delta_u2_grad_y1 = torch.cat((delta_u2_phi_x, delta_u2_grad_phi_y[0:1], delta_u2_phi_y[1:2]), dim=0)
    operand_A_grad_y1_delta_u2_grad_y1 = torch.einsum('dpn, dqn->dpqn', A_grad_y1, delta_u2_grad_y1)
    
    # 3. A∇_{y_2y_2}u_2
    delta_u2_grad_grad_y2 = torch.cat((delta_u2_phi_x, delta_u2_phi_y[0:1], delta_u2_grad_grad_phi_y[1:2]), dim=0)
    operand_A_delta_u2_grad_grad_y2 = torch.einsum('dpn, dqn->dpqn', A, delta_u2_grad_grad_y2)
    
    # 4. ∇_{y_2}A · ∇_{y_2}u_2
    delta_u2_grad_y2 = torch.cat((delta_u2_phi_x, delta_u2_phi_y[0:1], delta_u2_grad_phi_y[1:2]), dim=0)
    operand_A_grad_y2_delta_u2_grad_y2 = torch.einsum('dpn, dqn->dpqn', A_grad_y2, delta_u2_grad_y2)
    
    # 组合 LHS 的所有项
    my_operand_phi = torch.cat((operand_A_delta_u2_grad_grad_y1, operand_A_grad_y1_delta_u2_grad_y1,
                                operand_A_delta_u2_grad_grad_y2, operand_A_grad_y2_delta_u2_grad_y2), dim=1)
    my_operand_alpha = torch.cat((alpha_A, alpha_A_grad_y1, alpha_A, alpha_A_grad_y2))
    
    # 计算矩阵：my_matrix = ∫ (my_operand_phi, my_operand_phi)
    my_matrix = torch.sum(
        torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(
            torch.einsum('dn, dpqn, djkn->dpjqk', w1234, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    
    # 计算向量：my_vector = ∫ (my_operand_phi, rhs)
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_rhs).unsqueeze(dim=-1) * torch.prod(
        torch.einsum('dn, dpqn, dkn->dpkq', w1234, my_operand_phi, rhs_phi), dim=0), dim=(0, 1))
    
    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    
    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_rhs)
    # loss = torch.sqrt(torch.sum(torch.outer(my_alpha, my_alpha) * my_matrix) - 2 * torch.sum(my_alpha * my_vector) + norm_rhs)
    
    model_delta_u2_x.scaling_par().data.copy_(my_alpha.detach())
    
    return loss


def train_delta_u2(model_delta_u2_x, model_delta_u2_y, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        
        for e in range(epochs):
            loss = criterion_delta_u2(model_delta_u2_x, model_delta_u2_y)
            if e == 0:
                print('\n Step delta_u2 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                print('lr: ', scheduler.get_last_lr()[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_delta_u2(model_delta_u2_x, model_delta_u2_y)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])
                    torch.save(model_delta_u2_x, "./model_delta_u2/{}_final_delta_u2_x.pkl".format(name))
                    torch.save(model_delta_u2_y, "./model_delta_u2/{}_final_delta_u2_y.pkl".format(name))

    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_delta_u2(model_delta_u2_x, model_delta_u2_y)
            optimizer.zero_grad()
            loss.backward()
            return loss

        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_delta_u2(model_delta_u2_x, model_delta_u2_y)
                    print('\n Step delta_u2 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_delta_u2(model_delta_u2_x, model_delta_u2_y)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())



if __name__ == '__main__':
    sizes_periodic = [1, 5, 80, 80, 80, 20]
    activation = TNN_Sin

    if not os.path.exists("model_delta_u2"):
        os.mkdir("model_delta_u2")

    # model_delta_u2_x = torch.load("./model_delta_u2/d2_ex4_final_delta_u2_x.pkl", map_location=device)
    # model_delta_u2_y = torch.load("./model_delta_u2/d2_ex4_final_delta_u2_y.pkl", map_location=device)
    # loss = criterion_delta_u2(model_delta_u2_x, model_delta_u2_y)
    # print('loss =', loss.item())


    model_delta_u2_x = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=True).to(dtype).to(device)
    model_delta_u2_y = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=False).to(dtype).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, itertools.chain(model_delta_u2_x.parameters(), model_delta_u2_y.parameters())),
        lr=1e-3)

    epochs = 50000
    begin_time = time.time()
    train_delta_u2(model_delta_u2_x, model_delta_u2_y, optimizer, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_delta_u2_x, "./model_delta_u2/{}_final_delta_u2_x.pkl".format(name))
    torch.save(model_delta_u2_y, "./model_delta_u2/{}_final_delta_u2_y.pkl".format(name))

    # 可选：使用 LBFGS 进一步优化
    print('*' * 20, 'LBFGS', '*' * 20)
    lr = 0.1
    epochs = 5000
    optimizer = torch.optim.LBFGS(
        filter(lambda p: p.requires_grad, itertools.chain(model_delta_u2_x.parameters(), model_delta_u2_y.parameters())),
        lr=lr)
    train_delta_u2(model_delta_u2_x, model_delta_u2_y, optimizer, 'LBFGS', epochs)
    
    torch.save(model_delta_u2_x, "./model_delta_u2/{}_final_delta_u2_x.pkl".format(name))
    torch.save(model_delta_u2_y, "./model_delta_u2/{}_final_delta_u2_y.pkl".format(name))


    with torch.no_grad():
        # 分别在 delta_u2_x 和 delta_u2_y 网格上插值当前的 delta_u2
        # 1. 计算 delta_u2 当前值
        delta_u2_phi_x, delta_u2_phi_grad_x = model_delta_u2_x(w1, x1, need_grad=1)
        delta_u2_phi_y, delta_u2_phi_grad_y = model_delta_u2_y(w3, x3, need_grad=1)
        alpha_delta_u2 = model_delta_u2_x.scaling_par()
        delta_u2_x_norm_phi = model_delta_u2_x(w1, x1, normed=False)
        delta_u2_x_norm = torch.sqrt(torch.sum(w1 * delta_u2_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
        delta_u2_y_norm_phi = model_delta_u2_y(w3, x3, normed=False)
        delta_u2_y_norm = torch.sqrt(torch.sum(w3 * delta_u2_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
        delta_u2_y_shift = model_delta_u2_y(dummy, x_error_0, need_grad=0, normed=False) #[d,p,1]
        delta_u2_y_shift = delta_u2_y_shift / delta_u2_y_norm
        delta_u2 = torch.cat((delta_u2_phi_x, delta_u2_phi_y), dim=0)
        delta_u2_shift = torch.cat((delta_u2_phi_x, delta_u2_y_shift.expand(2, p_delta_u2, N)))
        delta_u2_grad_x1 = torch.cat((delta_u2_phi_grad_x[0:1, :, :],delta_u2_phi_x[1:2, :, :], delta_u2_phi_y), dim=0)
        delta_u2_grad_x1_shift = torch.cat((delta_u2_phi_grad_x[0:1, :, :],delta_u2_phi_x[1:2, :, :], delta_u2_y_shift.expand(2, p_delta_u2, N)), dim=0)
        delta_u2_grad_y1 = torch.cat((delta_u2_phi_x, delta_u2_phi_grad_y[0:1, :, :], delta_u2_phi_y[1:2, :, :]), dim=0)

        # 2. 加载/重新计算 chi1, chi2
        model_chi1_x = torch.load('./model_chi1/d2_ex4_final_chi1_x.pkl', map_location=device)
        model_chi1_y = torch.load('./model_chi1/d2_ex4_final_chi1_y.pkl', map_location=device)
        model_chi2_x = torch.load('./model_chi2/d2_ex4_final_chi2_x.pkl', map_location=device)
        model_chi2_y = torch.load('./model_chi2/d2_ex4_final_chi2_y.pkl', map_location=device)
        chi1_phi_x, chi1_phi_grad_x = model_chi1_x(w1, x1, need_grad=1)
        chi1_phi_y, chi1_phi_grad_y = model_chi1_y(w3, x3, need_grad=1)
        alpha_chi1 = model_chi1_x.scaling_par()
        chi1_x_norm_phi = model_chi1_x(w1, x1, normed=False)
        chi1_x_norm = torch.sqrt(torch.sum(w1 * chi1_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
        chi1_y_norm_phi = model_chi1_y(w3, x3, normed=False)
        chi1_y_norm = torch.sqrt(torch.sum(w3 * chi1_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
        chi1_y_shift = model_chi1_y(dummy, x_error_0, need_grad=0, normed=False) #[d,p,1]
        chi1_y_shift = chi1_y_shift / chi1_y_norm

        chi2_phi_x, chi2_phi_grad_x = model_chi2_x(w1, x1, need_grad=1)
        chi2_phi_y, chi2_phi_grad_y = model_chi2_y(w3, x3, need_grad=1)
        alpha_chi2 = model_chi2_x.scaling_par()
        chi2_x_norm_phi = model_chi2_x(w1, x1, normed=False)
        chi2_x_norm = torch.sqrt(torch.sum(w1 * chi2_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
        chi2_y_norm_phi = model_chi2_y(w3, x3, normed=False)
        chi2_y_norm = torch.sqrt(torch.sum(w3 * chi2_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
        chi2_y_shift = model_chi2_y(dummy, x_error_0, need_grad=0, normed=False) #[d,p,1]
        chi2_y_shift = chi2_y_shift / chi2_y_norm

        chi1 = torch.cat((chi1_phi_x, chi1_phi_y), dim=0)
        chi1_shift = torch.cat((chi1_phi_x, chi1_y_shift.expand(2, p_chi, N)), dim=0)
        chi2 = torch.cat((chi2_phi_x, chi2_phi_y), dim=0)
        chi2_shift = torch.cat((chi2_phi_x, chi2_y_shift.expand(2, p_chi, N)), dim=0)
        chi1_grad_x1 = torch.cat((chi1_phi_grad_x[0:1, :, :], chi1_phi_x[1:2, :, :], chi1_phi_y), dim=0)
        chi1_grad_x1_shift = torch.cat((chi1_phi_grad_x[0:1, :, :], chi1_phi_x[1:2, :, :], chi1_y_shift.expand(2, p_chi, N)), dim=0)
        chi2_grad_x1 = torch.cat((chi2_phi_grad_x[0:1, :, :], chi2_phi_x[1:2, :, :], chi2_phi_y), dim=0)
        chi2_grad_x1_shift = torch.cat((chi2_phi_grad_x[0:1, :, :], chi2_phi_x[1:2, :, :], chi2_y_shift.expand(2, p_chi, N)), dim=0)
        chi1_grad_x2 = torch.cat((chi1_phi_x[0:1, :, :], chi1_phi_grad_x[1:2, :, :], chi1_phi_y), dim=0)
        chi1_grad_x2_shift = torch.cat((chi1_phi_x[0:1, :, :], chi1_phi_grad_x[1:2, :, :], chi1_y_shift.expand(2, p_chi, N)), dim=0)
        chi2_grad_x2 = torch.cat((chi2_phi_x[0:1, :, :], chi2_phi_grad_x[1:2, :, :], chi2_phi_y), dim=0)
        chi2_grad_x2_shift = torch.cat((chi2_phi_x[0:1, :, :], chi2_phi_grad_x[1:2, :, :], chi2_y_shift.expand(2, p_chi, N)), dim=0)
        chi1_grad_y1 = torch.cat((chi1_phi_x, chi1_phi_grad_y[0:1, :, :], chi1_phi_y[1:2, :, :]), dim=0)
        chi2_grad_y1 = torch.cat((chi2_phi_x, chi2_phi_grad_y[0:1, :, :], chi2_phi_y[1:2, :, :]), dim=0)
        chi1_grad_y2 = torch.cat((chi1_phi_x, chi1_phi_y[0:1, :, :], chi1_phi_grad_y[1:2, :, :]), dim=0)
        chi2_grad_y2 = torch.cat((chi2_phi_x, chi2_phi_y[0:1, :, :], chi2_phi_grad_y[1:2, :, :]), dim=0)
        
        
        # 3. 计算 tilde_u1 在 x 上的梯度（∇_{x1} tilde_u1, ∇_{x2} tilde_u1）
        tilde_u1_phi, tilde_u1_grad_phi, tilde_u1_grad_grad_phi = model_tilde_u1(w1, x1, need_grad=2)
        tilde_u1_norm_phi = model_tilde_u1(w1, x1, normed=False)
        tilde_u1_norm = torch.sqrt(torch.sum(w1 * tilde_u1_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]

        tilde_u1_grad_x1 = torch.cat((tilde_u1_grad_phi[0:1, :, :], tilde_u1_phi[1:2, :, :], torch.ones((2, p_tilde_u1, N), dtype=dtype, device=device)), dim=0)
        tilde_u1_grad_x2 = torch.cat((tilde_u1_phi[0:1, :, :], tilde_u1_grad_phi[1:2, :, :], torch.ones((2, p_tilde_u1, N), dtype=dtype, device=device)), dim=0)
        tilde_u1_grad_grad_x1 = torch.cat((tilde_u1_grad_grad_phi[0:1, :, :], tilde_u1_phi[1:2, :, :], torch.ones((2, p_tilde_u1, N), dtype=dtype, device=device)), dim=0)
        tilde_u1_grad_x1_x2 = torch.cat((tilde_u1_grad_phi, torch.ones((2, p_tilde_u1, N), dtype=dtype, device=device)), dim=0)
        tilde_u1_grad_grad_x2 = torch.cat((tilde_u1_phi[0:1, :, :], tilde_u1_grad_grad_phi[1:2, :, :], torch.ones((2, p_tilde_u1, N), dtype=dtype, device=device)), dim=0)
        alpha_part_chi1 = torch.outer(alpha_chi1, alpha_tilde_u1).view(-1)
        alpha_part_chi2 = torch.outer(alpha_chi2, alpha_tilde_u1).view(-1)


        # (chi1 * ∇_{x1}tilde_u1 + chi2 * ∇_{x2}tilde_u1)
        # grad_x1 tilde_u1 × chi1
        chi1_tilde_u1_grad_x1 = torch.einsum('dpn,dqn->dpqn', chi1, tilde_u1_grad_x1).flatten(start_dim=1,end_dim=2)
        chi1_shift_tilde_u1_grad_x1 = torch.einsum('dpn,dqn->dpqn', chi1_shift, tilde_u1_grad_x1).flatten(start_dim=1,end_dim=2)
        # grad_x2 tilde_u1 × chi2
        chi2_tilde_u1_grad_x2 = torch.einsum('dpn,dqn->dpqn', chi2, tilde_u1_grad_x2).flatten(start_dim=1,end_dim=2)
        chi2_shift_tilde_u1_grad_x2 = torch.einsum('dpn,dqn->dpqn', chi2_shift, tilde_u1_grad_x2).flatten(start_dim=1,end_dim=2)

        operand_F = torch.cat((delta_u2,delta_u2_shift, chi1_tilde_u1_grad_x1,chi1_shift_tilde_u1_grad_x1, chi2_tilde_u1_grad_x2,chi2_shift_tilde_u1_grad_x2),dim=1)
        operand_alpha = torch.cat((-alpha_delta_u2,alpha_delta_u2, alpha_part_chi1,-alpha_part_chi1, alpha_part_chi2,-alpha_part_chi2))
        error1 = torch.sqrt(Int2TNN(w1234, operand_alpha, operand_F, operand_alpha, operand_F))


        print('\n' + '*' * 40)
        print('L2 error between current delta_u2 and chi1 * grad_x1 tilde_u1 + chi2 * grad_x2 tilde_u1:')
        print('Absolute error (using 4D integration) = ', error1.item())
        print('*' * 40 + '\n')

        #=======================
        part1 = torch.einsum('dpn,dqn->dpqn', chi1_grad_x1, tilde_u1_grad_x1).flatten(start_dim=1,end_dim=2)
        part1_shift = torch.einsum('dpn,dqn->dpqn', chi1_grad_x1_shift, tilde_u1_grad_x1).flatten(start_dim=1,end_dim=2)
        part2 = torch.einsum('dpn,dqn->dpqn', chi2_grad_x1, tilde_u1_grad_x2).flatten(start_dim=1,end_dim=2)
        part2_shift = torch.einsum('dpn,dqn->dpqn', chi2_grad_x1_shift, tilde_u1_grad_x2).flatten(start_dim=1,end_dim=2)

        part3 = torch.einsum('dpn,dqn->dpqn', chi1, tilde_u1_grad_grad_x1).flatten(start_dim=1,end_dim=2)
        part3_shift = torch.einsum('dpn,dqn->dpqn', chi1_shift, tilde_u1_grad_grad_x1).flatten(start_dim=1,end_dim=2)

        part4 = torch.einsum('dpn,dqn->dpqn', chi2, tilde_u1_grad_x1_x2).flatten(start_dim=1,end_dim=2)
        part4_shift = torch.einsum('dpn,dqn->dpqn', chi2_shift, tilde_u1_grad_x1_x2).flatten(start_dim=1,end_dim=2)

        
        F = torch.cat((delta_u2_grad_x1,delta_u2_grad_x1_shift, part1,part1_shift, part2,part2_shift, part3,part3_shift, part4,part4_shift), dim=1)
        alpha_F = torch.cat((-alpha_delta_u2,alpha_delta_u2, alpha_part_chi1,-alpha_part_chi1, alpha_part_chi2,-alpha_part_chi2, alpha_part_chi1,-alpha_part_chi1, alpha_part_chi2,-alpha_part_chi2))
        error2 = torch.sqrt(Int2TNN(w1234, alpha_F, F, alpha_F, F))

        print(('Absolute error between current delta_u2_grad_x1 and chi1 * grad_x1 tilde_u1 + chi2 * grad_x2 tilde_u1:'), error2.item())
        
        #=======================
        part5 = torch.einsum('dpn,dqn->dpqn', chi1_grad_y1, tilde_u1_grad_x1).flatten(start_dim=1,end_dim=2)
        part6 = torch.einsum('dpn,dqn->dpqn', chi2_grad_y1, tilde_u1_grad_x2).flatten(start_dim=1,end_dim=2)

        F1 = torch.cat((delta_u2_grad_y1, part5, part6), dim=1)
        alpha_F1 = torch.cat((-alpha_delta_u2, alpha_part_chi1, alpha_part_chi2))
        error3 = torch.sqrt(Int2TNN(w1234, alpha_F1, F1, alpha_F1, F1))

        print(('Absolute error between current delta_u2_grad_y1 and chi1 * grad_y1 tilde_u1 + chi2 * grad_y2 tilde_u1:'), error3.item())
