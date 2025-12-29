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
device = 'cuda:2'
name = 'x_free_y_per'

# 求前m个特征值
m = 5
dim=1
p = 18
a, b = 0, 1

p_A = 2
p_chi1 = 20
p_A0 = p_A + p_A * p_chi1

x1, w1 = composite_quadrature_1d(8, 0, 1, 200, device=device)
x2, w2 = composite_quadrature_1d(8, 0, 1, 200, device=device)

N = len(x1)


# exactA0
A0_exact = torch.sqrt((x1 + 1) * (x1 + 3))
alpha_A0_exact=torch.ones(1, dtype=dtype, device=device)

norm_A0_exact = torch.sqrt(torch.sum(w1 * A0_exact ** 2))

A = torch.ones((2, 2, N), dtype=dtype, device=device)
A[0, 0, :] = 2 + x1
A[1, 1, :] = torch.sin(2*pi * x2)
alpha_A = torch.ones(2, dtype=dtype, device=device)

# A_partial_y1 = torch.ones((2, 1, N), dtype=dtype, device=device)
# A_partial_y1[1, 0, :] = 2 * pi * torch.cos(2 * pi * x2)
# alpha_A_partial_y1 = torch.ones(1, dtype=dtype, device=device)

A_partial_x = torch.ones((2, 2, N), dtype=dtype, device=device)
A_partial_x[0,1,:] = 0
alpha_A_partial_x = torch.ones(1, dtype=dtype, device=device)



# p_A = 1
# # exactA0
# A0_exact = np.sqrt(3)/2*(x1 + 1)
# alpha_A0_exact=torch.ones(1, dtype=dtype, device=device)

# norm_A0_exact = torch.sqrt(torch.sum(w1 * A0_exact ** 2))

# A = torch.ones((2, 1, N), dtype=dtype, device=device)
# A[0, 0, :] = 1 + x1
# A[1, 0, :] = 1 + 0.5 * torch.sin(2*pi * x2)
# alpha_A = torch.ones(1, dtype=dtype, device=device)

# A_partial_x = torch.ones((2, 1, N), dtype=dtype, device=device)
# A_partial_x[1,0,:] = 1 + 0.5 * torch.sin(2*pi * x2)
# alpha_A_partial_x = torch.ones(1, dtype=dtype, device=device)

 
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

# model_chi1_x = torch.load('./model_chi1/x_plus_y_partbd_0per_40_final_chi1_x.pkl', map_location=device)
# model_chi1_y1 = torch.load('./model_chi1/x_plus_y_partbd_0per_40_final_chi1_y1.pkl', map_location=device)

model_chi1_x = torch.load('./model_chi1/x_free_y_per_final_chi1_x.pkl', map_location=device)
model_chi1_y1 = torch.load('./model_chi1/x_free_y_per_final_chi1_y1.pkl', map_location=device)

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

    A0_pointwise = torch.sum(alpha_A0[0:p_A].view(-1, 1) * A0[0, 0:p_A, :], dim=0) \
                        + torch.sum(alpha_A0[p_A:p_A + p_A * p_chi1].view(-1, 1) * A0[0, p_A:p_A + p_A * p_chi1, :], dim=0)
        
    print('relative error of homogenized matrix A0 : ', (torch.sqrt(torch.sum(w1 * (A0_pointwise - A0_exact)**2)) / norm_A0_exact).item())   


def einsum_flatten(a, b):
    return torch.einsum('dpn,dqn->dpqn', a, b).reshape(a.shape[0], -1, a.shape[2])

def batch_outer(alpha, u_batch):
    return torch.einsum('d,fm->dfm',alpha,u_batch).reshape(alpha.shape[0] * u_batch.shape[0], u_batch.shape[1])


# 计算\int A0_TNN (\nabla u)^2 dx =\lambda0 \int u^2 dx 对应的lambda0
def criterion_0(model_u0):
    u0_phi, u0_grad_phi, u0_grad_grad_phi = model_u0(w1, x1, need_grad=2)
    
    alpha = torch.ones(p, device=device, dtype=dtype)
    part1 = Int2TNN_amend_1d_with_coe(w1, w1, alpha, u0_phi, alpha, u0_phi, u0_grad_phi, u0_grad_phi, alpha_A0, A0,
                                      if_sum=False)
    part0 = Int2TNN(w1, alpha, u0_phi, alpha, u0_phi, if_sum=False)

    A = part1
    M = part0

    L = torch.linalg.cholesky(M)  # Cholesky分解：M = L L^T
    C = torch.linalg.solve(L, A.t()).t()  # C = A L^{-T}
    D = torch.linalg.solve(L, C)  # D = L^{-1} A L^{-T}
    E, V = torch.linalg.eigh(D)  # 解 D 的特征值问题
    ind = torch.argsort(E)
    # loss = torch.sum(E[ind][:m])
    lam = E[ind]
    u = torch.linalg.solve(L.t(),V[:,ind])
    # 后验误差loss
    my_operand_phi = torch.cat([einsum_flatten(A0_grad_x, u0_grad_phi),einsum_flatten(A0, u0_grad_grad_phi),u0_phi], dim=1)

    u_batch = u[:,0:m]
    my_operand_alpha = torch.cat([batch_outer(alpha_A0, u_batch), batch_outer(alpha_A0, u_batch), (lam[:m].unsqueeze(0)* u_batch)], dim=0).T
    raw_s = torch.abs(Int2TNN_batch(w1, my_operand_alpha, my_operand_phi, my_operand_alpha, my_operand_phi, if_sum=True))
    s = torch.sqrt(torch.abs(raw_s))
    has_zero = torch.any(torch.isclose(s, torch.tensor(0.0, dtype=dtype, device=device), atol=1e-14)) | \
               torch.any(torch.abs(raw_s) < 1e-30)  # 检查raw_s是否过小（可能是数值下溢）
    if not has_zero:
        loss = torch.sum(s)
    else:
        print('有零值，逐样本单独计算')
        print('有零值的equloss',s)
        loss = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)
        for i in range(m):
            u_i = u[:, i]  # 第i个特征向量
            lam_i = lam[i]  # 第i个特征值
            
            alpha_A_grad_x_u_grad_x = torch.outer(alpha_A0_grad_x, u_i).view(-1)
            alpha_A_u_grad2_x = torch.outer(alpha_A0, u_i).view(-1)
            my_operand_alpha_i = torch.cat((alpha_A_grad_x_u_grad_x, alpha_A_u_grad2_x, lam_i * u_i))
            
            single_raw = Int2TNN(w1, my_operand_alpha_i, my_operand_phi, my_operand_alpha_i, my_operand_phi, if_sum=True)
            single_equ_loss = torch.sqrt(torch.abs(single_raw))
            
            # 如果单个样本的残差仍然为0或过小，可能是数值下溢或真正的精确解
            # 添加一个小的正则化项避免梯度消失，或者使用更稳定的计算方式
            if torch.abs(single_raw) < 1e-30:
                print(f'警告: 样本{i}的强形式残差过小 ({single_raw.item():.2e})，可能是数值下溢或精确解')
                # 使用更稳定的计算：避免直接开方，而是使用log空间计算
                # 或者添加一个小的正则化项
                single_equ_loss = torch.clamp(single_equ_loss, min=1e-15)
            
            loss = loss + single_equ_loss
    return loss


def equ_error_test(model_u0):
    u0_phi, u0_grad_phi, u0_grad_grad_phi = model_u0(w1, x1, need_grad=2)

    alpha = torch.ones(p, device=device, dtype=dtype)
    part1 = Int2TNN_amend_1d_with_coe(w1, w1, alpha, u0_phi, alpha, u0_phi, u0_grad_phi, u0_grad_phi, alpha_A0, A0,
                                      if_sum=False)
    part0 = Int2TNN(w1, alpha, u0_phi, alpha, u0_phi, if_sum=False)

    A = part1
    M = part0


    L = torch.linalg.cholesky(M)  # Cholesky分解：M = L L^T
    C = torch.linalg.solve(L, A.t()).t()  # C = A L^{-T}
    D = torch.linalg.solve(L, C)  # D = L^{-1} A L^{-T}
    E, V = torch.linalg.eigh(D)  # 解 D 的特征值问题
    ind = torch.argsort(E)
    lam = E[ind]
    u = torch.linalg.solve(L.t(), V[:, ind]) # 求出广义特征向量

    print('A0TNN_eigenvalues', lam[:m].detach().cpu().numpy())
    # print('lam_absolute_error', torch.abs(lam[:m] - exactlam[:m]))
    # print('lam_relative_error', torch.abs(lam[:m] - exactlam[:m])/exactlam[:m])

    my_operand_phi = torch.cat([einsum_flatten(A0_grad_x, u0_grad_phi),einsum_flatten(A0, u0_grad_grad_phi),u0_phi], dim=1)

    u_batch = u[:,0:m]
    my_operand_alpha = torch.cat([batch_outer(alpha_A0, u_batch), batch_outer(alpha_A0, u_batch), (lam[:m].unsqueeze(0)* u_batch)], dim=0).T
    equ_loss = torch.sqrt(Int2TNN_batch(w1, my_operand_alpha, my_operand_phi, my_operand_alpha, my_operand_phi, if_sum=True))
    print('equ loss',equ_loss)


def train_0(model_u0, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_0(model_u0)
            if e == 0:
                print('\n Step 3 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                equ_error_test(model_u0)
                print('lr: ', scheduler.get_last_lr()[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_0(model_u0)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    equ_error_test(model_u0)
                    print('lr: ', scheduler.get_last_lr()[0])
                    torch.save(model_u0, "./model_u0/{}_final_u0.pkl".format(name))

    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_0(model_u0)
            optimizer.zero_grad()
            loss.backward()
            return loss

        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_0(model_u0)
                    print('\n Step 3 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())
                    equ_error_test(model_u0)

            loss = optimizer.step(closure)

            if (e + 1) % 1 == 0:
                with torch.no_grad():
                    loss = criterion_0(model_u0)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    equ_error_test(model_u0)
                    torch.save(model_u0, "./model_u0/{}_final_u0.pkl".format(name))

            

if __name__ == '__main__':
    sizes = [1, 80, 80, 80, 80, p]
    activation = TNN_Sin

    if not os.path.exists("model_u0"):
        os.mkdir("model_u0")

    model_u0 = TNN(1, sizes, activation, bd=bd_u0, grad_bd=grad_bd_u0, grad_grad_bd=grad_grad_bd_u0, \
        grad_grad_grad_bd=grad_grad_grad_bd_u0, scaling=True).to(dtype).to(device)
    # model_u0 = TNN(1, sizes, activation, scaling=True).to(dtype).to(device)
    # model_u0 = torch.load('./model_u0/ex4_final_u0.pkl',map_location=device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, itertools.chain(model_u0.parameters())), lr=1e-2)

    epochs = 20000
    begin_time = time.time()
    train_0(model_u0, optimizer, 'Adam', epochs)
    torch.save(model_u0, "./model_u0/{}_final_u0.pkl".format(name))

    print('*' * 20, 'LBFGS', '*' * 20)
    lr = 0.1
    epochs = 1000
    save = True 
    # optimizer used
    optimizer = torch.optim.LBFGS(model_u0.parameters(), lr=lr)
    train_0(model_u0, optimizer, 'LBFGS', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    # print('Elapsed time is {} seconds.'.format(end_time - begin_time))
    
    torch.save(model_u0, "./model_u0/{}_final_u0.pkl".format(name))