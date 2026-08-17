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
device = 'cuda:4'
name = 'd1_ex1'

# 求前m个特征值
# m = 1
dim=1

a, b = 0, 1
p_A = 2
p_u0 = 20
p_chi1 = 20
p_u1 = 20
p_A0 = p_A + p_A * p_chi1
x1, w1 = composite_quadrature_1d(4, 0, 1, 200, device=device)
x2, w2 = composite_quadrature_1d(4, 0, 1, 200, device=device)

dummy = torch.ones((1, 1), dtype=dtype, device=device)

x_error_0 = torch.zeros(1, dtype=dtype, device=device)

N = len(x1)
w12 = torch.stack((w1, w2), dim=0)

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

A = torch.ones((2, 2, N), dtype=dtype, device=device)
A[0, 0, :] = 2 + x1
A[1, 1, :] = torch.sin(2 * pi * x2)
alpha_A = torch.ones(2, dtype=dtype, device=device)

A_grad = torch.zeros((2, 2, N), dtype=dtype, device=device)
A_grad[0, 0, :] = 1
A_grad[1, 1, :] = 2 * pi * torch.cos(2 * pi * x2)

A_partial_y = torch.ones((2, 2, N), dtype=dtype, device=device)
A_partial_y[0, 0, :] = 0
A_partial_y[1, 1, :] = 2 * pi * torch.cos(2 * pi * x2)
alpha_A_partial_y = torch.ones(2, dtype=dtype, device=device)

model_u0 = torch.load('./model_u0/d1_ex1_final_u0.pkl',map_location=device)

with torch.no_grad():
    _, u0_grad_phi = model_u0(w1, x1, need_grad=1)
    alpha_u0 = model_u0.scaling_par()
    expand_u0_grad_x = torch.cat((u0_grad_phi, torch.ones((1, p_u0, N), dtype=dtype, device=device)), dim=0)
    
    # F = \nabla_y A \cdot \nabla_x u0
    F = torch.einsum('dpn,dqn->dpqn', A_partial_y, expand_u0_grad_x).flatten(start_dim=1, end_dim=2)
    alpha_F = torch.outer(alpha_A, alpha_u0).view(-1)
    norm_square_F = Int2TNN(w12, alpha_F, F, alpha_F, F)

def criterion_u1(model_u1_x, model_u1_y):
    u1_phi_x = model_u1_x(w1, x1, need_grad=0)
    _, u1_grad_phi_y, u1_grad_grad_phi_y = model_u1_y(w2, x2, need_grad=2)

    u1_partial_y = torch.cat((u1_phi_x, u1_grad_phi_y), dim=0)
    u1_partial2_y = torch.cat((u1_phi_x, u1_grad_grad_phi_y), dim=0)

    operand_A_partial_y_u1_partial_y = torch.einsum('dpn, dqn->dpqn', A_partial_y, u1_partial_y)
    operand_A_u1_partial2_y = torch.einsum('dpn, dqn->dpqn', A, u1_partial2_y)

    my_operand_phi = torch.cat((operand_A_partial_y_u1_partial_y, operand_A_u1_partial2_y), dim=1)
    my_operand_alpha = torch.cat((alpha_A_partial_y, alpha_A))

    # p * p matrix
    my_matrix = torch.sum(
        torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(
            torch.einsum('dn, dpqn, djkn->dpjqk', w12, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    # p vector
          
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_F).unsqueeze(dim=-1) * torch.prod(
        torch.einsum('dn, dpqn, dkn->dpkq', w12, my_operand_phi, F), dim=0), dim=(0, 1))

    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_u1_x.scaling_par().data.copy_(my_alpha)

    # my_alpha = model_u1_x.scaling_par()

    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_square_F)
    # loss = torch.sqrt(torch.sum(torch.outer(my_alpha, my_alpha) * my_matrix) - 2 * torch.sum(my_alpha * my_vector) + norm_square_A1_partial_y)

    return loss


def train_1(model_u1_x, model_u1_y, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_u1(model_u1_x, model_u1_y)
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
                    loss = criterion_u1(model_u1_x, model_u1_y)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])

        

    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_u1(model_u1_x, model_u1_y)
            optimizer.zero_grad()
            loss.backward()
            return loss

        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_u1(model_u1_x, model_u1_y)
                    print('\n Step 1 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_u1(model_u1_x, model_u1_y)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())

            


if __name__ == '__main__':
    p_net = 50
    p_net_periodic = 2
    sizes = [1, p_net, p_net, p_net, p_net, p_u1]
    sizes_periodic = [1, p_net_periodic, p_net, p_net, p_u1]
    activation = TNN_Sin

    if not os.path.exists("model_u1"):
        os.mkdir("model_u1")
 
    model_u1_x = TNN(1, sizes, activation, scaling=True).to(dtype).to(device)
    model_u1_y = TNN_periodic(1, sizes_periodic, activation, period=1, scaling=False).to(dtype).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, itertools.chain(model_u1_x.parameters(), model_u1_y.parameters())),
        lr=1e-2)

    epochs = 40000
    begin_time = time.time()
    train_1(model_u1_x, model_u1_y, optimizer, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_u1_x, "./model_u1/{}_final_u1_x.pkl".format(name))
    torch.save(model_u1_y, "./model_u1/{}_final_u1_y.pkl".format(name))

    
    epochs = 4000
    save = True 
    # optimizer used
    begin_time = time.time()
    train_1(model_u1_x, model_u1_y, optimizer, 'LBFGS', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_u1_x, "./model_u1/{}_final_u1_x.pkl".format(name))
    torch.save(model_u1_y, "./model_u1/{}_final_u1_y.pkl".format(name))
    
    # # 验证误差  u1 ~ chi1 * \nabla_x u0
    # model_chi1_x = torch.load('./model_chi1/x_free_y_per_final_chi1_x.pkl', map_location=device)
    # model_chi1_y = torch.load('./model_chi1/x_free_y_per_final_chi1_y.pkl', map_location=device)
    
    # with torch.no_grad():
    #     chi1_phi = torch.ones((2, p_chi1, N), dtype=dtype, device=device)
    #     chi1_grad_phi = torch.zeros((2, p_chi1, N), dtype=dtype, device=device)
    #     chi1_phi[0:1, :, :], chi1_grad_phi[0:1, :, :] = model_chi1_x(w1, x1, need_grad=1)
    #     chi1_phi[1:2, :, :], chi1_grad_phi[1:2, :, :] = model_chi1_y(w2, x2, need_grad=1)
    #     alpha_chi1 = model_chi1_x.scaling_par()
        
        

    
