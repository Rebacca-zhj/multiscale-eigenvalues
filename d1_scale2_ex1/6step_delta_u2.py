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
name = 'd1_ex1_test3'

# 求前m个特征值
# m = 1
dim=1

a, b = 0, 1

p_A = 2
p_tilde_u1 = 20
p_delta_u2 = 20
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

A_partial_y = torch.ones((2, 2, N), dtype=dtype, device=device)
A_partial_y[:, 0, :] = 0
A_partial_y[1, 1, :] = 2 * pi * torch.cos(2 * pi * x2)
alpha_A_partial_y = torch.ones(2, dtype=dtype, device=device)

model_tilde_u1 = torch.load('./model_tilde_u1/d1_ex1_final_tilde_u1.pkl', map_location=device)

with torch.no_grad():
    _, tilde_u1_grad_phi = model_tilde_u1(w1, x1, need_grad=1)
    alpha_tilde_u1 = model_tilde_u1.scaling_par()
    
    # rhs_term = \nabla_y A \cdot \nabla_x tilde_u1
    rhs_term = torch.ones((2, p_A * p_tilde_u1, N), dtype=dtype, device=device)
    rhs_term[0:1,:,:] = torch.einsum('dpn,dqn->dpqn', A_partial_y[0:1, :, :], tilde_u1_grad_phi).flatten(start_dim=1, end_dim=2)
    rhs_term[1:2,:,:] = torch.einsum('dpn,dqn->dpqn', A_partial_y[1:2, :, :], torch.ones((1, p_tilde_u1, N), dtype=dtype, device=device)).flatten(start_dim=1, end_dim=2) 
    alpha_rhs_term = torch.outer(alpha_A, alpha_tilde_u1).view(-1)
    norm_square_rhs_term = Int2TNN(w12, alpha_rhs_term, rhs_term, alpha_rhs_term, rhs_term)

def criterion_delta_u2(model_delta_u2_x, model_delta_u2_y):
    delta_u2_phi_x = model_delta_u2_x(w1, x1, need_grad=0)
    _, delta_u2_grad_phi_y, delta_u2_grad_grad_phi_y = model_delta_u2_y(w2, x2, need_grad=2)

    delta_u2_partial_y = torch.cat((delta_u2_phi_x, delta_u2_grad_phi_y), dim=0)
    delta_u2_partial2_y = torch.cat((delta_u2_phi_x, delta_u2_grad_grad_phi_y), dim=0)

    operand_A_partial_y_delta_u2_partial_y = torch.einsum('dpn, dqn->dpqn', A_partial_y, delta_u2_partial_y)
    operand_A_delta_u2_partial2_y = torch.einsum('dpn, dqn->dpqn', A, delta_u2_partial2_y)

    my_operand_phi = torch.cat((operand_A_partial_y_delta_u2_partial_y, operand_A_delta_u2_partial2_y), dim=1)
    my_operand_alpha = torch.cat((alpha_A_partial_y, alpha_A))

    # p * p matrix
    my_matrix = torch.sum(
        torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(
            torch.einsum('dn, dpqn, djkn->dpjqk', w12, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    # p vector
          
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_rhs_term).unsqueeze(dim=-1) * torch.prod(
        torch.einsum('dn, dpqn, dkn->dpkq', w12, my_operand_phi, rhs_term), dim=0), dim=(0, 1))

    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_delta_u2_x.scaling_par().data.copy_(my_alpha)

    # my_alpha = model_delta_u2_x.scaling_par()

    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_square_rhs_term)

    return loss


def train_delta_u2(model_delta_u2_x, model_delta_u2_y, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_delta_u2(model_delta_u2_x, model_delta_u2_y)
            if e == 0:
                print('\n Step 6 {} Initial:'.format(optimizer_flag))
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
                    print('\n Step 1 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_delta_u2(model_delta_u2_x, model_delta_u2_y)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())


if __name__ == '__main__':
    # sizes, sizes_periodic = [1, 50, 50, 50, 50, p_delta_u2], [1, 2, 50, 50, p_delta_u2]#2.38361032337617e-05
    sizes, sizes_periodic = [1, 80, 80, 80, 80, p_delta_u2], [1, 5, 80, 80, p_delta_u2]#2.5399667260234823e-06（✅）
    # sizes, sizes_periodic = [1, 100, 100, 100, 100, p_delta_u2], [1, 8, 100, 100, p_delta_u2]#
    activation = TNN_Sin

    if not os.path.exists("model_delta_u2"):
        os.mkdir("model_delta_u2")
 
    model_delta_u2_x = TNN(1, sizes, activation, scaling=True).to(dtype).to(device)
    model_delta_u2_y = TNN_periodic(1, sizes_periodic, activation, period=1, scaling=False).to(dtype).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, itertools.chain(model_delta_u2_x.parameters(), model_delta_u2_y.parameters())),
        lr=1e-2)

    epochs = 40000
    begin_time = time.time()
    train_delta_u2(model_delta_u2_x, model_delta_u2_y, optimizer, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_delta_u2_x, "./model_delta_u2/{}_final_delta_u2_x.pkl".format(name))
    torch.save(model_delta_u2_y, "./model_delta_u2/{}_final_delta_u2_y.pkl".format(name))

    
    epochs = 4000
    save = True 
    # optimizer used
    begin_time = time.time()
    train_delta_u2(model_delta_u2_x, model_delta_u2_y, optimizer, 'LBrhs_termGS', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_delta_u2_x, "./model_delta_u2/{}_final_delta_u2_x.pkl".format(name))
    torch.save(model_delta_u2_y, "./model_delta_u2/{}_final_delta_u2_y.pkl".format(name))
    