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
device = 'cuda:6'
name = 'd1_ex1_test5'

# 求前m个特征值
# m = 1
dim=1

a, b = 0, 1

p_A = 2
p_chi1 = 20
p_u0 = 20
p_u1 = 20
p_u2 = 20
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

A_partial_x = torch.ones((2, 2, N), dtype=dtype, device=device)
A_partial_x[0, 0, :] = 1
A_partial_x[1, 1, :] = 0
alpha_A_partial_x = torch.ones(2, dtype=dtype, device=device)

A_partial_y = torch.ones((2, 2, N), dtype=dtype, device=device)
A_partial_y[0, 0, :] = 0
A_partial_y[1, 1, :] = 2 * pi * torch.cos(2 * pi * x2)
alpha_A_partial_y = torch.ones(2, dtype=dtype, device=device)

F = x1.view(1, 1, -1)
alpha_F = 3 * torch.ones(1, dtype=dtype, device=device)
expand_F = torch.cat((F, torch.ones((1, 1, N), dtype=dtype, device=device)), dim=0)

model_u0_x = torch.load('./model_u0/d1_ex1_final_u0.pkl', map_location=device)
model_u1_x = torch.load('./model_u1/d1_ex1_final_u1_x.pkl', map_location=device)
model_u1_y = torch.load('./model_u1/d1_ex1_final_u1_y.pkl', map_location=device)

with torch.no_grad():
    u0_phi, u0_grad_phi, u0_grad_grad_phi = model_u0_x(w1, x1, need_grad=2)
    alpha_u0 = model_u0_x.scaling_par()

    expand_u0_grad_x = torch.cat((u0_grad_phi, torch.ones((1, p_u0, N), dtype=dtype, device=device)), dim=0)
    expand_u0_grad2_x = torch.cat((u0_grad_grad_phi, torch.ones((1, p_u0, N), dtype=dtype, device=device)), dim=0)

    u1_x_phi, u1_x_grad_phi = model_u1_x(w1, x1, need_grad=1)
    u1_y_phi, u1_y_grad_phi = model_u1_y(w2, x2, need_grad=1)
    alpha_u1 = model_u1_x.scaling_par()
    
    u1_y_norm_phi = model_u1_y(w2, x2, normed=False)
    u1_y_norm = torch.sqrt(torch.sum(w2 * u1_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    u1_y_shift = model_u1_y(dummy, x_error_0, need_grad=0, normed=False) #[d,p,1]
    u1_y_shift = u1_y_shift / u1_y_norm

    u1_grad_x = torch.cat((u1_x_grad_phi, u1_y_phi), dim=0)
    u1_grad_x_shift = torch.cat((u1_x_grad_phi, u1_y_shift.expand(1, p_u1, N)), dim=0)
    u1_grad_y = torch.cat((u1_x_phi, u1_y_grad_phi), dim=0)
    u1_grad_xy = torch.cat((u1_x_grad_phi, u1_y_grad_phi), dim=0)

    # \nabla_y A \nabla_x u1
    part1 = torch.einsum('dpn,dqn->dpqn', A_partial_y, u1_grad_x).flatten(start_dim=1, end_dim=2) 
    part1_shift = torch.einsum('dpn,dqn->dpqn', A_partial_y, u1_grad_x_shift).flatten(start_dim=1, end_dim=2) 
    alpha_part1 = torch.outer(alpha_A, alpha_u1).view(-1)

    # \nabla_x A \nabla_y u1
    part2 = torch.einsum('dpn,dqn->dpqn', A_partial_x, u1_grad_y).flatten(start_dim=1, end_dim=2)
    alpha_part2 = alpha_part1

    # 2 A \nabla_x \nabla_y u1
    part3 = torch.einsum('dpn,dqn->dpqn', A, u1_grad_xy).flatten(start_dim=1, end_dim=2) 
    alpha_part3 = 2 * alpha_part1

    # \nabla_x A \nabla_x u0
    part4 = torch.einsum('dpn,dqn->dpqn', A_partial_x, expand_u0_grad_x).flatten(start_dim=1, end_dim=2)
    alpha_part4 = torch.outer(alpha_A, alpha_u0).view(-1)
    
    # A \nabla_x \nabla_x u0
    part5 = torch.einsum('dpn,dqn->dpqn', A, expand_u0_grad2_x).flatten(start_dim=1, end_dim=2)
    alpha_part5 = alpha_part4
    
    rhs_term = torch.cat((expand_F, part1, part1_shift, part2, part3, part4, part5), dim=1)
    alpha_rhs_term = torch.cat((alpha_F, alpha_part1, -alpha_part1, alpha_part2, alpha_part3, alpha_part4, alpha_part5))

    norm_square_rhs_term = Int2TNN(w12, alpha_rhs_term, rhs_term, alpha_rhs_term, rhs_term)

def criterion_u2(model_u2_x, model_u2_y):
    u2_phi_x = model_u2_x(w1, x1, need_grad=0)
    _, u2_grad_phi_y, u2_grad_grad_phi_y = model_u2_y(w2, x2, need_grad=2)

    u2_partial_y = torch.cat((u2_phi_x, u2_grad_phi_y), dim=0)
    u2_partial2_y = torch.cat((u2_phi_x, u2_grad_grad_phi_y), dim=0)

    operand_A_partial_y_u2_partial_y = torch.einsum('dpn, dqn->dpqn', A_partial_y, u2_partial_y)
    operand_A_u2_partial2_y = torch.einsum('dpn, dqn->dpqn', A, u2_partial2_y)

    my_operand_phi = torch.cat((operand_A_partial_y_u2_partial_y, operand_A_u2_partial2_y), dim=1)
    my_operand_alpha = torch.cat((alpha_A_partial_y, alpha_A))

    # p * p matrix
    my_matrix = torch.sum(
        torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(
            torch.einsum('dn, dpqn, djkn->dpjqk', w12, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    # p vector
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_rhs_term).unsqueeze(dim=-1) * torch.prod(
        torch.einsum('dn, dpqn, dkn->dpkq', w12, my_operand_phi, rhs_term), dim=0), dim=(0, 1))

    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_u2_x.scaling_par().data.copy_(my_alpha)

    # my_alpha = model_u2_x.scaling_par()

    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_square_rhs_term)

    return loss


def train_u2(model_u2_x, model_u2_y, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_u2(model_u2_x, model_u2_y)
            if e == 0:
                print('\n Step 4 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                print('lr: ', scheduler.get_last_lr()[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_u2(model_u2_x, model_u2_y)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])
                    torch.save(model_u2_x, "./model_u2/{}_final_u2_x.pkl".format(name))
                    torch.save(model_u2_y, "./model_u2/{}_final_u2_y.pkl".format(name))


        

    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_u2(model_u2_x, model_u2_y)
            optimizer.zero_grad()
            loss.backward()
            return loss

        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_u2(model_u2_x, model_u2_y)
                    print('\n Step 1 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_u2(model_u2_x, model_u2_y)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())

            


if __name__ == '__main__':
    # sizes, sizes_periodic = [1, 50, 50, 50, p_u2], [1, 2, 50, 50, p_u2]#0.01905084503905593
    # sizes, sizes_periodic = [1, 80, 80, 80, p_u2], [1, 5, 80, 80, p_u2]#4.5710206994587365e-05
    # sizes, sizes_periodic = [1, 100, 100, 100, p_u2], [1, 8, 100, 100, p_u2]#6.10674211414302e-05
    # sizes, sizes_periodic = [1, 50, 50, 50, 50, p_u2], [1, 2, 50, 50, 50, p_u2]#0.01905088830868542
    # sizes, sizes_periodic = [1, 80, 80, 80, 80, p_u2], [1, 5, 80, 80, 80, p_u2]#9.631903581678543e-06
    sizes, sizes_periodic = [1, 100, 100, 100, 100, p_u2], [1, 8, 100, 100, 100, p_u2]#7.504251259580171e-06

    activation = TNN_Sin

    if not os.path.exists("model_u2"):
        os.mkdir("model_u2")

    # model_u2_x = torch.load("./model_u2/d1_ex1_final_u2_x.pkl", map_location=device)
    # model_u2_y = torch.load("./model_u2/d1_ex1_final_u2_y.pkl", map_location=device)
    # loss = criterion_u2(model_u2_x, model_u2_y)
    # print('loss: ', loss.item())
 
    model_u2_x = TNN(1, sizes, activation, scaling=True).to(dtype).to(device)
    model_u2_y = TNN_periodic(1, sizes_periodic, activation, period=1, scaling=False).to(dtype).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, itertools.chain(model_u2_x.parameters(), model_u2_y.parameters())),
        lr=1e-2)

    epochs = 50000
    begin_time = time.time()
    train_u2(model_u2_x, model_u2_y, optimizer, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_u2_x, "./model_u2/{}_final_u2_x.pkl".format(name))
    torch.save(model_u2_y, "./model_u2/{}_final_u2_y.pkl".format(name))

    
    epochs = 5000
    save = True 
    # optimizer used
    begin_time = time.time()
    train_u2(model_u2_x, model_u2_y, optimizer, 'LBFGS', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_u2_x, "./model_u2/{}_final_u2_x.pkl".format(name))
    torch.save(model_u2_y, "./model_u2/{}_final_u2_y.pkl".format(name))

    
