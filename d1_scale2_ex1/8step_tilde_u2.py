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

dim=1

a, b = 0, 1
p_A = 2
p_chi = 20
p_u0 = 20
p_u1 = 20
p_u2 = 20
p_delta_u2 = 20
p_u3 = 20
p_tilde_u2 = 20
x1, w1 = composite_quadrature_1d(4, 0, 1, 200, device=device)
x2, w2 = composite_quadrature_1d(4, 0, 1, 200, device=device)

dummy = torch.ones((1, 1), dtype=dtype, device=device)

x_error_0 = torch.zeros(1, dtype=dtype, device=device)

N = len(x1)

A = torch.ones((2, 2, N), dtype=dtype, device=device)
A[0, 0, :] = 2 + x1
A[1, 1, :] = torch.sin(2 * pi * x2)
alpha_A = torch.ones(2, dtype=dtype, device=device)

A_grad_x = torch.ones((2, 2, N), dtype=dtype, device=device)
A_grad_x[0, 0, :] = 1
A_grad_x[1, 1, :] = 0
alpha_A_grad_x = torch.ones(2, dtype=dtype, device=device)


def bd_u0(x):
    return (x - a) * (b - x)

def grad_bd_u0(x):
    return a + b - 2 * x

def grad_grad_bd_u0(x):
    return -2 * torch.ones_like(x)

def grad_grad_grad_bd_u0(x):
    return 0 * torch.ones_like(x)

model_chi_x = torch.load('./model_chi/d1_ex1_final_chi_x.pkl', map_location=device)
model_chi_y = torch.load('./model_chi/d1_ex1_final_chi_y.pkl', map_location=device)
model_u0 = torch.load('./model_u0/d1_ex1_final_u0.pkl', map_location=device)
model_u2_x = torch.load('./model_u2/d1_ex1_final_u2_x.pkl', map_location=device)
model_u2_y = torch.load('./model_u2/d1_ex1_final_u2_y.pkl', map_location=device)
model_delta_u2_x = torch.load('./model_delta_u2/d1_ex1_final_delta_u2_x.pkl', map_location=device)
model_delta_u2_y = torch.load('./model_delta_u2/d1_ex1_final_delta_u2_y.pkl', map_location=device)
model_u3_x = torch.load('./model_u3/d1_ex1_final_u3_x.pkl', map_location=device)
model_u3_y = torch.load('./model_u3/d1_ex1_final_u3_y.pkl', map_location=device)


with torch.no_grad():
    # 计算均匀化系数 A_0^*
    chi = torch.ones((2, p_chi, N), dtype=dtype, device=device)
    chi_grad_phi = torch.zeros((2, p_chi, N), dtype=dtype, device=device)
    chi[0:1, :, :], chi_grad_phi[0:1, :, :] = model_chi_x(w1, x1, need_grad=1)
    chi[1:2, :, :], chi_grad_phi[1:2, :, :] = model_chi_y(w2, x2, need_grad=1)
    alpha_chi = model_chi_x.scaling_par()

    A0 = torch.ones((1, p_A + p_A * p_chi, N), dtype=dtype, device=device)
    A0[0:1, 0:p_A, :] = A[0:1, :, :]
    A0[0:1, p_A:p_A + p_A * p_chi, :] = torch.einsum('dpn,dqn->dpqn', A[0:1, :, :], chi[0:1, :, :]).flatten(start_dim=1, end_dim=2)

    alpha_A0 = torch.ones(p_A + p_A * p_chi, dtype=dtype, device=device)
    alpha_A0[0:p_A] = torch.sum(w2 * A[1, :, :], dim=-1).squeeze(dim=0) * alpha_A
    alpha_A0[p_A:p_A + p_A * p_chi] = (torch.einsum('pn,qn->pq', w2 * A[1, :, :], chi_grad_phi[1, :, :]) * torch.outer(alpha_A, alpha_chi)).view(-1)
    
    A0_grad_x = torch.ones((1, p_A + p_A * p_chi, N), dtype=dtype, device=device)
    A0_grad_x[0:1, 0:p_A, :] = A_grad_x[0:1, :, :]
    A0_grad_x[0:1, p_A:p_A + p_A * p_chi, :] = torch.einsum('dpn,dqn->dpqn', A_grad_x[0:1, :, :], chi[0:1, :, :]).flatten(start_dim=1, end_dim=2) \
                                                        + torch.einsum('dpn,dqn->dpqn', A[0:1, :, :], chi_grad_phi[0:1, :, :]).flatten(start_dim=1, end_dim=2)

    u2_phi = torch.ones((2, p_u2, N), dtype=dtype, device=device)
    u2_grad_phi = torch.zeros((2, p_u2, N), dtype=dtype, device=device)
    u2_grad2_phi = torch.zeros((2, p_u2, N), dtype=dtype, device=device)
    u2_phi[0:1, :, :], u2_grad_phi[0:1, :, :], u2_grad2_phi[0:1, :, :] = model_u2_x(w1, x1, need_grad=2)
    u2_phi[1:2, :, :], u2_grad_phi[1:2, :, :], u2_grad2_phi[1:2, :, :] = model_u2_y(w2, x2, need_grad=2)
    alpha_u2 = model_u2_x.scaling_par()

    u2_y_norm_phi = model_u2_y(w2, x2, normed=False)
    u2_y_norm = torch.sqrt(torch.sum(w2 * u2_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    u2_y_shift = model_u2_y(dummy, x_error_0, need_grad=0, normed=False) #[d,p,1]
    u2_y_shift = (u2_y_shift / u2_y_norm).squeeze(dim=0)

    delta_u2_phi = torch.ones((2, p_delta_u2, N), dtype=dtype, device=device)
    delta_u2_grad_phi = torch.zeros((2, p_delta_u2, N), dtype=dtype, device=device)
    delta_u2_grad2_phi = torch.zeros((2, p_delta_u2, N), dtype=dtype, device=device)
    delta_u2_phi[0:1, :, :], delta_u2_grad_phi[0:1, :, :], delta_u2_grad2_phi[0:1,:,:] = model_delta_u2_x(w1, x1, need_grad=2)
    delta_u2_phi[1:2, :, :], delta_u2_grad_phi[1:2, :, :], delta_u2_grad2_phi[1:2,:,:] = model_delta_u2_y(w2, x2, need_grad=2)
    alpha_delta_u2 = model_delta_u2_x.scaling_par()

    delta_u2_y_norm_phi = model_delta_u2_y(w2, x2, normed=False)
    delta_u2_y_norm = torch.sqrt(torch.sum(w2 * delta_u2_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    delta_u2_y_shift = model_delta_u2_y(dummy, x_error_0, need_grad=0, normed=False) #[d,p,1]
    delta_u2_y_shift = (delta_u2_y_shift / delta_u2_y_norm).squeeze(dim=0)

    u3_phi = torch.ones((2, p_u3, N), dtype=dtype, device=device)
    u3_grad_phi = torch.zeros((2, p_u3, N), dtype=dtype, device=device)
    u3_phi[0:1, :, :], u3_grad_phi[0:1, :, :] = model_u3_x(w1, x1, need_grad=1)
    u3_phi[1:2, :, :], u3_grad_phi[1:2, :, :] = model_u3_y(w2, x2, need_grad=1)
    alpha_u3 = model_u3_x.scaling_par()
    
    # <A \nabla_x (\hat u2 + \delta u2)>
    A_u2_xx = torch.einsum('dpn,dqn->dpqn', A[0:1, :, :], torch.cat((u2_grad2_phi[0:1, :, :], delta_u2_grad2_phi[0:1, :, :]), dim=1)).flatten(start_dim=1, end_dim=2)
    A_x_u2_x = torch.einsum('dpn,dqn->dpqn', A_grad_x[0:1, :, :], torch.cat((u2_grad_phi[0:1, :, :], delta_u2_grad_phi[0:1, :, :]), dim=1)).flatten(start_dim=1, end_dim=2)
    int_y_A_u2_x = torch.einsum('pn,qn,n->pq', A[1, :, :], torch.cat((u2_phi[1, :, :] - u2_y_shift, delta_u2_phi[1, :, :] - delta_u2_y_shift), dim=0), w2).view(-1)
    alpha_A_u2 = torch.outer(alpha_A, torch.cat((alpha_u2, alpha_delta_u2), dim=0)).view(-1) * int_y_A_u2_x
    
    # <A \nabla_y u3>
    A_u3_x = torch.einsum('dpn,dqn->dpqn', A[0:1, :, :], u3_grad_phi[0:1, :, :]).flatten(start_dim=1, end_dim=2)
    A_x_u3 = torch.einsum('dpn,dqn->dpqn', A_grad_x[0:1, :, :], u3_phi[0:1, :, :]).flatten(start_dim=1, end_dim=2)
    int_y_A_u3_y = torch.einsum('pn,qn,n->pq', A[1, :, :], u3_grad_phi[1, :, :], w2).view(-1)
    alpha_A_u3 = torch.outer(alpha_A, alpha_u3).view(-1) * int_y_A_u3_y
    
    rhs = torch.cat([A_u2_xx, A_x_u2_x, A_u3_x, A_x_u3], dim=1)
    alpha_rhs = torch.cat([alpha_A_u2, alpha_A_u2, alpha_A_u3, alpha_A_u3])

    norm_square_rhs = Int2TNN(w1, alpha_rhs, rhs, alpha_rhs, rhs)


def criterion_tilde_u2(model_tilde_u2):
    # 强形式
    _, tilde_u2_grad_phi, tilde_u2_grad_grad_phi = model_tilde_u2(w1, x1, need_grad=2)

    operand_A0_grad_x_tilde_u2_grad_x = torch.einsum('dpn, dqn->dpqn', A0_grad_x, tilde_u2_grad_phi)
    operand_A0_tilde_u2_grad2_x = torch.einsum('dpn, dqn->dpqn', A0, tilde_u2_grad_grad_phi)

    my_operand_phi = torch.cat((operand_A0_grad_x_tilde_u2_grad_x, operand_A0_tilde_u2_grad2_x), dim=1)
    my_operand_alpha = torch.cat((alpha_A0,alpha_A0))

    my_matrix = torch.sum(
        torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(
            torch.einsum('n, dpqn, djkn->dpjqk', w1, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))

    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_rhs).unsqueeze(dim=-1) * torch.prod(
        torch.einsum('n, dpqn, dkn->dpkq', w1, my_operand_phi, rhs), dim=0), dim=(0, 1))

    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_tilde_u2.scaling_par().data.copy_(my_alpha)

    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_square_rhs)
    return loss


def train_tilde_u2(model_tilde_u2, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_tilde_u2(model_tilde_u2)
            if e == 0:
                print('\n Step 8 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                print('lr: ', scheduler.get_last_lr()[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_tilde_u2(model_tilde_u2)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])

    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_tilde_u2(model_tilde_u2)
            optimizer.zero_grad()
            loss.backward()
            return loss

        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_tilde_u2(model_tilde_u2)
                    print('\n Step 1 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_tilde_u2(model_tilde_u2)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())

if __name__ == '__main__':

    # sizes = [1, 50, 50, 50, 50, p_tilde_u2]#0.0388524086490525
    # sizes = [1, 80, 80, 80, 80, p_tilde_u2]#0.0001495609668835317
    sizes = [1, 100, 100, 100, 100, p_tilde_u2]#
    # sizes = [1, 50, 50, 50, p_tilde_u2]#
    # sizes = [1, 80, 80, 80, p_tilde_u2]#
    # sizes = [1, 100, 100, 100, p_tilde_u2]#
    activation = TNN_Sin

    if not os.path.exists("model_tilde_u2"):
        os.mkdir("model_tilde_u2")    

    # model_tilde_u2 = torch.load("./model_tilde_u2/d1_ex1_test3_final_tilde_u2.pkl", map_location=device)
    # loss = criterion_tilde_u2(model_tilde_u2)
    # print('loss: ', loss.item())
 
    model_tilde_u2 = TNN(1, sizes, activation, bd=bd_u0, grad_bd=grad_bd_u0, grad_grad_bd=grad_grad_bd_u0, \
        grad_grad_grad_bd=grad_grad_grad_bd_u0, scaling=True).to(dtype).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, itertools.chain(model_tilde_u2.parameters())), lr=1e-2)

    epochs = 40000
    begin_time = time.time()
    train_tilde_u2(model_tilde_u2, optimizer, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_tilde_u2, "./model_tilde_u2/{}_final_tilde_u2.pkl".format(name))

    
    epochs = 2000
    save = True 
    # optimizer used
    begin_time = time.time()
    train_tilde_u2(model_tilde_u2, optimizer, 'LBFGS', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_tilde_u2, "./model_tilde_u2/{}_final_tilde_u2.pkl".format(name))

