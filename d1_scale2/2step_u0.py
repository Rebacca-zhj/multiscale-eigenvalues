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

pi = 3.14159265358979323846
dtype = torch.double
device = 'cuda:6'
name = 'd1_ex1'

p_A = 2
p_chi = 20
p = 20
p_A0 = p_A + p_A * p_chi

a, b = 0, 1
x1, w1 = composite_quadrature_1d(4, 0, 1, 200, device=device)
x2, w2 = composite_quadrature_1d(4, 0, 1, 200, device=device)
w12 = torch.stack((w1, w2), dim=0)
N = len(x1)

dummy = torch.ones((1, 1), dtype=dtype, device=device)

F = x1.view(1, 1, -1)
alpha_F = 3 * torch.ones(1, dtype=dtype, device=device)

# coefficient A = 0.5 sin(2 pi y) + sin(x) + 2
A = torch.ones((2, 2, N), dtype=dtype, device=device)
A[0, 0, :] = 2 + x1
A[1, 1, :] = torch.sin(2 * pi * x2)
alpha_A = torch.ones(2, dtype=dtype, device=device)

A_grad = torch.zeros((2, 2, N), dtype=dtype, device=device)
A_grad[0, 0, :] = 1
A_grad[1, 1, :] = 2 * pi * torch.cos(2 * pi * x2)


model_chi_x = torch.load('./model_chi/d1_ex1_final_chi_x.pkl', map_location=device)
model_chi_y = torch.load('./model_chi/d1_ex1_final_chi_y.pkl', map_location=device)

with torch.no_grad():
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


def criterion_u0(model_u0):
    u0_phi, u0_grad_phi = model_u0(w1, x1, need_grad=1)
    # alpha = model_u0.scaling_par()

    my_matrix = torch.sum(alpha_A0 * torch.einsum('n, pn, qn, kn->pqk', w1, u0_grad_phi.squeeze(dim=0), u0_grad_phi.squeeze(dim=0), A0.squeeze(dim=0)), dim=-1)
    my_vector = torch.sum(alpha_F * torch.einsum('n, pn, qn->pq', w1, u0_phi.squeeze(dim=0), F.squeeze(dim=0)), dim=-1)

    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_u0.scaling_par().data.copy_(my_alpha)

    loss = -0.5 * torch.sum(my_alpha * my_vector)

    return loss



def train_u0(model_u0, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_u0(model_u0)
            if e == 0:
                print('\n Step 2 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                print('lr: ', scheduler.get_last_lr()[0])
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_u0(model_u0)
                    print('\n [ epoch %d ]' % (e+1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])

        # if not os.path.exists("model_u0"):
        #     os.mkdir("model_u0")
        # torch.save(model_u0, "model_u0/{}_Adam_u0.pkl".format(name))
                
    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_u0(model_u0)
            optimizer.zero_grad()
            loss.backward()
            return loss
        
        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_u0(model_u0)
                    print('\n Step 3 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_u0(model_u0)
                    print('\n [ epoch %d ]' % (e+1))
                    print('loss: ', loss.item())    

            # if (e + 1) == 1000:
            #     torch.save(model_u0, "model_u0/{}_LBFGS_1000_u0.pkl".format(name))
        
        # torch.save(model_u0, "model_u0/{}_final_u0.pkl".format(name))

def bd_u0(x):
    return (x - a) * (b - x)

def grad_bd_u0(x):
    return a + b - 2 * x

def grad_grad_bd_u0(x):
    return -2 * torch.ones_like(x)

def grad_grad_grad_bd_u0(x):
    # return -pi * pi * pi * torch.cos(pi * x)
    return 0 * torch.ones_like(x)

if __name__ == '__main__':
    sizes = [1, 50, 50, 50, p]
    activation = TNN_Sin
    if not os.path.exists("model_u0"):
        os.mkdir("model_u0")

    model_u0 = TNN(1, sizes, activation, bd=bd_u0, grad_bd=grad_bd_u0, grad_grad_bd=grad_grad_bd_u0, grad_grad_grad_bd=grad_grad_grad_bd_u0, scaling=True).to(dtype).to(device)

    optimizer = optim.Adam(filter(lambda p:p.requires_grad, itertools.chain(model_u0.parameters())), lr=1e-2)

    epochs = 20000
    begin_time = time.time()
    train_u0(model_u0, optimizer, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))
    
    torch.save(model_u0, "model_u0/{}_final_u0.pkl".format(name))
    
    # optimizer = optim.LBFGS(filter(lambda p:p.requires_grad, itertools.chain(model_u0.parameters())), lr=0.01, \
    #                         tolerance_grad=0, tolerance_change=0)

    # epochs = 2000
    # train_u0(model_u0, optimizer, 'LBFGS', epochs)

    # model_u0 = torch.load('./model_u0/0224_final_u0.pkl', map_location=device)


