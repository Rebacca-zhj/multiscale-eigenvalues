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
device = 'cuda:5'
name = 'x_free_y_per_shift'

# 求前m个特征值
m = 1
dim=1

a, b = 0, 1

p_A = 2
p_chi1 = 20
p_eta2 = 20
p_A0 = p_A + p_A * p_chi1

x1, w1 = composite_quadrature_1d(8, 0, 1, 200, device=device)
x2, w2 = composite_quadrature_1d(8, 0, 1, 200, device=device)

dummy = torch.ones((1, 1), dtype=dtype, device=device)

x_error_0 = torch.zeros(1, dtype=dtype, device=device)

N = len(x1)
w12 = torch.stack((w1, w2), dim=0)

A = torch.ones((2, 2, N), dtype=dtype, device=device)
A[0, 0, :] = 2 + x1
A[1, 1, :] = torch.sin(2*pi * x2)
alpha_A = torch.ones(2, dtype=dtype, device=device)

A_partial_x = torch.ones((2, 2, N), dtype=dtype, device=device)
A_partial_x[:,1,:] = 0
alpha_A_partial_x = torch.ones(2, dtype=dtype, device=device)

A_partial_y1 = torch.ones((2, 2, N), dtype=dtype, device=device)
A_partial_y1[:,0,:] = 0
A_partial_y1[1,1,:] = 2*pi * torch.cos(2*pi*x2)
alpha_A_partial_y1 = torch.ones(2, dtype=dtype, device=device)

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

    chi1_y1_norm_phi = model_chi1_y1(w2, x2, normed=False)
    chi1_y1_norm = torch.sqrt(torch.sum(w2 * chi1_y1_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    chi1_y1_shift = model_chi1_y1(dummy, x_error_0, need_grad=0, normed=False) #[d,p,1]
    chi1_y1_shift = chi1_y1_shift / chi1_y1_norm

    part1 = torch.ones((2, p_A * p_chi1, N), dtype=dtype, device=device)
    part1[0:1,:,:]= torch.einsum('dpn,dqn->dpqn', A[0:1, :, :], chi1[0:1, :, :]).flatten(start_dim=1, end_dim=2)
    part1[1:2,:,:]= torch.einsum('dpn,dqn->dpqn', A_partial_y1[1:2, :, :], (chi1-chi1_y1_shift)[1:2, :, :]).flatten(start_dim=1, end_dim=2) 
    alpha_part1 = torch.outer(alpha_A, alpha_chi1).view(-1)

    # chi1_y1_integral_term = torch.sum(w2 * chi1_grad_phi[1:2, :, :], dim=-1).squeeze(dim=0)
    # part2 = torch.ones((2, p_A * p_chi1, N), dtype=dtype, device=device)
    # part2[0:1,:,:]= torch.einsum('dpn,dqn->dpqn', A[0:1, :, :], chi1[0:1, :, :]).flatten(start_dim=1, end_dim=2)
    # part2[1:2,:,:]= torch.einsum('dpn,dqn->dpqn', A_partial_y1[1:2, :, :], torch.ones((1,p_chi1,N), dtype=dtype, device=device)).flatten(start_dim=1, end_dim=2)
    # # alpha_part2 = torch.outer(alpha_A, alpha_chi1 * chi1_y1_integral_term).view(-1)
    # alpha_part2 = torch.outer(alpha_A, alpha_chi1 * chi1_y1_shift.squeeze(dim=0).squeeze(dim=-1)).view(-1)

    part2 = torch.ones((2, p_A * p_chi1, N), dtype=dtype, device=device)
    part2[0:1,:,:]= torch.einsum('dpn,dqn->dpqn', A[0:1, :, :], chi1[0:1, :, :]).flatten(start_dim=1, end_dim=2) 
    part2[1:2,:,:]= torch.einsum('dpn,dqn->dpqn', A[1:2, :, :], chi1_grad_phi[1:2, :, :]).flatten(start_dim=1, end_dim=2)
    alpha_part2 = alpha_part1

    part3 = torch.ones((2, p_A + p_A * p_chi1, N), dtype=dtype, device=device)
    part3[0,:,:] = A0
    
    F = torch.cat((part1, part2, part3, A), dim=1)
    alpha_F = torch.cat((alpha_part1, 2*alpha_part2, -alpha_A0, alpha_A_partial_x))

    norm_square_F = Int2TNN(w12, alpha_F, F, alpha_F, F)

def criterion_1(model_eta2_x, model_eta2_y1):
    eta2_phi_x = model_eta2_x(w1, x1, need_grad=0)
    _, eta2_grad_phi_y1, eta2_grad_grad_phi_y1 = model_eta2_y1(w2, x2, need_grad=2)

    eta2_partial_y1 = torch.cat((eta2_phi_x, eta2_grad_phi_y1), dim=0)
    eta2_partial2_y1 = torch.cat((eta2_phi_x, eta2_grad_grad_phi_y1), dim=0)

    operand_A_partial_y1_eta2_partial_y1 = torch.einsum('dpn, dqn->dpqn', A_partial_y1, eta2_partial_y1)
    operand_A_eta2_partial2_y1 = torch.einsum('dpn, dqn->dpqn', A, eta2_partial2_y1)

    my_operand_phi = torch.cat((operand_A_partial_y1_eta2_partial_y1, operand_A_eta2_partial2_y1), dim=1)
    my_operand_alpha = torch.cat((alpha_A_partial_y1, alpha_A))

    # p * p matrix
    my_matrix = torch.sum(
        torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(
            torch.einsum('dn, dpqn, djkn->dpjqk', w12, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    # p vector
          
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_F).unsqueeze(dim=-1) * torch.prod(
        torch.einsum('dn, dpqn, dkn->dpkq', w12, my_operand_phi, F), dim=0), dim=(0, 1))

    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_eta2_x.scaling_par().data.copy_(my_alpha)

    # my_alpha = model_eta2_x.scaling_par()

    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_square_F)
    # loss = torch.sqrt(torch.sum(torch.outer(my_alpha, my_alpha) * my_matrix) - 2 * torch.sum(my_alpha * my_vector) + norm_square_A1_partial_y1)

    return loss


def train_1(model_eta2_x, model_eta2_y1, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_1(model_eta2_x, model_eta2_y1)
            if e == 0:
                print('\n Step 5 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                print('lr: ', scheduler.get_last_lr()[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_1(model_eta2_x, model_eta2_y1)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])

        

    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_1(model_eta2_x, model_eta2_y1)
            optimizer.zero_grad()
            loss.backward()
            return loss

        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_1(model_eta2_x, model_eta2_y1)
                    print('\n Step 1 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 10 == 0:
                with torch.no_grad():
                    loss = criterion_1(model_eta2_x, model_eta2_y1)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())

            


if __name__ == '__main__':
    sizes = [1, 50, 50, 50, 50, p_eta2]
    sizes_periodic = [1, 5, 50, 50, p_eta2]
    activation = TNN_Sin

    # model_eta2_x = torch.load('./model_eta2/ex4_final_eta2_x.pkl',map_location=device)
    # model_eta2_y1 = torch.load('./model_eta2/ex4_final_eta2_y1.pkl',map_location=device)
    model_eta2_x = TNN(1, sizes, activation, scaling=True).to(dtype).to(device)
    model_eta2_y1 = TNN_periodic(1, sizes_periodic, activation, period=1, scaling=False).to(dtype).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, itertools.chain(model_eta2_x.parameters(), model_eta2_y1.parameters())),
        lr=1e-2)

    epochs = 40000
    begin_time = time.time()
    train_1(model_eta2_x, model_eta2_y1, optimizer, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_eta2_x, "./model_eta2/{}_final_eta2_x.pkl".format(name))
    torch.save(model_eta2_y1, "./model_eta2/{}_final_eta2_y1.pkl".format(name))
    
    epochs = 4000
    save = True 
    # optimizer used
    begin_time = time.time()
    train_1(model_eta2_x, model_eta2_y1, optimizer, 'LBFGS', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_eta2_x, "./model_eta2/{}_final_eta2_x.pkl".format(name))
    torch.save(model_eta2_y1, "./model_eta2/{}_final_eta2_y1.pkl".format(name))


