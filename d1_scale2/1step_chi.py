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
name = 'd1_ex1_p20'

p = 20

a, b = 0, 1

x1, w1 = composite_quadrature_1d(4, 0, 1, 200, device=device)
N = len(x1)
x2, w2 = composite_quadrature_1d(4, 0, 1, 200, device=device)

w12 = torch.stack((w1, w2), dim=0)

# coefficient A = 0.5 sin(2 pi y) + sin(x) + 2
A = torch.ones((2, 2, N), dtype=dtype, device=device)
A[0, 0, :] = 2 + x1
A[1, 1, :] = torch.sin(2 * pi * x2)
alpha_A = torch.ones(2, dtype=dtype, device=device)

A_grad = torch.zeros((2, 2, N), dtype=dtype, device=device)
A_grad[0, 0, :] = 1
A_grad[1, 1, :] = 2 * pi * torch.cos(2 * pi * x2)

A_partial_y = torch.ones((2, 1, N), dtype=dtype, device=device)
A_partial_y[1, 0, :] = 2 * pi * torch.cos(2 * pi * x2)

alpha_A_partial_y = torch.ones(1, dtype=dtype, device=device)

norm_square_A_partial_y = Int2TNN(w12, alpha_A_partial_y, A_partial_y, alpha_A_partial_y, A_partial_y)


# 在 n1 * n2 的积分点上 A 逐点的值
A_pointwise_2dim = torch.sum(alpha_A.view(-1, 1, 1) * torch.einsum('pm,pn->pmn', A[0,:,:], A[1,:,:]), dim=0)

A0_exact_pointwise = 1 / torch.sum(w2 / A_pointwise_2dim, dim=-1)
norm_A0_exact = torch.sqrt(torch.sum(w1 * A0_exact_pointwise**2))

del A_pointwise_2dim


# \int\int \left[  \frac{\partial A}{\partial y} 
#         + \frac{\partial A}{\partial y}\frac{\partial\chi}{\partial y} 
#         + A\frac{\partial^2 \chi}{\partial y^2}   \right]^2  dxdy
def criterion_1(model_chi_x, model_chi_y):
    chi_phi_x = model_chi_x(w1, x1, need_grad=0)
    chi_phi_y, chi_grad_phi_y, chi_grad_grad_phi_y = model_chi_y(w2, x2, need_grad=2)

    chi_partial_y = torch.cat((chi_phi_x, chi_grad_phi_y), dim=0)
    chi_partial2_y = torch.cat((chi_phi_x, chi_grad_grad_phi_y), dim=0)

    operand_A_partial_y_chi_partial_y = torch.einsum('dpn, dqn->dpqn', A_partial_y, chi_partial_y)
    operand_A_chi_partial2_y = torch.einsum('dpn, dqn->dpqn', A, chi_partial2_y)

    my_operand_phi = torch.cat((operand_A_partial_y_chi_partial_y, operand_A_chi_partial2_y), dim=1)
    my_operand_alpha = torch.cat((alpha_A_partial_y, alpha_A))
    # p * p matrix
    my_matrix = torch.sum(torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(torch.einsum('dn, dpqn, djkn->dpjqk', w12, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    # p vector
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_A_partial_y).unsqueeze(dim=-1) * torch.prod(torch.einsum('dn, dpqn, dkn->dpkq', w12, my_operand_phi, A_partial_y), dim=0), dim=(0, 1))

    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_chi_x.scaling_par().data.copy_(my_alpha)

    # my_alpha = model_chi_x.scaling_par()

    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_square_A_partial_y)
    # loss = torch.sqrt(torch.sum(torch.outer(my_alpha, my_alpha) * my_matrix) - 2 * torch.sum(my_alpha * my_vector) + norm_square_A1_partial_y)

    return loss

def compute_homogenized_matrix_0_error(model_chi_x, model_chi_y):
    with torch.no_grad():
        chi_partial_y = torch.ones((2, p, N), dtype=dtype, device=device)
        chi_partial_y[0:1, :, :] = model_chi_x(w1, x1, need_grad=0)
        _, chi_partial_y[1:2, :, :] = model_chi_y(w2, x2, need_grad=1)
        alpha_chi_partial_y = model_chi_x.scaling_par()

        # a^{*}=\int_{Y}\left(a + a \frac{\partial\chi}{\partial y}\right) dy
        A0_TNN_phi_1 = A[0:1, :, :]
        A0_TNN_alpha_1 = torch.sum(w2 * A[1, :, :], dim=-1) * alpha_A

        A0_TNN_phi_2 = torch.einsum('dpn,dqn->dpqn', A[0:1, :, :], chi_partial_y[0:1, :, :]).flatten(start_dim=1, end_dim=2)
        A0_TNN_alpha_2 = (torch.einsum('pn,qn->pq', w2 * A[1, :, :], chi_partial_y[1, :, :]) * torch.outer(alpha_A, alpha_chi_partial_y)).view(-1)
        
        A0_pointwise = torch.sum(A0_TNN_alpha_1.view(-1, 1) * A0_TNN_phi_1[0, :, :], dim=0) \
                        + torch.sum(A0_TNN_alpha_2.view(-1, 1) * A0_TNN_phi_2[0, :, :], dim=0)
        
    print('relative error of homogenized matrix A0 : ', (torch.sqrt(torch.sum(w1 * (A0_pointwise - A0_exact_pointwise)**2)) / norm_A0_exact).item())

def train_1(model_chi_x, model_chi_y, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_1(model_chi_x, model_chi_y)
            if e == 0:
                print('\n Step 1 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                compute_homogenized_matrix_0_error(model_chi_x, model_chi_y)
                print('lr: ', scheduler.get_last_lr()[0])
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_1(model_chi_x, model_chi_y)
                    print('\n [ epoch %d ]' % (e+1))
                    print('loss: ', loss.item())
                    compute_homogenized_matrix_0_error(model_chi_x, model_chi_y)
                    print('lr: ', scheduler.get_last_lr()[0])

        # if not os.path.exists("model_chi"):
        #     os.mkdir("model_chi")
        # torch.save(model_chi_x, "model_chi/{}_Adam_chi_x.pkl".format(name))
        # torch.save(model_chi_y, "model_chi/{}_Adam_chi_y.pkl".format(name))
                
    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_1(model_chi_x, model_chi_y)
            optimizer.zero_grad()
            loss.backward()
            return loss
        
        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_1(model_chi_x, model_chi_y)
                    print('\n Step 1 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())
                    compute_homogenized_matrix_0_error(model_chi_x, model_chi_y)

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_1(model_chi_x, model_chi_y)
                    print('\n [ epoch %d ]' % (e+1))
                    print('loss: ', loss.item())
                    compute_homogenized_matrix_0_error(model_chi_x, model_chi_y)

            # if (e + 1) == 1000:
            #     torch.save(model_chi_x, "model_chi/{}_LBFGS_1000_chi_x.pkl".format(name))
            #     torch.save(model_chi_y, "model_chi/{}_LBFGS_1000_chi_y.pkl".format(name))
        
        # torch.save(model_chi_x, "model_chi/{}_final_chi_x.pkl".format(name))
        # torch.save(model_chi_y, "model_chi/{}_final_chi_y.pkl".format(name))
    return loss



if __name__ == '__main__':
    sizes = [1, 50, 50, 50, p]
    sizes_periodic = [1, 2, 50, 50, p]
    activation = TNN_Sin

    if not os.path.exists("model_chi"):
        os.mkdir("model_chi")
    
    model_chi_x = TNN(1, sizes, activation, scaling=True).to(dtype).to(device)
    model_chi_y = TNN_periodic(1, sizes_periodic, activation, period=1, scaling=False).to(dtype).to(device)

    compute_homogenized_matrix_0_error(model_chi_x, model_chi_y)

    optimizer = optim.Adam(filter(lambda p:p.requires_grad, itertools.chain(model_chi_x.parameters(), model_chi_y.parameters())), lr=1e-2)

    epochs = 50000
    begin_time = time.time()
    train_1(model_chi_x, model_chi_y, optimizer, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    if not os.path.exists("model_chi"):
        os.mkdir("model_chi")
    torch.save(model_chi_x, "model_chi/{}_final_chi_x.pkl".format(name))
    torch.save(model_chi_y, "model_chi/{}_final_chi_y.pkl".format(name))

    
    optimizer = optim.LBFGS(filter(lambda p:p.requires_grad, itertools.chain(model_chi_x.parameters(), model_chi_y.parameters())), lr=0.01, \
                            tolerance_grad=0, tolerance_change=0)

    epochs = 2000
    train_1(model_chi_x, model_chi_y, optimizer, 'LBFGS', epochs)




