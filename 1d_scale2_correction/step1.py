import torch
import math
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
# device = 'cuda:2'
device = 'cuda:5'
name = 'x_free_y_per'

p = 20
p_A = 2

a, b = 0, 1

x1, w1 = composite_quadrature_1d(8, 0, 1, 200, device=device)
N = len(x1)
x2, w2 = composite_quadrature_1d(8, 0, 1, 200, device=device)

w12 = torch.stack((w1, w2), dim=0)

dummy = torch.ones((1, 1), dtype=dtype, device=device)
x_error_0 = torch.zeros(1, dtype=dtype, device=device)

# coefficient A = 2 + x1 + sin(2*pi*y)
A = torch.ones((2, 2, N), dtype=dtype, device=device)
A[0, 0, :] = 2 + x1
A[1, 1, :] = torch.sin(2*pi * x2)
alpha_A = torch.ones(2, dtype=dtype, device=device)

A_partial_y1 = torch.ones((2, 1, N), dtype=dtype, device=device)
A_partial_y1[1, 0, :] = 2 * pi * torch.cos(2 * pi * x2)
alpha_A_partial_y1 = torch.ones(1, dtype=dtype, device=device)

# # coefficient A = (1 + x1)*(1 + 0.5sin(2*pi*y))
# p_A = 1
# A = torch.ones((2, 1, N), dtype=dtype, device=device)
# A[0, 0, :] = 1 + x1
# A[1, 0, :] = 1 + 0.5 * torch.sin(2*pi * x2)
# alpha_A = torch.ones(1, dtype=dtype, device=device)

# A_partial_y1 = torch.ones((2, 1, N), dtype=dtype, device=device)
# A_partial_y1[0, 0, :] = 1 + x1
# A_partial_y1[1, 0, :] = pi * torch.cos(2 * pi * x2)
# alpha_A_partial_y1 = torch.ones(1, dtype=dtype, device=device)

norm_square_A_partial_y1 = Int2TNN(w12, alpha_A_partial_y1, A_partial_y1, alpha_A_partial_y1, A_partial_y1)

# 在 n1 * n2 的积分点上 A 逐点的值
# A_pointwise_2dim = torch.sum(alpha_A.view(-1, 1, 1) * torch.einsum('pm,pn->pmn', A[0, :, :], A[1, :, :]), dim=0)

# A0_exact_pointwise = 1 / torch.sum(w2 / A_pointwise_2dim, dim=-1)

A0_exact_pointwise = torch.sqrt((x1 + 1) * (x1 + 3))
alpha_A0_exact=torch.ones(1, dtype=dtype, device=device)


# A0_exact_pointwise = np.sqrt(3)/2*(x1 + 1)
# alpha_A0_exact=torch.ones(1, dtype=dtype, device=device)

norm_A0_exact = torch.sqrt(torch.sum(w1 * A0_exact_pointwise ** 2))


# \int\int \left[  \frac{\partial A}{\partial y1}
#         + \frac{\partial A}{\partial y1}\frac{\partial\chi1}{\partial y1}
#         + A\frac{\partial^2 \chi1}{\partial y1^2}   \right]^2  dxdy1
def criterion_1(model_chi1_x, model_chi1_y1):
    chi1_phi_x = model_chi1_x(w1, x1, need_grad=0)
    chi1_phi_y1, chi1_grad_phi_y1, chi1_grad_grad_phi_y1 = model_chi1_y1(w2, x2, need_grad=2)

    chi1_partial_y1 = torch.cat((chi1_phi_x, chi1_grad_phi_y1), dim=0)
    chi1_partial2_y1 = torch.cat((chi1_phi_x, chi1_grad_grad_phi_y1), dim=0)

    operand_A_partial_y1_chi1_partial_y1 = torch.einsum('dpn, dqn->dpqn', A_partial_y1, chi1_partial_y1)
    operand_A_chi1_partial2_y1 = torch.einsum('dpn, dqn->dpqn', A, chi1_partial2_y1)

    my_operand_phi = torch.cat((operand_A_partial_y1_chi1_partial_y1, operand_A_chi1_partial2_y1), dim=1)
    my_operand_alpha = torch.cat((alpha_A_partial_y1, alpha_A))

    # p * p matrix
    my_matrix = torch.sum(
        torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(
            torch.einsum('dn, dpqn, djkn->dpjqk', w12, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    # p vector
          
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_A_partial_y1).unsqueeze(dim=-1) * torch.prod(
        torch.einsum('dn, dpqn, dkn->dpkq', w12, my_operand_phi, A_partial_y1), dim=0), dim=(0, 1))

    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_chi1_x.scaling_par().data.copy_(my_alpha)

    # my_alpha = model_chi1_x.scaling_par()

    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_square_A_partial_y1)
    # loss = torch.sqrt(torch.sum(torch.outer(my_alpha, my_alpha) * my_matrix) - 2 * torch.sum(my_alpha * my_vector) + norm_square_A1_partial_y1)

    return loss


def compute_homogenized_matrix_0_error(model_chi1_x, model_chi1_y1):
    with torch.no_grad():
        chi1_partial_y1 = torch.ones((2, p, N), dtype=dtype, device=device)
        chi1_partial_y1[0:1, :, :] = model_chi1_x(w1, x1, need_grad=0)
        _, chi1_partial_y1[1:2, :, :] = model_chi1_y1(w2, x2, need_grad=1)
        alpha_chi1_partial_y1 = model_chi1_x.scaling_par()

        # a^{*}=\int_{Y}\left(a + a \frac{\partial\chi1}{\partial y1}\right) dy1
        A0_TNN_phi_1 = A[0:1, :, :]
        A0_TNN_alpha_1 = torch.sum(w2 * A[1, :, :], dim=-1) * alpha_A

        A0_TNN_phi_2 = torch.einsum('dpn,dqn->dpqn', A[0:1, :, :], chi1_partial_y1[0:1, :, :]).flatten(start_dim=1, end_dim=2)
        A0_TNN_alpha_2 = (torch.einsum('pn,qn->pq', w2 * A[1, :, :], chi1_partial_y1[1, :, :]) * torch.outer(alpha_A, alpha_chi1_partial_y1)).view(-1)

        A0_pointwise = torch.sum(A0_TNN_alpha_1.view(-1, 1) * A0_TNN_phi_1[0, :, :], dim=0) \
                       + torch.sum(A0_TNN_alpha_2.view(-1, 1) * A0_TNN_phi_2[0, :, :], dim=0)
    print('relative error of homogenized matrix A0 : ',
          (torch.sqrt(torch.sum(w1 * (A0_pointwise - A0_exact_pointwise) ** 2)) / norm_A0_exact).item())

    # chi1 = torch.ones((2, p, N), dtype=dtype, device=device)
    # chi1_grad_phi = torch.zeros((2, p, N), dtype=dtype, device=device)
    # chi1[0:1, :, :], chi1_grad_phi[0:1, :, :] = model_chi1_x(w1, x1, need_grad=1)
    # chi1[1:2, :, :], chi1_grad_phi[1:2, :, :] = model_chi1_y1(w2, x2, need_grad=1)
    # alpha_chi1 = model_chi1_x.scaling_par()

    # A0 = torch.ones((1, p_A + p_A * p, N), dtype=dtype, device=device)
    # A0[0:1, 0:p_A, :] = A[0:1, :, :]
    # A0[0:1, p_A:p_A + p_A * p, :] = torch.einsum('dpn,dqn->dpqn', A[0:1, :, :], chi1[0:1, :, :]).squeeze(dim=1)

    # alpha_A0 = torch.ones(1 + 1 * p, dtype=dtype, device=device)
    # alpha_A0[0:1] = torch.sum(w2 * A[1, :, :], dim=-1).squeeze(dim=0) * alpha_A
    # alpha_A0[1:1 + 1 * p] = (torch.einsum('pn,qn->pq', w2 * A[1, :, :], chi1_grad_phi[1, :, :]) * torch.outer(alpha_A, alpha_chi1)).view(-1)


def train_1(model_chi1_x, model_chi1_y1, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_1(model_chi1_x, model_chi1_y1)
            if e == 0:
                print('\n Step 1 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                compute_homogenized_matrix_0_error(model_chi1_x, model_chi1_y1)
                print('lr: ', scheduler.get_last_lr()[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_1(model_chi1_x, model_chi1_y1)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    compute_homogenized_matrix_0_error(model_chi1_x, model_chi1_y1)
                    print('lr: ', scheduler.get_last_lr()[0])

        

    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_1(model_chi1_x, model_chi1_y1)
            optimizer.zero_grad()
            loss.backward()
            return loss

        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_1(model_chi1_x, model_chi1_y1)
                    print('\n Step 1 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())
                    compute_homogenized_matrix_0_error(model_chi1_x, model_chi1_y1)

            loss = optimizer.step(closure)

            if (e + 1) % 10 == 0:
                with torch.no_grad():
                    loss = criterion_1(model_chi1_x, model_chi1_y1)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    compute_homogenized_matrix_0_error(model_chi1_x, model_chi1_y1)

            
def bd_u0(x):
    return (x - a) * (b - x)

def grad_bd_u0(x):
    return a + b - 2 * x

def grad_grad_bd_u0(x):
    return -2 * torch.ones_like(x)


if __name__ == '__main__':
    sizes = [1, 50, 50, 50, 50, p]
    sizes_periodic = [1, 5, 50, 50, p]
    activation = TNN_Sin
    # model_chi1_x = torch.load('./model_chi1/x_free_y_per_final_chi1_x.pkl',map_location=device)
    # model_chi1_y1 = torch.load('./model_chi1/x_free_y_per_final_chi1_y1.pkl',map_location=device)


    
    model_chi1_x = TNN(1, sizes, activation, scaling=True).to(dtype).to(device)
    model_chi1_y1 = TNN_periodic(1, sizes_periodic, activation, period=1, scaling=False).to(dtype).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, itertools.chain(model_chi1_x.parameters(), model_chi1_y1.parameters())),
        lr=1e-2)

    epochs = 4000
    begin_time = time.time()
    train_1(model_chi1_x, model_chi1_y1, optimizer, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    # torch.save(model_chi1_x, "./model_chi1/{}_final_chi1_x.pkl".format(name))
    # torch.save(model_chi1_y1, "./model_chi1/{}_final_chi1_y1.pkl".format(name))

    optimizer = optim.LBFGS(filter(lambda p:p.requires_grad, itertools.chain(model_chi1_x.parameters(), model_chi1_y1.parameters())), lr=0.01, \
                            tolerance_grad=0, tolerance_change=0)

    epochs = 2000
    train_1(model_chi1_x, model_chi1_y1, optimizer, 'LBFGS', epochs)

    # torch.save(model_chi1_x, "./model_chi1/{}_final_chi1_x.pkl".format(name))
    # torch.save(model_chi1_y1, "./model_chi1/{}_final_chi1_y1.pkl".format(name))

    eps=5
    x_error, w_error = composite_quadrature_1d(4, 0, 1, 10000, device=device, dtype=dtype)
    y_error_eps = x_error * eps - (x_error * eps).floor()
    chi1_x_norm_phi = model_chi1_x(w1, x1, normed=False)
    chi1_x_norm = torch.sqrt(torch.sum(w1 * chi1_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    chi1_x_phi = model_chi1_x(dummy, x_error, normed=False)
    chi1_x_phi = chi1_x_phi / chi1_x_norm
    chi1_alpha = model_chi1_x.scaling_par()

    chi1_y1_norm_phi = model_chi1_y1(w2, x2, normed=False)
    chi1_y1_norm = torch.sqrt(torch.sum(w2 * chi1_y1_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    
    chi1_y1_phi = model_chi1_y1(dummy, y_error_eps, need_grad=0, normed=False)
    chi1_y1_phi = chi1_y1_phi / chi1_y1_norm
    chi1 = torch.sum((chi1_alpha.unsqueeze(dim=-1) * chi1_x_phi * chi1_y1_phi).squeeze(dim=0), dim=0)
    # INSERT_YOUR_CODE
    # 已经求出来chi1了，画chi的图像（一维）

    chi1_phi_y1_0 = model_chi1_y1(w2, x_error_0, need_grad=0, normed=False)
    chi1_phi_y1_norm = torch.sqrt(torch.sum(w2 * chi1_phi_y1_0**2, dim=-1)).unsqueeze(dim=-1)
    chi1_phi_y1_shift = model_chi1_y1(dummy, x_error_0, need_grad=0, normed=False)
    chi1_phi_y1_shift = chi1_phi_y1_shift / chi1_phi_y1_norm
    chi1_phi_y1_shift_value = torch.sum(chi1_alpha * torch.prod(chi1_phi_y1_shift.squeeze(dim=-1), dim=0))
    print('chi1_phi_y1_shift_value: ', chi1_phi_y1_shift_value)

    import matplotlib.pyplot as plt

    # 取 x_error（[0, 1]上差分点）和对应chi1的数值，画一维曲线图
    x_plot = x_error.detach().cpu().numpy()
    chi1_plot = chi1.detach().cpu().numpy()  # 只画第一个特征对

    plt.figure(figsize=(8, 4))
    plt.plot(x_plot, chi1_plot, label=r'$\chi_1$')
    plt.xlabel('x')
    plt.ylabel(r'$\chi_1(x)$')
    plt.title(r'Plot of $\chi_1(x)$ (eps={})'.format(eps))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'png/chi1_plot_eps{eps}.png', dpi=200)
    plt.show()




