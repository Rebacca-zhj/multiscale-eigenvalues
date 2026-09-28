import torch
torch.set_printoptions(profile="full", precision=16)
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import torch.optim as optim
import importlib.util

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
name = 'd2_ex2_test2'
p = 20

dim = 2
a, b = 0, 1

x1, w1 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
N = len(x1)
x2, w2 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
x3, w3 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
x4, w4 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)

w1234 = torch.stack((w1, w2, w3, w4), dim=0)
w12 = torch.stack((w1, w2), dim=0)

dummy = torch.ones((1, 1), dtype=dtype, device=device)
x_error_0 = torch.zeros(1, dtype=dtype, device=device)

# 从0step_A.py导入A矩阵信息（使用importlib因为模块名不能以数字开头）
spec = importlib.util.spec_from_file_location("step_0_A", os.path.join(os.path.dirname(__file__), "0step_A.py"))
if spec is None or spec.loader is None:
    raise ImportError("Cannot load module 0step_A.py")
step_0_A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step_0_A)
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


norm_square_A_grad_y1 = Int2TNN(w1234, alpha_A_grad_y1, A_grad_y1, alpha_A_grad_y1, A_grad_y1)
print('norm_square_A_grad_y1 =', norm_square_A_grad_y1.item())
norm_square_A_grad_y2 = Int2TNN(w1234, alpha_A_grad_y2, A_grad_y2, alpha_A_grad_y2, A_grad_y2)
print('norm_square_A_grad_y2 =', norm_square_A_grad_y2.item())

# alpha_A0_exact = (1.5 * math.sqrt(2)) * torch.ones(1, dtype=dtype, device=device)

# \int\int \left[  \frac{\grad A}{\grad y_i}
#         + \nabla_y A \cdot \nabla_y \chi_i
#         + A \nabla_y^2 \chi_i
def criterion_chi2(model_chi2_x, model_chi2_y):
    chi2_phi_x = model_chi2_x(w1, x1, need_grad=0)
    chi2_phi_y, chi2_grad_phi_y, chi2_grad_grad_phi_y = model_chi2_y(w3, x3, need_grad=2)

    chi2_grad_y1 = torch.cat((chi2_phi_x, chi2_grad_phi_y[0:1], chi2_phi_y[1:2]), dim=0)
    chi2_grad_y2 = torch.cat((chi2_phi_x, chi2_phi_y[0:1], chi2_grad_phi_y[1:2]), dim=0)
    chi2_grad2_y1 = torch.cat((chi2_phi_x, chi2_grad_grad_phi_y[0:1], chi2_phi_y[1:2]), dim=0)
    chi2_grad2_y2 = torch.cat((chi2_phi_x, chi2_phi_y[0:1], chi2_grad_grad_phi_y[1:2]), dim=0)

    operand_A_grad_y1_chi2_grad_y1 = torch.einsum('dpn, dqn->dpqn', A_grad_y1, chi2_grad_y1)
    operand_A_grad_y2_chi2_grad_y2 = torch.einsum('dpn, dqn->dpqn', A_grad_y2, chi2_grad_y2)
    operand_A_chi2_grad2_y1 = torch.einsum('dpn, dqn->dpqn', A, chi2_grad2_y1)
    operand_A_chi2_grad2_y2 = torch.einsum('dpn, dqn->dpqn', A, chi2_grad2_y2)

    my_operand_phi = torch.cat((operand_A_grad_y1_chi2_grad_y1, operand_A_grad_y2_chi2_grad_y2, operand_A_chi2_grad2_y1, operand_A_chi2_grad2_y2), dim=1)
    my_operand_alpha = torch.cat((alpha_A_grad_y1, alpha_A_grad_y2, alpha_A, alpha_A))
    # p * p matrix
    my_matrix = torch.sum(torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(torch.einsum('dn, dpqn, djkn->dpjqk', w1234, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    # p vector
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_A_grad_y2).unsqueeze(dim=-1) * torch.prod(torch.einsum('dn, dpqn, dkn->dpkq', w1234, my_operand_phi, A_grad_y2), dim=0), dim=(0, 1))
    
    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_chi2_x.scaling_par().data.copy_(my_alpha)

    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_square_A_grad_y2)
    # loss = torch.sqrt(torch.sum(torch.outer(my_alpha, my_alpha) * my_matrix) - 2 * torch.sum(my_alpha * my_vector) + norm_square_A1_grad_y)

    return loss

def train_chi2(model_chi2_x, model_chi2_y, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_chi2(model_chi2_x, model_chi2_y)
            if e == 0:
                print('\n For j=2, Step 1 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                print('lr: ', scheduler.get_last_lr()[0])
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_chi2(model_chi2_x, model_chi2_y)
                    print('\n [ epoch %d ]' % (e+1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])

                
    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_chi2(model_chi2_x, model_chi2_y)
            optimizer.zero_grad()
            loss.backward()
            return loss
        
        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_chi2(model_chi2_x, model_chi2_y)
                    print('\n Step 1 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_chi2(model_chi2_x, model_chi2_y)
                    print('\n [ epoch %d ]' % (e+1))
                    print('loss: ', loss.item())



if __name__ == '__main__':
    # 选择网络[1,2,50,50,50,20]
    p_net = 50
    p_net_periodic = 2
    sizes_periodic = [1, p_net_periodic, p_net, p_net, p_net, p]
    print(sizes_periodic)
    activation = TNN_Sin

    if not os.path.exists("model_chi2"):
        os.mkdir("model_chi2")
    

    model_chi2_x = torch.load("./model_chi2/d2_ex2_final_chi2_x.pkl",map_location=device)
    model_chi2_y = torch.load("./model_chi2/d2_ex2_final_chi2_y.pkl",map_location=device)
    loss = criterion_chi2(model_chi2_x, model_chi2_y)
    print('loss: ', loss.item())

    model_chi2_x = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=True).to(dtype).to(device)
    model_chi2_y = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=False).to(dtype).to(device)

    optimizer = optim.Adam(filter(lambda p:p.requires_grad, itertools.chain(model_chi2_x.parameters(), model_chi2_y.parameters())), lr=1e-2)

    epochs = 50000

    begin_time = time.time()
    train_chi2(model_chi2_x, model_chi2_y, optimizer, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_chi2_x, "./model_chi2/{}_final_chi2_x.pkl".format(name))
    torch.save(model_chi2_y, "./model_chi2/{}_final_chi2_y.pkl".format(name))
