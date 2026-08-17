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
import time
import importlib.util

pi = 3.14159265358979323846
dtype = torch.double
device = 'cuda:5'
name = 'd2_ex2'

dim=2
a, b = 0, 1

p_A = 1
p_chi = 20
p_u0 = 20
p_u1 = 20
p_u2 = 20
p_u3 = 20
p_tilde_u1 = 20
p_delta_u2 = 20
p_tilde_u2 = 40



x1, w1 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
N = len(x1)
x2, w2 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
x3, w3 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
x4, w4 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)

w12 = torch.stack((w1, w2), dim=0)
w34 = torch.stack((w3, w4), dim=0)
w1234 = torch.stack((w1, w2, w3, w4), dim=0)
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

model_u0 = torch.load('./model_u0/d2_ex2_final_u0.pkl', map_location=device)
model_u1_x = torch.load('./model_u1/d2_ex2_final_u1_x.pkl', map_location=device)
model_u1_y = torch.load('./model_u1/d2_ex2_final_u1_y.pkl', map_location=device)
model_u2_x = torch.load('./model_u2/d2_ex2_final_u2_x.pkl', map_location=device)
model_u2_y = torch.load('./model_u2/d2_ex2_final_u2_y.pkl', map_location=device)
model_tilde_u1 = torch.load('./model_tilde_u1/d2_ex2_final_tilde_u1.pkl', map_location=device)
model_delta_u2_x = torch.load('./model_delta_u2/d2_ex2_final_delta_u2_x.pkl', map_location=device)
model_delta_u2_y = torch.load('./model_delta_u2/d2_ex2_final_delta_u2_y.pkl', map_location=device)
model_u3_x = torch.load('./model_u3/d2_ex2_final_u3_x.pkl', map_location=device)
model_u3_y = torch.load('./model_u3/d2_ex2_final_u3_y.pkl', map_location=device)
model_chi1_x = torch.load('./model_chi1/d2_ex2_final_chi1_x.pkl', map_location=device)
model_chi1_y = torch.load('./model_chi1/d2_ex2_final_chi1_y.pkl', map_location=device)
model_chi2_x = torch.load('./model_chi2/d2_ex2_final_chi2_x.pkl', map_location=device)
model_chi2_y = torch.load('./model_chi2/d2_ex2_final_chi2_y.pkl', map_location=device)
 
with torch.no_grad():
    chi1_phi_x, chi1_grad_phi_x = model_chi1_x(w1, x1, need_grad=1)
    chi1_phi_y, chi1_grad_phi_y = model_chi1_y(w3, x3, need_grad=1)
    alpha_chi1 = model_chi1_x.scaling_par()

    chi2_phi_x, chi2_grad_phi_x = model_chi2_x(w1, x1, need_grad=1)
    chi2_phi_y, chi2_grad_phi_y = model_chi2_y(w3, x3, need_grad=1)
    alpha_chi2 = model_chi2_x.scaling_par()

    # a_{ij}^{*}=\int_{Y}\left(a \delta_{ij} + a \frac{\grad\chi_j}{\grad y_i}\right) dy
    A0_11 = torch.ones((2, p_A + p_A * p_chi, N), dtype=dtype, device=device)
    A0_11[0:2, 0:p_A, :] = A[0:2, :, :]
    A0_11[0:2, p_A:p_A + p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[0:2, :, :], chi1_phi_x).flatten(start_dim=1, end_dim=2)

    A0_11_grad_x1 = torch.ones((2, p_A + p_A * p_chi, N), dtype=dtype, device=device)
    A0_11_grad_x1[0:1, 0:p_A, :] = A_grad[0:1, :, :]
    A0_11_grad_x1[1:2, 0:p_A, :] = A[1:2, :, :]
    A0_11_grad_x1[0:1, p_A:p_A + p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A_grad[0:1, :, :], chi1_phi_x[0:1, :, :]).flatten(start_dim=1, end_dim=2) \
                                                        + torch.einsum('dpn, dqn->dpqn', A[0:1, :, :], chi1_grad_phi_x[0:1, :, :]).flatten(start_dim=1, end_dim=2)
    A0_11_grad_x1[1:2, p_A:p_A + p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[1:2, :, :], chi1_phi_x[1:2, :, :]).flatten(start_dim=1, end_dim=2)

    alpha_A0_11 = torch.ones(p_A + p_A * p_chi, dtype=dtype, device=device)
    alpha_A0_11[0:p_A] = torch.prod(torch.einsum('dn, dpn->dp', w34, A[2:4]), dim=0) * alpha_A
    alpha_A0_11[p_A:p_A + p_A * p_chi] = (torch.einsum('n, pn, qn->pq', w3, A[2, :, :], chi1_grad_phi_y[0, :, :]) * torch.einsum('n, pn, qn->pq', w4, A[3, :, :], chi1_phi_y[1, :, :]) * torch.outer(alpha_A, alpha_chi1)).view(-1)
    
    A0_12 = torch.ones((2, p_A * p_chi, N), dtype=dtype, device=device)
    A0_12[0:2, 0:p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[0:2, :, :], chi2_phi_x).flatten(start_dim=1, end_dim=2)

    A0_12_grad_x1 = torch.ones((2, p_A * p_chi, N), dtype=dtype, device=device)
    A0_12_grad_x1[0:1, 0:p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A_grad[0:1, :, :], chi2_phi_x[0:1, :, :]).flatten(start_dim=1, end_dim=2) \
                                                 + torch.einsum('dpn, dqn->dpqn', A[0:1, :, :], chi2_grad_phi_x[0:1, :, :]).flatten(start_dim=1, end_dim=2)
    A0_12_grad_x1[1:2, 0:p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[1:2, :, :], chi2_phi_x[1:2, :, :]).flatten(start_dim=1, end_dim=2)

    alpha_A0_12 = torch.ones(p_A * p_chi, dtype=dtype, device=device)
    alpha_A0_12[0:p_A * p_chi] = (torch.einsum('n, pn, qn->pq', w3, A[2, :, :], chi2_grad_phi_y[0, :, :]) * torch.einsum('n, pn, qn->pq', w4, A[3, :, :], chi2_phi_y[1, :, :]) * torch.outer(alpha_A, alpha_chi2)).view(-1)

    A0_21 = torch.ones((2, p_A * p_chi, N), dtype=dtype, device=device)
    A0_21[0:2, 0:p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[0:2, :, :], chi1_phi_x).flatten(start_dim=1, end_dim=2)

    A0_21_grad_x2 = torch.ones((2, p_A * p_chi, N), dtype=dtype, device=device)
    A0_21_grad_x2[0:1, 0:p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[0:1, :, :], chi1_phi_x[0:1, :, :]).flatten(start_dim=1, end_dim=2)
    A0_21_grad_x2[1:2, 0:p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A_grad[1:2, :, :], chi1_phi_x)[1:2, :, :].flatten(start_dim=1, end_dim=2) \
                                                + torch.einsum('dpn, dqn->dpqn', A[1:2, :, :], chi1_grad_phi_x)[1:2, :, :].flatten(start_dim=1, end_dim=2)

    alpha_A0_21 = torch.ones(p_A * p_chi, dtype=dtype, device=device)
    alpha_A0_21[0:p_A * p_chi] = (torch.einsum('n, pn, qn->pq', w3, A[2, :, :], chi1_phi_y[0, :, :]) * torch.einsum('n, pn, qn->pq', w4, A[3, :, :], chi1_grad_phi_y[1, :, :]) * torch.outer(alpha_A, alpha_chi1)).view(-1)

    A0_22 = torch.ones((2, p_A + p_A * p_chi, N), dtype=dtype, device=device)
    A0_22[0:2, 0:p_A, :] = A[0:2, :, :]
    A0_22[0:2, p_A:p_A + p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[0:2, :, :], chi2_phi_x).flatten(start_dim=1, end_dim=2)

    A0_22_grad_x2 = torch.ones((2, p_A + p_A * p_chi, N), dtype=dtype, device=device)
    A0_22_grad_x2[0:1, 0:p_A, :] = A[0:1, :, :]
    A0_22_grad_x2[1:2, 0:p_A, :] = A_grad[1:2, :, :]
    A0_22_grad_x2[0:1, p_A:p_A + p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A[0:1, :, :], chi2_phi_x[0:1, :, :]).flatten(start_dim=1, end_dim=2)
    A0_22_grad_x2[1:2, p_A:p_A + p_A * p_chi, :] = torch.einsum('dpn, dqn->dpqn', A_grad[1:2, :, :], chi2_phi_x[1:2, :, :]).flatten(start_dim=1, end_dim=2)\
                                                         + torch.einsum('dpn, dqn->dpqn', A[1:2, :, :], chi2_grad_phi_x[1:2, :, :]).flatten(start_dim=1, end_dim=2)

    alpha_A0_22 = torch.ones(p_A + p_A * p_chi, dtype=dtype, device=device)
    alpha_A0_22[0:p_A] = torch.prod(torch.einsum('dn, dpn->dp', w34, A[2:4]), dim=0) * alpha_A
    alpha_A0_22[p_A:p_A + p_A * p_chi] = (torch.einsum('n, pn, qn->pq', w3, A[2, :, :], chi2_phi_y[0, :, :]) * torch.einsum('n, pn, qn->pq', w4, A[3, :, :], chi2_grad_phi_y[1, :, :]) * torch.outer(alpha_A, alpha_chi2)).view(-1)

# 计算特征值，不需要梯度
with torch.no_grad():
    u0_x, u0_x_grad, u0_x_grad_grad = model_u0(w1, x1, need_grad=2)
    alpha_u0 = model_u0.scaling_par()
    u0_grad_x1 = torch.cat((u0_x_grad[0:1,:,:], u0_x[1:2,:,:]), dim=0)
    u0_grad_x2 = torch.cat((u0_x[0:1,:,:], u0_x_grad[1:2,:,:]), dim=0)

    u1_x, u1_x_grad = model_u1_x(w1, x1, need_grad=1)
    u1_y, u1_y_grad = model_u1_y(w3, x3, need_grad=1)
    alpha_u1 = model_u1_x.scaling_par()
    u1_norm_phi_y = model_u1_y(w3, x3, normed=False)
    u1_norm = torch.sqrt(torch.sum(w3 * u1_norm_phi_y**2, dim=-1)).unsqueeze(dim=-1)
    u1_y_shift = model_u1_y(dummy, x_error_0, need_grad=0, normed=False)
    u1_y_shift = u1_y_shift / u1_norm
    expand_u1_phi = torch.cat((u1_x, u1_y), dim=0)
    expand_u1_phi_shift = torch.cat((u1_x, u1_y_shift.expand(2, p_u1, N)), dim=0)

    tilde_u1_phi, tilde_u1_grad_phi, tilde_u1_grad_grad_phi = model_tilde_u1(w1, x1, need_grad=2)
    alpha_tilde_u1 = model_tilde_u1.scaling_par()
    expand_tilde_u1_phi = torch.cat((tilde_u1_phi, torch.ones((2, p_tilde_u1, N), dtype=dtype, device=device)), dim=0)
    
    cat_u1 = torch.cat((expand_u1_phi, expand_u1_phi_shift, expand_tilde_u1_phi), dim=1)
    alpha_cat_u1 = torch.cat((alpha_u1, -alpha_u1, alpha_tilde_u1), dim=0)

    u2_x, u2_x_grad, u2_x_grad_grad = model_u2_x(w1, x1, need_grad=2)
    u2_y = model_u2_y(w3, x3, need_grad=0)
    alpha_u2 = model_u2_x.scaling_par()
    u2_norm_phi_y = model_u2_y(w3, x3, normed=False)
    u2_norm = torch.sqrt(torch.sum(w3 * u2_norm_phi_y**2, dim=-1)).unsqueeze(dim=-1)
    u2_y_shift = model_u2_y(dummy, x_error_0, need_grad=0, normed=False)
    u2_y_shift = u2_y_shift / u2_norm

    expand_u2_phi = torch.cat((u2_x, u2_y), dim=0)
    expand_u2_phi_shift = torch.cat((u2_x, u2_y_shift.expand(2, p_u2, N)), dim=0)
    expand_u2_grad_x1 = torch.cat((u2_x_grad[0:1, :, :], u2_x[1:2, :, :], u2_y), dim=0)
    expand_u2_grad_x1_shift = torch.cat((u2_x_grad[0:1, :, :], u2_x[1:2, :, :], u2_y_shift.expand(2, p_u2, N)), dim=0)
    expand_u2_grad_x2 = torch.cat((u2_x[0:1, :, :], u2_x_grad[1:2, :, :], u2_y), dim=0)
    expand_u2_grad_x2_shift = torch.cat((u2_x[0:1, :, :], u2_x_grad[1:2, :, :], u2_y_shift.expand(2, p_u2, N)), dim=0)
    expand_u2_grad_grad_x1 = torch.cat((u2_x_grad_grad[0:1, :, :], u2_x[1:2, :, :], u2_y), dim=0)
    expand_u2_grad_grad_x1_shift = torch.cat((u2_x_grad_grad[0:1, :, :], u2_x[1:2, :, :], u2_y_shift.expand(2, p_u2, N)), dim=0)
    expand_u2_grad_grad_x2 = torch.cat((u2_x[0:1, :, :], u2_x_grad_grad[1:2, :, :], u2_y), dim=0)
    expand_u2_grad_grad_x2_shift = torch.cat((u2_x[0:1, :, :], u2_x_grad_grad[1:2, :, :], u2_y_shift.expand(2, p_u2, N)), dim=0)

    delta_u2_phi_x, delta_u2_grad_phi_x, delta_u2_grad_grad_phi_x = model_delta_u2_x(w1, x1, need_grad=2)
    delta_u2_phi_y = model_delta_u2_y(w3, x3, need_grad=0)
    alpha_delta_u2 = model_delta_u2_x.scaling_par()
    delta_u2_norm_phi_y = model_delta_u2_y(w3, x3, normed=False)
    delta_u2_norm = torch.sqrt(torch.sum(w3 * delta_u2_norm_phi_y**2, dim=-1)).unsqueeze(dim=-1)
    delta_u2_y_shift = model_delta_u2_y(dummy, x_error_0, need_grad=0, normed=False)
    delta_u2_y_shift = delta_u2_y_shift / delta_u2_norm

    expand_delta_u2_phi = torch.cat((delta_u2_phi_x, delta_u2_phi_y), dim=0)
    expand_delta_u2_phi_shift = torch.cat((delta_u2_phi_x, delta_u2_y_shift.expand(2, p_delta_u2, N)), dim=0)
    expand_delta_u2_grad_x1 = torch.cat((delta_u2_grad_phi_x[0:1, :, :], delta_u2_phi_x[1:2, :, :], delta_u2_phi_y), dim=0)
    expand_delta_u2_grad_x1_shift = torch.cat((delta_u2_grad_phi_x[0:1, :, :], delta_u2_phi_x[1:2, :, :], delta_u2_y_shift.expand(2, p_delta_u2, N)), dim=0)
    expand_delta_u2_grad_x2 = torch.cat((delta_u2_phi_x[0:1, :, :], delta_u2_grad_phi_x[1:2, :, :], delta_u2_phi_y), dim=0)
    expand_delta_u2_grad_x2_shift = torch.cat((delta_u2_phi_x[0:1, :, :], delta_u2_grad_phi_x[1:2, :, :], delta_u2_y_shift.expand(2, p_delta_u2, N)), dim=0)
    expand_delta_u2_grad_grad_x1 = torch.cat((delta_u2_grad_grad_phi_x[0:1, :, :], delta_u2_phi_x[1:2, :, :], delta_u2_phi_y), dim=0)
    expand_delta_u2_grad_grad_x1_shift = torch.cat((delta_u2_grad_grad_phi_x[0:1, :, :], delta_u2_phi_x[1:2, :, :], delta_u2_y_shift.expand(2, p_delta_u2, N)), dim=0)
    expand_delta_u2_grad_grad_x2 = torch.cat((delta_u2_phi_x[0:1, :, :], delta_u2_grad_grad_phi_x[1:2, :, :], delta_u2_phi_y), dim=0)
    expand_delta_u2_grad_grad_x2_shift = torch.cat((delta_u2_phi_x[0:1, :, :], delta_u2_grad_grad_phi_x[1:2, :, :], delta_u2_y_shift.expand(2, p_delta_u2, N)), dim=0)

    cat_u2 = torch.cat((expand_u2_phi, expand_u2_phi_shift, expand_delta_u2_phi, expand_delta_u2_phi_shift), dim=1)
    cat_u2_grad_x1 = torch.cat((expand_u2_grad_x1, expand_u2_grad_x1_shift, expand_delta_u2_grad_x1, expand_delta_u2_grad_x1_shift), dim=1)
    cat_u2_grad_x2 = torch.cat((expand_u2_grad_x2, expand_u2_grad_x2_shift, expand_delta_u2_grad_x2, expand_delta_u2_grad_x2_shift), dim=1)
    cat_u2_grad_grad_x1 = torch.cat((expand_u2_grad_grad_x1, expand_u2_grad_grad_x1_shift, expand_delta_u2_grad_grad_x1, expand_delta_u2_grad_grad_x1_shift), dim=1)
    cat_u2_grad_grad_x2 = torch.cat((expand_u2_grad_grad_x2, expand_u2_grad_grad_x2_shift, expand_delta_u2_grad_grad_x2, expand_delta_u2_grad_grad_x2_shift), dim=1)
    alpha_cat_u2 = torch.cat((alpha_u2, -alpha_u2, alpha_delta_u2, -alpha_delta_u2), dim=0)

    u3_x, u3_x_grad = model_u3_x(w1, x1, need_grad=1)
    u3_y, u3_y_grad = model_u3_y(w3, x3, need_grad=1)
    alpha_u3 = model_u3_x.scaling_par()
    u3_x_grad_x1 = torch.cat((u3_x_grad[0:1,:,:], u3_x[1:2,:,:]), dim=0)
    u3_x_grad_x2 = torch.cat((u3_x[0:1,:,:], u3_x_grad[1:2,:,:]), dim=0)
    u3_y_grad_y1 = torch.cat((u3_y_grad[0:1,:,:], u3_y[1:2,:,:]), dim=0)
    u3_y_grad_y2 = torch.cat((u3_y[0:1,:,:], u3_y_grad[1:2,:,:]), dim=0)


    part2_int = torch.prod(torch.einsum('dn,dpn,dqn->dpq', w34, A[2:4,:,:], cat_u2[2:4, :, :]), dim=0)
    alpha_part2 = (torch.outer(alpha_A, alpha_cat_u2) * part2_int).view(-1)
    part2 = torch.einsum('dpn,dqn->dpqn', A[0:2,:,:], cat_u2_grad_grad_x1[0:2, :, :]).flatten(start_dim=1, end_dim=2)
    part3 = torch.einsum('dpn,dqn->dpqn', A[0:2,:,:], cat_u2_grad_grad_x2[0:2, :, :]).flatten(start_dim=1, end_dim=2)
    part4 = torch.einsum('dpn,dqn->dpqn', A_grad_x1[0:2,:,:], cat_u2_grad_x1[0:2, :, :]).flatten(start_dim=1, end_dim=2)
    part5 = torch.einsum('dpn,dqn->dpqn', A_grad_x2[0:2,:,:], cat_u2_grad_x2[0:2, :, :]).flatten(start_dim=1, end_dim=2)

    part6_int = torch.prod(torch.einsum('dn,dpn,dqn->dpq', w34, A[2:4,:,:], u3_y_grad_y1), dim=0)
    alpha_part6 = (torch.outer(alpha_A, alpha_u3) * part6_int).view(-1)
    part6 = torch.einsum('dpn,dqn->dpqn', A[0:2,:,:], u3_x_grad_x1).flatten(start_dim=1, end_dim=2)
    part7 = torch.einsum('dpn,dqn->dpqn', A_grad_x1[0:2,:,:], u3_x).flatten(start_dim=1, end_dim=2)

    part8_int = torch.prod(torch.einsum('dn,dpn,dqn->dpq', w34, A[2:4,:,:], u3_y_grad_y2), dim=0)
    alpha_part8 = (torch.outer(alpha_A, alpha_u3) * part8_int).view(-1)
    part8 = torch.einsum('dpn,dqn->dpqn', A[0:2,:,:], u3_x_grad_x2).flatten(start_dim=1, end_dim=2)
    part9 = torch.einsum('dpn,dqn->dpqn', A_grad_x2[0:2,:,:], u3_x).flatten(start_dim=1, end_dim=2)

    rhs = torch.cat((part2, part3, part4, part5, part6, part7, part8, part9), dim=1)
    alpha_rhs = torch.cat((alpha_part2, alpha_part2, alpha_part2, alpha_part2, alpha_part6, alpha_part6, alpha_part8, alpha_part8))
    norm_square_rhs = Int2TNN(w12, alpha_rhs, rhs, alpha_rhs, rhs)

    print(Int1TNN(w1, alpha_rhs, rhs))
    
def criterion_tilde_u2(model_tilde_u2):
    tilde_u2_phi, tilde_u2_grad_phi, tilde_u2_grad_grad_phi = model_tilde_u2(w1, x1, need_grad=2)
    tilde_u2_x_grad_x1 = torch.cat((tilde_u2_grad_phi[0:1,:,:], tilde_u2_phi[1:2,:,:]), dim=0)
    tilde_u2_x_grad_x2 = torch.cat((tilde_u2_phi[0:1,:,:], tilde_u2_grad_phi[1:2,:,:]), dim=0)
    tilde_u2_x_grad2_x1 = torch.cat((tilde_u2_grad_grad_phi[0:1,:,:], tilde_u2_phi[1:2,:,:]), dim=0)
    tilde_u2_x_grad2_x2 = torch.cat((tilde_u2_phi[0:1,:,:], tilde_u2_grad_grad_phi[1:2,:,:]), dim=0)

    operand_A0_grad_x1_tilde_u2_grad_x1 = torch.einsum('dpn, dqn->dpqn', A0_11_grad_x1, tilde_u2_x_grad_x1)
    operand_A0_tilde_u2_grad2_x1 = torch.einsum('dpn, dqn->dpqn', A0_11, tilde_u2_x_grad2_x1)
    operand_A0_grad_x1_tilde_u2_grad_x2 = torch.einsum('dpn, dqn->dpqn', A0_12_grad_x1, tilde_u2_x_grad_x2)
    operand_A0_tilde_u2_grad_x1_x2 = torch.einsum('dpn, dqn->dpqn', A0_12, tilde_u2_grad_phi)
    operand_A0_grad_x2_tilde_u2_grad_x1 = torch.einsum('dpn, dqn->dpqn', A0_21_grad_x2, tilde_u2_x_grad_x1)
    operand_A0_tilde_u2_grad_x2_x1 = torch.einsum('dpn, dqn->dpqn', A0_21, tilde_u2_grad_phi)
    operand_A0_grad_x2_tilde_u2_grad_x2 = torch.einsum('dpn, dqn->dpqn', A0_22_grad_x2, tilde_u2_x_grad_x2)
    operand_A0_tilde_u2_grad2_x2 = torch.einsum('dpn, dqn->dpqn', A0_22, tilde_u2_x_grad2_x2)
 
    my_operand_phi = torch.cat((operand_A0_grad_x1_tilde_u2_grad_x1, operand_A0_tilde_u2_grad2_x1, operand_A0_grad_x1_tilde_u2_grad_x2,operand_A0_tilde_u2_grad_x1_x2,operand_A0_grad_x2_tilde_u2_grad_x1,operand_A0_tilde_u2_grad_x2_x1,operand_A0_grad_x2_tilde_u2_grad_x2,operand_A0_tilde_u2_grad2_x2), dim=1)
    my_operand_alpha = torch.cat((alpha_A0_11,alpha_A0_11,alpha_A0_12,alpha_A0_12,alpha_A0_21,alpha_A0_21,alpha_A0_22,alpha_A0_22))
    # p * p matrix
    my_matrix = torch.sum(
        torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(
            torch.einsum('n, dpqn, djkn->dpjqk', w1, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    # p vector
          
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_rhs).unsqueeze(dim=-1) * torch.prod(
        torch.einsum('n, dpqn, dkn->dpkq', w1, my_operand_phi, rhs), dim=0), dim=(0, 1))

    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_tilde_u2.scaling_par().data.copy_(my_alpha)

    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_square_rhs)
    
    return loss


def train_1(model_tilde_u2, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_tilde_u2(model_tilde_u2)
            if e == 0:
                print('\n Step 9 {} Initial:'.format(optimizer_flag))
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
                    torch.save(model_tilde_u2, "./model_tilde_u2/{}_final_tilde_u2.pkl".format(name))

        
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
    # sizes_periodic = [1, 2, 50, 50, 50, 20] #test1 38.956921642808396
    # sizes_periodic = [1, 5, 80, 80, 80, 20] #test2 0.14844737134487795
    # sizes_periodic = [1, 8, 100, 100, 100, 20] # test3 0.08467888962555953
    sizes_periodic = [1, 8, 150, 150, 150, 20] # test4 0.0678646839178611(✅)
    # sizes_periodic = [1, 2, 50, 50, 50, 50, 20] #test5 no
    # sizes_periodic = [1, 5, 80, 80, 80, 80, 20] #test6 0.08335284675984007
    # sizes_periodic = [1, 8, 100, 100, 100, 100, 20] # test7 3.248708690735737
    # sizes_periodic = [1, 8, 150, 150, 150, 150, 20] # test8 10.914631896113713
    # sizes_periodic = [1, 8, 200, 200, 200, 20] # test9 0.7912882207335266
    # sizes_periodic = [1, 8, 200, 200, 200, 200, 20] # test10 0.22025247792266814
    # sizes_periodic = [1, 5, 80, 80, 80, 40] #test11 0.30167501694467563
    # sizes_periodic = [1, 8, 150, 150, 150, 40] # test12 0.1417722963444948
    # sizes_periodic = [1, 5, 80, 80, 80, 80, 40] #test13 0.1585630195547527
    # sizes_periodic = [1, 8, 200, 200, 200, 40] # test14 8.637868603385947
    print(sizes_periodic)

    activation = TNN_Sin

    if not os.path.exists("model_tilde_u2"):
        os.mkdir("model_tilde_u2")

    # model_tilde_u2 = torch.load('./model_tilde_u2/d2_ex2_final_tilde_u2.pkl',map_location=device)
    # loss = criterion_tilde_u2(model_tilde_u2)
    # print('loss: ', loss.item())

    model_tilde_u2 = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=True).to(dtype).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, itertools.chain(model_tilde_u2.parameters())),
        lr=1e-2)

    epochs = 50000
    begin_time = time.time()
    train_1(model_tilde_u2, optimizer, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_tilde_u2, "./model_tilde_u2/{}_final_tilde_u2.pkl".format(name))

    # 可选：使用 LBFGS 进一步优化
    print('*' * 20, 'LBFGS', '*' * 20)
    lr = 0.1
    epochs = 5000
    optimizer = torch.optim.LBFGS(
        filter(lambda p: p.requires_grad, itertools.chain(model_tilde_u2.parameters())),
        lr=lr)
    train_1(model_tilde_u2, optimizer, 'LBFGS', epochs)
    
    torch.save(model_tilde_u2, "./model_tilde_u2/{}_final_tilde_u2.pkl".format(name))
