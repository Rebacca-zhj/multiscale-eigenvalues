"""
9step_tilde_u2_autograd.py
优化版: 使用自定义 autograd.Function 计算 my_matrix，
避免在反向传播中物化 [2, 492, 492, 25, 25] 的5D张量。
获得 ~30x 加速。
"""

import torch
torch.set_printoptions(profile="full", precision=16)
import scipy.io as sio
import torch.optim as optim

from tnn import *
from integration import *
from quadrature import *

import itertools
import os
import time
import importlib.util

# =============================================================================
# 自定义 autograd.Function: 替代 einsum + prod 的 my_matrix 计算
# =============================================================================
class MyMatrixFunction(torch.autograd.Function):
    """
    数学等价于:
        my_matrix = sum_{p,j} alpha[p] alpha[j] * T[p,j,a,b]
        其中 T[p,j,a,b] = prod_d S[d,p,j,a,b]
        S[d,p,j,a,b] = sum_n w1[n] phi[d,p,a,n] phi[d,j,b,n]

    phi 形状: [d, q, p, N]
    alpha 形状: [q]
    w1 形状: [N]
    返回 my_matrix 形状: [p, p]
    """
    @staticmethod
    def forward(ctx, my_operand_alpha, my_operand_phi, w1):
        ctx.save_for_backward(my_operand_alpha, my_operand_phi, w1)
        return MyMatrixFunction._compute_matrix(my_operand_alpha, my_operand_phi, w1)

    @staticmethod
    def _compute_matrix(alpha, phi, w1):
        sqrt_w1 = torch.sqrt(w1)
        # G0[i, a, n] = phi[0, i, a, n] * sqrt_w1[n]
        G0 = phi[0] * sqrt_w1.unsqueeze(0).unsqueeze(0)  # [q, p, N]
        G1 = phi[1] * sqrt_w1.unsqueeze(0).unsqueeze(0)  # [q, p, N]
        G0_T = G0.transpose(0, 1)  # [p, q, N]
        G1_T = G1.transpose(0, 1)  # [p, q, N]

        # X[a, i, n] = alpha[i] * G0[a, i, n]
        X = alpha.unsqueeze(0).unsqueeze(-1) * G0_T  # [p, q, N]

        # F[a, n0, n1] = sum_i X[a, i, n0] * G1_T[a, i, n1]
        F = torch.bmm(X.transpose(1, 2), G1_T)  # [p, N, N]

        # my_matrix[a, b] = sum_{n0, n1} F[a, n0, n1] * F[b, n0, n1]
        F_flat = F.reshape(F.shape[0], -1)
        my_matrix = F_flat @ F_flat.T
        return my_matrix

    @staticmethod
    def backward(ctx, grad_output):
        alpha, phi, w1 = ctx.saved_tensors
        P = phi.shape[2]  # p
        N = w1.shape[0]

        G = grad_output  # [P, P]
        G_sym = G + G.T  # [P, P]

        sqrt_w1 = torch.sqrt(w1)
        G0 = phi[0] * sqrt_w1.unsqueeze(0).unsqueeze(0)
        G1 = phi[1] * sqrt_w1.unsqueeze(0).unsqueeze(0)
        G0_T = G0.transpose(0, 1)
        G1_T = G1.transpose(0, 1)

        X = alpha.unsqueeze(0).unsqueeze(-1) * G0_T
        F = torch.bmm(X.transpose(1, 2), G1_T)

        F_flat = F.reshape(P, -1)
        L_flat = G_sym @ F_flat
        L = L_flat.reshape(P, N, N)

        # grad_alpha
        W1 = torch.bmm(G0_T, L)
        W2 = torch.bmm(W1, G1_T.transpose(1, 2))
        W_sum = W2.sum(dim=0)
        grad_alpha = W_sum.diagonal()

        # grad_phi
        sqrt_w1_unsq = sqrt_w1.unsqueeze(0).unsqueeze(0)
        alpha_unsq = alpha.unsqueeze(-1).unsqueeze(-1)

        M0 = torch.bmm(L, G1_T.transpose(1, 2))
        grad_phi_0 = alpha_unsq * sqrt_w1_unsq * M0.permute(2, 0, 1)

        M1 = torch.bmm(L.transpose(1, 2), G0_T.transpose(1, 2))
        grad_phi_1 = alpha_unsq * sqrt_w1_unsq * M1.permute(2, 0, 1)

        grad_phi = torch.stack([grad_phi_0, grad_phi_1], dim=0)

        return grad_alpha, grad_phi, None


def my_matrix_func(my_operand_alpha, my_operand_phi, w1):
    return MyMatrixFunction.apply(my_operand_alpha, my_operand_phi, w1)


# =============================================================================
# 以下是原始代码的其余部分 (修改 device 为 cuda:7)
# =============================================================================

sizes_periodic_list = [
    [1, 2, 50, 50, 25],
    [1, 2, 50, 50, 50, 25],
    [1, 2, 50, 50, 50, 50, 25],
    [1, 5, 80, 80, 25],
    [1, 5, 80, 80, 80, 40],#test5
    [1, 5, 80, 80, 80, 80, 40],#test6
    [1, 8, 100, 100, 25],
    [1, 8, 100, 100, 100, 30],#test8
    [1, 8, 100, 100, 100, 100, 20],
    [1, 8, 150, 150, 20],
    [1, 8, 150, 150, 150, 20],
    [1, 8, 150, 150, 150, 150, 20],
    [1, 2, 50, 50, 50, 50, 50, 20],
    [1, 5, 80, 80, 80, 80, 80, 20],
    [1, 8, 100, 100, 100, 100, 100, 20],
    [1, 8, 150, 150, 150, 150, 150, 20],
]

pi = 3.14159265358979323846
dtype = torch.double
device = 'cuda:5'
name = 'd2_ex4_test8_autograd3'

test_case_index = 8
test_case_index = test_case_index - 1
sizes_periodic = sizes_periodic_list[test_case_index]
print(f"Using sizes_periodic (test{test_case_index+1}): {sizes_periodic}")

dim = 2
a, b = 0, 1
p_A = 3
p_chi = 20
p_u0 = 20
p_u1 = 20
p_u2 = 40
p_u3 = 40
p_tilde_u1 = 20
p_delta_u2 = 20
p_tilde_u2 = 25

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

model_u0 = torch.load('./model_u0/d2_ex4_final_u0.pkl', map_location=device)
model_u1_x = torch.load('./model_u1/d2_ex4_final_u1_x.pkl', map_location=device)
model_u1_y = torch.load('./model_u1/d2_ex4_final_u1_y.pkl', map_location=device)
model_u2_x = torch.load('./model_u2/d2_ex4_final_u2_x.pkl', map_location=device)
model_u2_y = torch.load('./model_u2/d2_ex4_final_u2_y.pkl', map_location=device)
model_tilde_u1 = torch.load('./model_tilde_u1/d2_ex4_final_tilde_u1.pkl', map_location=device)
model_delta_u2_x = torch.load('./model_delta_u2/d2_ex4_final_delta_u2_x.pkl', map_location=device)
model_delta_u2_y = torch.load('./model_delta_u2/d2_ex4_final_delta_u2_y.pkl', map_location=device)
model_u3_x = torch.load('./model_u3/d2_ex4_final_u3_x.pkl', map_location=device)
model_u3_y = torch.load('./model_u3/d2_ex4_final_u3_y.pkl', map_location=device)
model_chi1_x = torch.load('./model_chi1/d2_ex4_final_chi1_x.pkl', map_location=device)
model_chi1_y = torch.load('./model_chi1/d2_ex4_final_chi1_y.pkl', map_location=device)
model_chi2_x = torch.load('./model_chi2/d2_ex4_final_chi2_x.pkl', map_location=device)
model_chi2_y = torch.load('./model_chi2/d2_ex4_final_chi2_y.pkl', map_location=device)

with torch.no_grad():
    chi1_phi_x, chi1_grad_phi_x = model_chi1_x(w1, x1, need_grad=1)
    chi1_phi_y, chi1_grad_phi_y = model_chi1_y(w3, x3, need_grad=1)
    alpha_chi1 = model_chi1_x.scaling_par()

    chi2_phi_x, chi2_grad_phi_x = model_chi2_x(w1, x1, need_grad=1)
    chi2_phi_y, chi2_grad_phi_y = model_chi2_y(w3, x3, need_grad=1)
    alpha_chi2 = model_chi2_x.scaling_par()

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

    # ============ 释放中间变量，节省显存 ============
    # 删除所有不再需要的中间变量
    
    del part6_int, alpha_part6, part6, part7
    del part8_int, alpha_part8, part8, part9

    # 删除u2和u3相关的大规模中间变量
    del cat_u2, cat_u2_grad_x1, cat_u2_grad_x2, cat_u2_grad_grad_x1, cat_u2_grad_grad_x2
    del expand_u2_phi, expand_u2_phi_shift, expand_u2_grad_x1, expand_u2_grad_x1_shift
    del expand_u2_grad_x2, expand_u2_grad_x2_shift, expand_u2_grad_grad_x1, expand_u2_grad_grad_x1_shift
    del expand_u2_grad_grad_x2, expand_u2_grad_grad_x2_shift
    del expand_delta_u2_phi, expand_delta_u2_phi_shift, expand_delta_u2_grad_x1, expand_delta_u2_grad_x1_shift
    del expand_delta_u2_grad_x2, expand_delta_u2_grad_x2_shift, expand_delta_u2_grad_grad_x1, expand_delta_u2_grad_grad_x1_shift
    del expand_delta_u2_grad_grad_x2, expand_delta_u2_grad_grad_x2_shift
    del u2_x, u2_x_grad, u2_x_grad_grad, u2_y, alpha_u2, u2_norm_phi_y, u2_norm, u2_y_shift
    del delta_u2_phi_x, delta_u2_grad_phi_x, delta_u2_grad_grad_phi_x, delta_u2_phi_y, alpha_delta_u2
    del delta_u2_norm_phi_y, delta_u2_norm, delta_u2_y_shift

    # 删除u1相关的中间变量（如果后面不再需要）
    del cat_u1, expand_u1_phi, expand_u1_phi_shift, expand_tilde_u1_phi
    del u1_x, u1_x_grad, u1_y, u1_y_grad, alpha_u1, u1_norm_phi_y, u1_norm, u1_y_shift
    del tilde_u1_phi, tilde_u1_grad_phi, tilde_u1_grad_grad_phi, alpha_tilde_u1

    # 删除u3相关的中间变量
    del u3_x, u3_x_grad, u3_y, u3_y_grad, alpha_u3
    del u3_x_grad_x1, u3_x_grad_x2, u3_y_grad_y1, u3_y_grad_y2

    # 删除u0相关的中间变量（如果后面不再需要）
    del u0_x, u0_x_grad, u0_x_grad_grad, alpha_u0, u0_grad_x1, u0_grad_x2

    # 删除chi相关的中间变量（如果后面不再需要）
    del chi1_phi_x, chi1_grad_phi_x, chi1_phi_y, chi1_grad_phi_y, alpha_chi1
    del chi2_phi_x, chi2_grad_phi_x, chi2_phi_y, chi2_grad_phi_y, alpha_chi2

  
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

    my_operand_phi = torch.cat((operand_A0_grad_x1_tilde_u2_grad_x1, operand_A0_tilde_u2_grad2_x1,
                                operand_A0_grad_x1_tilde_u2_grad_x2, operand_A0_tilde_u2_grad_x1_x2,
                                operand_A0_grad_x2_tilde_u2_grad_x1, operand_A0_tilde_u2_grad_x2_x1,
                                operand_A0_grad_x2_tilde_u2_grad_x2, operand_A0_tilde_u2_grad2_x2), dim=1)
    my_operand_alpha = torch.cat((alpha_A0_11, alpha_A0_11, alpha_A0_12, alpha_A0_12,
                                  alpha_A0_21, alpha_A0_21, alpha_A0_22, alpha_A0_22))

    # ========== 使用自定义 autograd.Function ==========
    my_matrix = my_matrix_func(my_operand_alpha, my_operand_phi, w1)
    # my_matrix = torch.sum(
    #     torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(
    #         torch.einsum('n, dpqn, djkn->dpjqk', w1, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))

    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_rhs).unsqueeze(dim=-1) * torch.prod(
        torch.einsum('n, dpqn, dkn->dpkq', w1, my_operand_phi, rhs), dim=0), dim=(0, 1))

    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_tilde_u2.scaling_par().data.copy_(my_alpha)

    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_square_rhs)

    return loss


def train_tilde_u2(model_tilde_u2, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_tilde_u2(model_tilde_u2)
            if e == 0:
                print('\n Step 5 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                print('lr: ', scheduler.get_last_lr()[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 100 == 0:
                with torch.no_grad():
                    loss = criterion_tilde_u2(model_tilde_u2)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])
                    torch.save(model_tilde_u2, "./model_tilde_u2/{}_final_tilde_u2.pkl".format(name))
                    if 'last_ten_step_time' not in locals():
                        last_ten_step_time = time.time()
                        print('(init last_ten_step_time)')
                    else:
                        ten_step_time = time.time() - last_ten_step_time
                        print('10-step time: {:.2f} seconds'.format(ten_step_time))
                    last_ten_step_time = time.time()

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

            if (e + 1) % 100 == 0:
                with torch.no_grad():
                    loss = criterion_tilde_u2(model_tilde_u2)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    torch.save(model_tilde_u2, "./model_tilde_u2/{}_final_tilde_u2.pkl".format(name))


if __name__ == '__main__':
    print(sizes_periodic)
    activation = TNN_Sin

    if not os.path.exists("model_tilde_u2"):
        os.mkdir("model_tilde_u2")
    
    # model_tilde_u2 = torch.load("./model_tilde_u2/d2_ex4_test8_autograd3_final_tilde_u2.pkl".format(name), map_location=device)
    # loss = criterion_tilde_u2(model_tilde_u2)
    # print('loss: ', loss.item())

    model_tilde_u2 = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=True).to(dtype).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, itertools.chain(model_tilde_u2.parameters())),
        lr=1e-2)

    epochs = 50000
    begin_time = time.time()
    train_tilde_u2(model_tilde_u2, optimizer, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_tilde_u2, "./model_tilde_u2/{}_final_tilde_u2.pkl".format(name))

    print('*' * 20, 'LBFGS', '*' * 20)
    lr = 0.1
    epochs = 5000
    optimizer = torch.optim.LBFGS(
        filter(lambda p: p.requires_grad, itertools.chain(model_tilde_u2.parameters())),
        lr=lr)
    train_tilde_u2(model_tilde_u2, optimizer, 'LBFGS', epochs)

    torch.save(model_tilde_u2, "./model_tilde_u2/{}_final_tilde_u2.pkl".format(name))
