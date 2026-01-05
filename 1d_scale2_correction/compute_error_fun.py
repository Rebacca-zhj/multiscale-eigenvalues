import torch
import numpy as np
import scipy.io as sio

from tnn import *
from integration import *
from quadrature import *
import matplotlib.pyplot as plt

pi = 3.14159265358979323846
dtype = torch.double
device = 'cuda:0'
 
m = 1
p_chi1 = 20
p_u0 = 18
p_A = 2
p_A0 = p_A + p_A * p_chi1

a, b = 0, 1

def bd_u0(x):
    return (x - a) * (b - x)

def grad_bd_u0(x):
    return a + b - 2 * x

def grad_grad_bd_u0(x):
    return -2 * torch.ones_like(x)

def grad_grad_grad_bd_u0(x):
    return 0 * torch.ones_like(x)

model_u0 = torch.load('./model_u0/x_free_y_per_final_u0.pkl', map_location=device)

x1, w1 = composite_quadrature_1d(8, 0, 1, 200, device=device)
x2, w2 = composite_quadrature_1d(8, 0, 1, 200, device=device)

N = len(x1)
dummy = torch.ones((1, 1), dtype=dtype, device=device)

x_error, w_error = composite_quadrature_1d(8, 0, 1, 5000, device=device, dtype=dtype)
N_error = len(x_error)
x_error_0 = torch.zeros(1, dtype=dtype, device=device)

A = torch.ones((2, 2, N), dtype=dtype, device=device)
A[0, 0, :] = 2 + x1
A[1, 1, :] = torch.sin(2*pi * x2)
alpha_A = torch.ones(2, dtype=dtype, device=device)

model_chi1_x = torch.load('./model_chi1/x_free_y_per_final_chi1_x.pkl', map_location=device)
model_chi1_y1 = torch.load('./model_chi1/x_free_y_per_final_chi1_y1.pkl', map_location=device)
model_eta1_x = torch.load('./model_eta1/x_free_y_per_shift_final_eta1_x.pkl', map_location=device)
model_eta1_y1 = torch.load('./model_eta1/x_free_y_per_shift_final_eta1_y1.pkl', map_location=device)
model_eta2_x = torch.load('./model_eta2/x_free_y_per_shift_final_eta2_x.pkl', map_location=device)
model_eta2_y1 = torch.load('./model_eta2/x_free_y_per_shift_final_eta2_y1.pkl', map_location=device)
model_w = torch.load('./model_w/x_free_y_per_shift_final_w_x.pkl', map_location=device)

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

# 计算\int A0_TNN (\nabla u)^2 dx =\lambda0 \int u^2 dx 对应的lambda0
def criterion_0(model_u0):
    u0_phi, u0_grad_phi = model_u0(w1, x1, need_grad=1)
    
    alpha = torch.ones(p_u0, device=device, dtype=dtype)
    A = Int2TNN_amend_1d_with_coe(w1, w1, alpha, u0_phi, alpha, u0_phi, u0_grad_phi, u0_grad_phi, alpha_A0, A0,
                                      if_sum=False)
    M = Int2TNN(w1, alpha, u0_phi, alpha, u0_phi, if_sum=False)

    L = torch.linalg.cholesky(M)  # Cholesky分解：M = L L^T
    C = torch.linalg.solve(L, A.t()).t()  # C = A L^{-T}
    D = torch.linalg.solve(L, C)  # D = L^{-1} A L^{-T}
    E, V = torch.linalg.eigh(D)  # 解 D 的特征值问题
    ind = torch.argsort(E)
    lam1 = E[ind]
    u1 = torch.linalg.solve(L.t(),V[:,ind])

    return lam1, u1

lam, u = criterion_0(model_u0)
print('lam[:m]=', lam[:m])

with torch.no_grad():
    # 归一化及函数分量准备
    u0_norm_phi = model_u0(w1, x1, normed=False)
    u0_norm = torch.sqrt(torch.sum(w1 * u0_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u0_phi, u0_grad_phi, u0_grad_grad_phi = model_u0(dummy, x_error, need_grad=2, normed=False)
    u0_phi = u0_phi  / u0_norm
    u0_grad_phi = u0_grad_phi / u0_norm
    u0_grad_grad_phi = u0_grad_grad_phi / u0_norm

    u0 = torch.sum(torch.einsum('pm,pn->pnm', u[:, :m], u0_phi.squeeze(dim=0)), dim=0).squeeze(-1)
    u0_grad = torch.sum(torch.einsum('pm,pn->pnm', u[:, :m], u0_grad_phi.squeeze(dim=0)), dim=0).squeeze(-1)
    u0_grad_grad = torch.sum(torch.einsum('pm,pn->pnm', u[:, :m], u0_grad_grad_phi.squeeze(dim=0)), dim=0).squeeze(-1)

    chi1_x_norm_phi = model_chi1_x(w1, x1, normed=False)
    chi1_x_norm = torch.sqrt(torch.sum(w1 * chi1_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    chi1_x_phi = model_chi1_x(dummy, x_error, normed=False)
    chi1_x_phi = chi1_x_phi / chi1_x_norm
    chi1_alpha = model_chi1_x.scaling_par()

    chi1_y1_norm_phi = model_chi1_y1(w2, x2, normed=False)
    chi1_y1_norm = torch.sqrt(torch.sum(w2 * chi1_y1_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    chi1_y1_shift = model_chi1_y1(dummy, x_error_0, need_grad=0, normed=False)
    chi1_y1_shift = chi1_y1_shift / chi1_y1_norm

    w_norm_phi = model_w(w1, x1, normed=False)
    w_norm = torch.sqrt(torch.sum(w1 * w_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    w_phi, w_grad_phi = model_w(dummy, x_error, need_grad=1, normed=False)
    w_phi = w_phi / w_norm
    w_grad_phi = w_grad_phi / w_norm
    w_alpha = model_w.scaling_par()
    w = torch.sum(((w_phi.squeeze(dim=0)) * w_alpha.unsqueeze(dim=-1)).unsqueeze(dim=-1), dim=0)[:, 0]
    w_grad = torch.sum(((w_grad_phi.squeeze(dim=0)) * w_alpha.unsqueeze(dim=-1)).unsqueeze(dim=-1), dim=0)[:, 0]

    eta1_x_norm_phi = model_eta1_x(w1, x1, normed=False)
    eta1_x_norm = torch.sqrt(torch.sum(w1 * eta1_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    eta1_x_phi = model_eta1_x(dummy, x_error, normed=False)
    eta1_x_phi = eta1_x_phi / eta1_x_norm
    eta1_alpha = model_eta1_x.scaling_par()

    eta1_y1_norm_phi = model_eta1_y1(w2, x2, normed=False)
    eta1_y1_norm = torch.sqrt(torch.sum(w2 * eta1_y1_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    eta1_y1_shift = model_eta1_y1(dummy, x_error_0, need_grad=0, normed=False)
    eta1_y1_shift = eta1_y1_shift / eta1_y1_norm

    eta2_x_norm_phi = model_eta2_x(w1, x1, normed=False)
    eta2_x_norm = torch.sqrt(torch.sum(w1 * eta2_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    eta2_x_phi = model_eta2_x(dummy, x_error, normed=False)
    eta2_x_phi = eta2_x_phi / eta2_x_norm
    eta2_alpha = model_eta2_x.scaling_par()

    eta2_y1_norm_phi = model_eta2_y1(w2, x2, normed=False)
    eta2_y1_norm = torch.sqrt(torch.sum(w2 * eta2_y1_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    eta2_y1_shift = model_eta2_y1(dummy, x_error_0, need_grad=0, normed=False)
    eta2_y1_shift = eta2_y1_shift / eta2_y1_norm

epsilons = [3,5,8,10,12]
l2_norm1_list = []
l2_norm2_list = []
l2_norm3_list = []

s = [1,-1,1,-1,1,-1]

for idx, eps in enumerate(epsilons):
    # 计算一阶展开项
    y_error_eps = x_error * eps - (x_error * eps).floor()
    chi1_y1_phi = model_chi1_y1(dummy, y_error_eps, need_grad=0, normed=False)
    chi1_y1_phi = chi1_y1_phi / chi1_y1_norm
    chi1 = torch.sum((chi1_alpha.unsqueeze(dim=-1) * chi1_x_phi * (chi1_y1_phi - chi1_y1_shift)).squeeze(dim=0), dim=0)
    u1 = chi1 * u0_grad + w
    # u1 = chi1 * u0_grad
    norm_ue = torch.sqrt(torch.sum(w_error * (u0 + u1 / eps) ** 2)).unsqueeze(dim=-1)
    u0_new = u0 / norm_ue
    u1_new = u1 / norm_ue

    # 二阶展开
    eta1_y1_phi = model_eta1_y1(dummy, y_error_eps, need_grad=0, normed=False)
    eta1_y1_phi = eta1_y1_phi / eta1_y1_norm
    eta2_y1_phi = model_eta2_y1(dummy, y_error_eps, need_grad=0, normed=False)
    eta2_y1_phi = eta2_y1_phi / eta2_y1_norm

    eta1 = torch.sum((eta1_alpha.unsqueeze(dim=-1) * eta1_x_phi * (eta1_y1_phi - eta1_y1_shift)).squeeze(dim=0), dim=0)
    eta2 = torch.sum((eta2_alpha.unsqueeze(dim=-1) * eta2_x_phi * (eta2_y1_phi - eta2_y1_shift)).squeeze(dim=0), dim=0)
    u2 = eta1 * u0_grad + eta2 * u0_grad_grad + chi1 * w_grad
    # u2 = eta1 * u0_grad + eta2 * u0_grad_grad
    norm_ue_2 = torch.sqrt(torch.sum(w_error * (u0 + u1 / eps + u2 / (eps ** 2)) ** 2)).unsqueeze(dim=-1)
    u0_new_2 = u0 / norm_ue_2
    u1_new_2 = u1 / norm_ue_2
    u2_new_2 = u2 / norm_ue_2

    # 加载FEM参考解
    func_values_mat = sio.loadmat(f'data_Axy_plus/func_values_{eps}.mat')
    func_values_matrix = func_values_mat['func_values']  # 5000x16
    func_values_vector = func_values_matrix.reshape(-1)
    if idx < len(s) and s[idx] == -1:
        func_values_vector = -func_values_vector

    # 第一项L2误差
    u0_plot = u0.detach().cpu().numpy()
    u0_new_plot = u0_new.detach().cpu().numpy()
    u1_new_plot = u1_new.detach().cpu().numpy()
    u0_new_2_plot = u0_new_2.detach().cpu().numpy()
    u1_new_2_plot = u1_new_2.detach().cpu().numpy()
    u2_new_2_plot = u2_new_2.detach().cpu().numpy()
    w_error_np = w_error.detach().cpu().numpy().squeeze()

    eps_value = 1 / eps
    diff1 = func_values_vector - u0_plot
    diff2 = func_values_vector - (u0_new_plot + eps_value * u1_new_plot)
    diff3 = func_values_vector - (u0_new_2_plot + eps_value * u1_new_2_plot + eps_value ** 2 * u2_new_2_plot)
    l2_norm1 = np.sqrt(np.sum(w_error_np * diff1 ** 2))
    l2_norm2 = np.sqrt(np.sum(w_error_np * diff2 ** 2))
    l2_norm3 = np.sqrt(np.sum(w_error_np * diff3 ** 2))
    print(f'L2 norm of func_values_vector - u0 for eps={eps}:', l2_norm1)
    print(f'L2 norm of func_values_vector - (u0 + eps * u1) for eps={eps}:', l2_norm2)
    print(f'L2 norm of func_values_vector - (u0 + eps * u1 + eps^2 * u2) for eps={eps}:', l2_norm3)
    print('--------------------------------')
    l2_norm1_list.append(l2_norm1)
    l2_norm2_list.append(l2_norm2)
    l2_norm3_list.append(l2_norm3)

# 绘制L2误差收敛图
epsilons_np = np.array(epsilons)
l2_norm1_arr = np.array(l2_norm1_list)
l2_norm2_arr = np.array(l2_norm2_list)
l2_norm3_arr = np.array(l2_norm3_list)

plt.figure(figsize=(8, 6))
inv_epsilons = 1 / epsilons_np

plt.loglog(inv_epsilons, l2_norm1_arr, 'o-', label=r'$\|u_\epsilon - u_0\|_{L^2}$')
plt.loglog(inv_epsilons, l2_norm2_arr, 's-', label=r'$\|u_\epsilon - (u_0 + \epsilon u_1)\|_{L^2}$')
plt.loglog(inv_epsilons, l2_norm3_arr, '^-', label=r'$\|u_\epsilon - (u_0 + \epsilon u_1 + \epsilon^2 u_2)\|_{L^2}$')

# 画参考线
ref_x = np.array([inv_epsilons[0], inv_epsilons[-1]])
ref1_y0 = l2_norm1_arr[0]
ref2_y0 = l2_norm2_arr[0]
ref3_y0 = l2_norm3_arr[0]
ref1 = ref1_y0 * (ref_x / ref_x[0]) ** 1
ref2 = ref2_y0 * (ref_x / ref_x[0]) ** 2
ref3 = ref3_y0 * (ref_x / ref_x[0]) ** 3
plt.loglog(ref_x, ref1, 'k--', alpha=0.7, label=r'ref: slope=1')
plt.loglog(ref_x, ref2, 'k-.', alpha=0.7, label=r'ref: slope=2')
plt.loglog(ref_x, ref3, 'k:', alpha=0.7, label=r'ref: slope=3')

plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$L^2$ error')
plt.title(r'$L^2$ error convergence vs $\epsilon$')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig('png/l2_error_vs_epsilon.png')
plt.close()
