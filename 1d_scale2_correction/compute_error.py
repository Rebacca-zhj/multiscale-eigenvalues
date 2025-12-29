import torch
torch.set_printoptions(profile="full", precision=16)
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import scipy.io as sio
import torch.optim as optim
import math

from tnn import *
from integration import *
from quadrature import *

import itertools
import os
import sys

pi = 3.14159265358979323846
dtype = torch.double
device = 'cuda:5'
 
m=1
p_chi1 = 20
p_u0 = 18
p_A = 2
p_A0 = p_A + p_A * p_chi1

a,b=0,1

def bd_u0(x):
    return (x - a) * (b - x)

def grad_bd_u0(x):
    return a + b - 2 * x

def grad_grad_bd_u0(x):
    return -2 * torch.ones_like(x)

def grad_grad_grad_bd_u0(x):
    return 0 * torch.ones_like(x)

model_u0 = torch.load('./model_u0/x_free_y_per_final_u0.pkl', map_location=device)
# model_u0 = torch.load('./model_u0/ex4_3_1_final_u0.pkl', map_location=device)

x1, w1 = composite_quadrature_1d(8, 0, 1, 200, device=device)
x2, w2 = composite_quadrature_1d(8, 0, 1, 200, device=device)

w12 = torch.stack((w1, w2), dim=0)
N = len(x1)

dummy = torch.ones((1, 1), dtype=dtype, device=device)

x_error, w_error = composite_quadrature_1d(4, 0, 1, 10000, device=device, dtype=dtype)
N_error = len(x_error)

x_error_0 = torch.zeros(1, dtype=dtype, device=device)

A = torch.ones((2, 2, N), dtype=dtype, device=device)
A[0, 0, :] = 2 + x1
A[1, 1, :] = torch.sin(2*pi * x2)
alpha_A = torch.ones(2, dtype=dtype, device=device)

# p_A = 1
# p_A0 = p_A + p_A * p_chi1
# A = torch.ones((2, 1, N), dtype=dtype, device=device)
# A[0, 0, :] = 1 + x1
# A[1, 0, :] = 1 + 0.5 * torch.sin(2*pi * x2)
# alpha_A = torch.ones(1, dtype=dtype, device=device)

# A_partial_x = torch.ones((2, 2, N), dtype=dtype, device=device)
# A_partial_x[0,1,:] = 0
# alpha_A_partial_x = torch.ones(1, dtype=dtype, device=device)

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

    # A0_grad_x = torch.ones((1, p_A + p_A * p_chi1, N), dtype=dtype, device=device)
    # A0_grad_x[0:1, 0:p_A, :] = A_partial_x[0:1, :, :]
    # A0_grad_x[0:1, p_A:p_A + p_A * p_chi1, :] = torch.einsum('dpn,dqn->dpqn', A_partial_x[0:1, :, :], chi1[0:1, :, :]).flatten(start_dim=1, end_dim=2) \
    #                                                     + torch.einsum('dpn,dqn->dpqn', A[0:1, :, :], chi1_grad_phi[0:1, :, :]).flatten(start_dim=1, end_dim=2)

    # alpha_A0_grad_x = alpha_A0



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

# def plot_eigen_error_vs_epsilon(epsilon_list, eigen_error_list, alpha=None, C=None, show=True, save_path=None, filename=None, ylabel='eigen_error'):
#     """
#     画 eigen_error 关于 epsilon 的收敛图(loglog),对eigen_error_list的每一列都画一条线并画一条标准线用于对比。
#     参数:
#         epsilon_list: array-like, shape (m,)
#         eigen_error_list: array-like, shape (m, N)
#         alpha: float, 理论收敛阶数。如果为 None,则自动拟合
#         C: float, 标准线的系数。如果为 None,则自动拟合
#         show: bool, 是否显示图像
#         save_path: str, 如果不为 None,则保存图片到该路径
#         filename: str, 如果不为 None,则作为图片标题的一部分
#         ylabel: str, 纵坐标标签
#     """
#     epsilon_arr = np.array(epsilon_list)
#     error_arr = np.array(eigen_error_list)
#     m, N = error_arr.shape

#     plt.figure(figsize=(7, 5))
#     colors = plt.cm.viridis(np.linspace(0, 1, N))

#     for i in range(N):
#         err_col = error_arr[:, i]
#         plt.loglog(epsilon_arr, err_col, 'o-', color=colors[i], label=f'eigen pairs {i+1}')

#         # 拟合 alpha 和 C（如果未指定，分别对每一列拟合）
#         fit_alpha = alpha
#         fit_C = C
#         if alpha is None or C is None:
#             log_eps = np.log(epsilon_arr)
#             log_err = np.log(err_col)
#             coeffs = np.polyfit(log_eps, log_err, 1)
#             fitted_alpha = coeffs[0]
#             fitted_logC = coeffs[1]
#             if alpha is None:
#                 fit_alpha = fitted_alpha
#             if C is None:
#                 fit_C = np.exp(fitted_logC)
#         # 画标准线
#         std_line = fit_C * epsilon_arr ** fit_alpha
#         plt.loglog(epsilon_arr, std_line, '--', color=colors[i], label=f'standard line {i+1}: $C\\epsilon^{{{fit_alpha:.2f}}}$')
#         # 显示 alpha 拟合结果
#         # plt.text(0.7 * epsilon_arr.max(), 0.7 * err_col.max(), f'alpha {i+1} = {fit_alpha:.3f}', color=colors[i], fontsize=11)
        

#     plt.xlabel(r'$\epsilon$')
#     plt.ylabel(ylabel)

#     # 根据是否提供文件名设置标题
#     if filename is not None:
#         plt.title(f'{filename}')
#     else:
#         plt.title(r'error ~ $\epsilon$ ')

#     # plt.legend()
#     plt.grid(True, which="both", ls="--", alpha=0.5)
#     # 将图例放在图片右下侧，字体调小
#     plt.legend(loc='lower right', fontsize=6)

#     if save_path is not None:
#         plt.savefig(save_path, bbox_inches='tight', dpi=200)
#     else:
#         plt.savefig('png/eigen_error_vs_epsilon.png', bbox_inches='tight', dpi=200)
#     if show:
#         plt.show()
#     plt.close()

def plot_eigen_error_vs_epsilon(epsilon_list, eigen_error_list, alpha=None, C=None, show=True, save_path=None, filename=None, ylabel='eigen_error'):
    """
    画 eigen_error 关于 epsilon 的收敛图(loglog),只画第0个特征对及其标准线。
    参数:
        epsilon_list: array-like, shape (m,)
        eigen_error_list: array-like, shape (m, N)
        alpha: float, 理论收敛阶数。如果为 None,则自动拟合
        C: float, 标准线的系数。如果为 None,则自动拟合
        show: bool, 是否显示图像
        save_path: str, 如果不为 None,则保存图片到该路径
        filename: str, 如果不为 None,则作为图片标题的一部分
        ylabel: str, 纵坐标标签
    """
    epsilon_arr = np.array(epsilon_list)
    error_arr = np.array(eigen_error_list)
    m, N = error_arr.shape

    plt.figure(figsize=(7, 5))
    # 只取第0个特征对，颜色任意取一个
    color = plt.cm.viridis(0.0)

    err_col = error_arr[:, 0]
    plt.loglog(epsilon_arr, err_col, 'o-', color=color, label=f'eigen pair 1')

    # 拟合 alpha 和 C（如果未指定，自动拟合）
    fit_alpha = alpha
    fit_C = C
    if alpha is None or C is None:
        log_eps = np.log(epsilon_arr)
        log_err = np.log(err_col)
        coeffs = np.polyfit(log_eps, log_err, 1)
        fitted_alpha = coeffs[0]
        fitted_logC = coeffs[1]
        if alpha is None:
            fit_alpha = fitted_alpha
        if C is None:
            fit_C = np.exp(fitted_logC)
    # 以最后一个点为起点画标准线
    eps_ref = epsilon_arr[-1]
    err_ref = err_col[-1]
    std_line = err_ref * (epsilon_arr / eps_ref) ** fit_alpha
    plt.loglog(epsilon_arr, std_line, '--', color=color, label=f'standard line: $c\\epsilon^{{{fit_alpha:.2f}}}$')
    # 可选显示 alpha 拟合结果
    # plt.text(0.7 * epsilon_arr.max(), 0.7 * err_col.max(), f'alpha = {fit_alpha:.3f}', color=color, fontsize=11)

    plt.xlabel(r'$\epsilon$')
    plt.ylabel(ylabel)

    # 根据是否提供文件名设置标题
    if filename is not None:
        plt.title(f'{filename}')
    else:
        plt.title(r'error ~ $\epsilon$ ')

    plt.grid(True, which="both", ls="--", alpha=0.5)
    # 将图例放在图片右下侧，字体调小
    plt.legend(loc='lower right', fontsize=8)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    else:
        plt.savefig('png/eigen_error_vs_epsilon.png', bbox_inches='tight', dpi=200)
    if show:
        plt.show()
    plt.close()

with torch.no_grad():
    u0_norm_phi = model_u0(w1, x1, normed=False)
    u0_norm = torch.sqrt(torch.sum(w1 * u0_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u0_phi, u0_grad_phi, u0_grad_grad_phi = model_u0(dummy, x_error, need_grad=2, normed=False)
    u0_phi = u0_phi  / u0_norm
    u0_grad_phi = u0_grad_phi / u0_norm

    u0 = torch.sum(torch.einsum('pm,pn->pnm',u[:,:m], u0_phi.squeeze(dim=0)),dim=0)
    u0_grad = torch.sum(torch.einsum('pm,pn->pnm',u[:,:m], u0_grad_phi.squeeze(dim=0)),dim=0)
    u0_grad_grad = torch.sum(torch.einsum('pm,pn->pnm',u[:,:m], u0_grad_grad_phi.squeeze(dim=0)),dim=0)

    chi1_x_norm_phi = model_chi1_x(w1, x1, normed=False)
    chi1_x_norm = torch.sqrt(torch.sum(w1 * chi1_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    chi1_x_phi = model_chi1_x(dummy, x_error, normed=False)
    chi1_x_phi = chi1_x_phi / chi1_x_norm
    chi1_alpha = model_chi1_x.scaling_par()

    chi1_y1_norm_phi = model_chi1_y1(w2, x2, normed=False)
    chi1_y1_norm = torch.sqrt(torch.sum(w2 * chi1_y1_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    chi1_y1_shift = model_chi1_y1(dummy, x_error_0, need_grad=0, normed=False) #[d,p,1]
    chi1_y1_shift = chi1_y1_shift / chi1_y1_norm
    # chi1_y1_integral_term = (torch.sum(w2 * chi1_y1_norm_phi, dim=-1) / torch.sqrt(torch.sum(w2 * chi1_y1_norm_phi**2, dim=-1))).unsqueeze(dim=-1)

    w_norm_phi = model_w(w1, x1, normed=False)
    w_norm = torch.sqrt(torch.sum(w1 * w_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    w_phi = model_w(dummy, x_error, need_grad=0, normed=False)
    w_phi = w_phi / w_norm
    w_alpha = model_w.scaling_par()
    w = torch.sum(((w_phi.squeeze(dim=0)) * w_alpha.unsqueeze(dim=-1)).unsqueeze(dim=-1), dim=0)

    eta1_x_norm_phi = model_eta1_x(w1, x1, normed=False)
    eta1_x_norm = torch.sqrt(torch.sum(w1 * eta1_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    eta1_x_phi = model_eta1_x(dummy, x_error, normed=False)
    eta1_x_phi = eta1_x_phi / eta1_x_norm
    eta1_alpha = model_eta1_x.scaling_par()

    eta1_y1_norm_phi = model_eta1_y1(w2, x2, normed=False)
    eta1_y1_norm = torch.sqrt(torch.sum(w2 * eta1_y1_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    eta1_y1_shift = model_eta1_y1(dummy, x_error_0, need_grad=0, normed=False) #[d,p,1]
    eta1_y1_shift = eta1_y1_shift / eta1_y1_norm
    # eta1_y1_integral_term = (torch.sum(w2 * eta1_y1_norm_phi, dim=-1) / torch.sqrt(torch.sum(w2 * eta1_y1_norm_phi**2, dim=-1))).unsqueeze(dim=-1)

    eta2_x_norm_phi = model_eta2_x(w1, x1, normed=False)
    eta2_x_norm = torch.sqrt(torch.sum(w1 * eta2_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    eta2_x_phi = model_eta2_x(dummy, x_error, normed=False)
    eta2_x_phi = eta2_x_phi / eta2_x_norm
    eta2_alpha = model_eta2_x.scaling_par()

    eta2_y1_norm_phi = model_eta2_y1(w2, x2, normed=False)
    eta2_y1_norm = torch.sqrt(torch.sum(w2 * eta2_y1_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    eta2_y1_shift = model_eta2_y1(dummy, x_error_0, need_grad=0, normed=False) #[d,p,1]
    eta2_y1_shift = eta2_y1_shift / eta2_y1_norm
    # eta2_y1_integral_term = (torch.sum(w2 * eta2_y1_norm_phi, dim=-1) / torch.sqrt(torch.sum(w2 * eta2_y1_norm_phi**2, dim=-1))).unsqueeze(dim=-1)


lam_0 = lam[:m].detach().cpu().numpy()
eig_ms_FEM = np.empty((0, 6))
# ε=3 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM,[22.920721577215190,
92.671721037727266,
202.320838867336249,
383.897108711624355,
586.341519031419466,
839.471009526864577]))
# ε=5 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM,[22.611156540678955,
90.517251162099413,
203.538837991687984,
365.704775028756728,
539.165556593393262,
879.636282985819207]))
# ε=8 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM,[22.452571741400472,
89.909280203913269,
202.093692233041111,
358.542422819675835,
558.458595874086654,
801.803104845780695]))
# ε=10 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM,[22.400713985024286,
89.721031714657215,
201.752748724725620,
358.187673460167218,
558.471394027006227,
801.630942808510895]))
# ε=12 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM,[22.366263929218789,
89.595753720315372,
201.523006894734010,
357.934223079881519,
558.468293112219612,
802.523937834335925]))
# ε=15 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM,[22.331855046544749,
89.469623859181070,
201.285439234642070,
357.642600256422440,
558.325122120440710,
803.002851132699107]))


# 定义一系列 epsilon 值
epsilons = [3,5,8,10,12,15]
lam_0_lam_1 = np.empty((0, m))
lam_0_lam_1_lam_2 = np.empty((0, m))

l2_norm1_list = []
l2_norm2_list = []
l2_norm3_list = []

# u0_mat = sio.loadmat(f'data/func_values_0.mat')
# u0_matrix = u0_mat['func_values']  # 5000x16
# u0 = u0_matrix.reshape(-1)

# u0_grad_mat = sio.loadmat(f'data/grad_values_0.mat')
# u0_grad_matrix = u0_grad_mat['grad_values']  # 5000x16
# u0_grad = u0_grad_matrix.reshape(-1)
s = [1,-1,1,-1,1,-1]
for idx, eps in enumerate(epsilons):
    # 对于u_ep的一阶逼近
    y_error_eps = x_error * eps - (x_error * eps).floor()
    chi1_y1_phi = model_chi1_y1(dummy, y_error_eps, need_grad=0, normed=False)
    chi1_y1_phi = chi1_y1_phi / chi1_y1_norm
    chi1 = torch.sum((chi1_alpha.unsqueeze(dim=-1) * chi1_x_phi * (chi1_y1_phi - chi1_y1_shift)).squeeze(dim=0), dim=0)
    u1 = torch.einsum('n,nm->nm', chi1, u0_grad) 
    norm_ue = torch.sqrt(torch.einsum('n,nm->m', w_error, (u0 + u1/eps)**2))
    u0_new = u0 / norm_ue
    u1_new = u1 / norm_ue

    # 对于u_ep的二阶逼近
    eta1_y1_phi = model_eta1_y1(dummy, y_error_eps, need_grad=0, normed=False)
    eta1_y1_phi = eta1_y1_phi / eta1_y1_norm
    eta2_y1_phi = model_eta2_y1(dummy, y_error_eps, need_grad=0, normed=False)
    eta2_y1_phi = eta2_y1_phi / eta2_y1_norm

    eta1 = torch.sum((eta1_alpha.unsqueeze(dim=-1) * eta1_x_phi * (eta1_y1_phi-eta1_y1_shift)).squeeze(dim=0), dim=0)
    eta2 = torch.sum((eta2_alpha.unsqueeze(dim=-1) * eta2_x_phi * (eta2_y1_phi-eta2_y1_shift)).squeeze(dim=0), dim=0)
    u2 = torch.einsum('n,nm->nm', eta1, u0_grad) + torch.einsum('n,nm->nm', eta2, u0_grad_grad)
    norm_ue_2 = torch.sqrt(torch.einsum('n,nm->m', w_error, (u0 + u1/eps + u2/(eps**2))**2))
    u0_new_2 = u0 / norm_ue_2
    u1_new_2 = u1 / norm_ue_2
    u2_new_2 = u2 / norm_ue_2

    # 对于lam_ep的一阶逼近
    int_u1_u0 = torch.einsum('n,nm,nm->m', w_error, u1+w, u0)
    lam1 = lam[:m] * int_u1_u0 
    print('lam1 (eps={})='.format(eps), lam1.detach().cpu().numpy())
    lam_0_lam_1 = np.vstack((lam_0_lam_1, (lam[:m] - lam1[:m] / eps).detach().cpu().numpy()))
    
    # 对于lam_ep的二阶逼近
    int_u2_u0 = torch.einsum('n,nm,nm->m', w_error, u2, u0)  # ∫ u2 * u0 dx
    int_u1_w1 = torch.einsum('n,nm,nm->m', w_error, u1, u1+w) 
    int_u0_w1 = torch.einsum('n,nm,nm->m', w_error, u0, u1+w) 
    int_u0_u1 = torch.einsum('n,nm,nm->m', w_error, u0, u1)  # ∫ u0 * u1 dx

    lam2 = lam[:m] * (int_u2_u0 + int_u1_w1) + lam1 * (int_u0_w1 + int_u0_u1)
    print('lam2 (eps={})='.format(eps), lam2.detach().cpu().numpy())
    lam_0_lam_1_lam_2 = np.vstack((lam_0_lam_1_lam_2,(lam[:m] - lam1[:m] / eps - lam2[:m] / (eps ** 2)).detach().cpu().numpy(),))

    # 画图
    func_values_mat = sio.loadmat(f'data/func_values_{eps}.mat')
    func_values_matrix = func_values_mat['func_values']  # 5000x16
    func_values_vector = func_values_matrix.reshape(-1)  # 按行（row-major）拼为一维
    # 根据s列表中对应位置的符号来决定是否加负号
    if idx < len(s) and s[idx] == -1:
        func_values_vector = -func_values_vector

    # 画图，横坐标采样点是x_error, 只要第一个特征对，画出 u0, u1, chi1, u0+eps*u1, func_values_vector, 以及误差 func_values_vector-(u0+eps*u1)，并将图片保存输出
    fig, ax = plt.subplots(figsize=(10, 6))
    eps_value = 1 / eps  # 当前循环中的epsilon, 用于 u0+eps*u1

    # 只取第一个特征对(ground state), 即下标0
    x_plot = x_error.cpu().numpy()
    u0_plot = u0[:, 0].detach().cpu().numpy()

    u0_new_plot = u0_new[:, 0].detach().cpu().numpy()
    eta1_plot = eta1.detach().cpu().numpy()
    eta2_plot = eta2.detach().cpu().numpy()

    u1_new_plot = u1_new[:, 0].detach().cpu().numpy()
    u0_u1_plot = u0_new_plot + eps_value * u1_new_plot

    u0_new_2_plot = u0_new_2[:, 0].detach().cpu().numpy()
    u1_new_2_plot = u1_new_2[:, 0].detach().cpu().numpy()
    u2_new_2_plot = u2_new_2[:, 0].detach().cpu().numpy()
    u0_u1_u2_plot = u0_new_2_plot + eps_value * u1_new_2_plot + eps_value * eps_value * u2_new_2_plot

    func_plot = func_values_vector
    diff_plot = func_values_vector - u0_u1_plot
    diff_2_plot = func_values_vector - u0_u1_u2_plot

    ax.plot(x_plot, u0_plot, label='u0')
    ax.plot(x_plot, eta1_plot, label='eta1_plot')
    ax.plot(x_plot, eta2_plot, label='eta2_plot')
    
    ax.plot(x_plot, u0_u1_plot, label='u0 + eps*u1')
    ax.plot(x_plot, u0_u1_u2_plot, label='u0 + eps*u1 + ep**2 * u2')
    ax.plot(x_plot, func_plot, label='u_fem')
    ax.plot(x_plot, diff_plot, label='u_fem - (u0 + eps*u1)')
    ax.plot(x_plot, diff_2_plot, label='u_fem - (u0 + eps*u1+ ep**2 * u2)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_title(f'u0, u1, chi1, u0+eps*u1, u_fem, u_fem-(u0+eps*u1) (eps=1/{eps})')
    plt.tight_layout()
    plt.savefig(f'png/u0_u1_chi1_u0+epsu1_func_vector_with_err_eps{eps}_3.png')
    plt.close(fig)

    # 计算 func_values_vector-u0 和 func_values_vector-u0_plot - eps_value * u1_plot 的L2范数
    diff1 = func_values_vector - u0_plot
    diff2 = func_values_vector - (u0_new_plot + eps_value * u1_new_plot)
    diff3 = func_values_vector - (u0_new_2_plot + eps_value * u1_new_2_plot + eps_value * eps_value * u2_new_2_plot)
    w_error_np = w_error.cpu().numpy()
    l2_norm1 = np.sqrt(np.sum(w_error_np * diff1 ** 2))
    l2_norm2 = np.sqrt(np.sum(w_error_np * diff2 ** 2))
    l2_norm3 = np.sqrt(np.sum(w_error_np * diff3 ** 2))
    print(f'L2 norm of func_values_vector - u0 for eps={eps}:', l2_norm1)
    print(f'L2 norm of func_values_vector - (u0 + eps * u1) for eps={eps}:', l2_norm2)
    print(f'L2 norm of func_values_vector - (u0 + eps * u1 + ep**2 * u2) for eps={eps}:', l2_norm3)
    print('--------------------------------')
    l2_norm1_list.append(l2_norm1)
    l2_norm2_list.append(l2_norm2)
    l2_norm3_list.append(l2_norm3)

# 画L2范数收敛图
epsilons_np = np.array(epsilons)
l2_norm1_arr = np.array(l2_norm1_list)
l2_norm2_arr = np.array(l2_norm2_list)
l2_norm3_arr = np.array(l2_norm3_list)

plt.figure(figsize=(8, 6))
inv_epsilons = 1 / epsilons_np

# loglog画图
plt.loglog(inv_epsilons, l2_norm1_arr, 'o-', label=r'$\|u_\epsilon - u_0\|_{L^2}$')
plt.loglog(inv_epsilons, l2_norm2_arr, 's-', label=r'$\|u_\epsilon - (u_0 + \epsilon u_1)\|_{L^2}$')
plt.loglog(inv_epsilons, l2_norm3_arr, 's-', label=r'$\|u_\epsilon - (u_0 + \epsilon u_1 + \epsilon^2 u_2)\|_{L^2}$')

# 画参考线: O(ε) (斜率1) 和 O(ε^2) (斜率2)
# 选取两点用于画参考线
ref_x = np.array([inv_epsilons[0], inv_epsilons[-1]])
# 参考线起始点设置为对应误差的起点，幅值自适应可调：
ref1_y0 = l2_norm1_arr[0]
ref2_y0 = l2_norm2_arr[0]
ref3_y0 = l2_norm3_arr[0]
ref1 = ref1_y0 * (ref_x / ref_x[0])**1  # 斜率1
ref2 = ref2_y0 * (ref_x / ref_x[0])**2  # 斜率2
ref3 = ref2_y0 * (ref_x / ref_x[0])**3  # 斜率3
plt.loglog(ref_x, ref1, 'k--', alpha=0.7, label=r'ref: slope=1')
plt.loglog(ref_x, ref2, 'k-.', alpha=0.7, label=r'ref: slope=2')
plt.loglog(ref_x, ref3, 'k-.', alpha=0.7, label=r'ref: slope=3')

plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$L^2$ error')
plt.title(r'$L^2$ error convergence vs $\epsilon$')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig('png/l2_error_vs_epsilon.png')
plt.close()


print('lam_0_lam_1=', lam_0_lam_1)
print('lam_0_lam_1_lam_2=', lam_0_lam_1_lam_2)


if __name__ == '__main__':
    eps = [1 / x for x in epsilons]
    eps = np.array(eps)

    # print('eig_homo_TNN-eig_homo_FEM',abs(eig_homo_TNN-eig_homo_FEM))

    # eigen_error_1 = (abs(eig_ms_FEM - lam_0))
    # print('------------------')
    # print('eigen_error_1=', eigen_error_1)
    # plot_eigen_error_vs_epsilon(eps, eigen_error_1, save_path='png/eig_error_ms_FEM_lam_0.png', filename=r'error $\| \lambda_0^{TNN} - \lambda_{\epsilon}^{FEM} \|_{L^2(\Omega)}$ ~ $\epsilon$')

    # eigen_error_2 = (abs(eig_ms_FEM - lam_0_lam_1))
    # print('------------------')
    # print('eigen_error_2=', eigen_error_2)
    # plot_eigen_error_vs_epsilon(eps, eigen_error_2, save_path='png/eig_error_ms_FEM_lam0_lam1.png', filename=r'error $\| \lambda_0^{TNN} + \epsilon \lambda_1^{TNN} - \lambda_{\epsilon}^{FEM} \|_{L^2(\Omega)}$ ~ $\epsilon$')

    eigen_error_1 = (abs(eig_ms_FEM - lam_0))
    print('------------------')
    print('eigen_error_1=', eigen_error_1)
    plot_eigen_error_vs_epsilon(eps, eigen_error_1, save_path='png/eig_error_ms_FEM_lam_0.png', filename=r'error $\| \lambda_0^{TNN} - \lambda_{\epsilon}^{FEM} \|_{L^2(\Omega)}$ ~ $\epsilon$',alpha=1)

    eigen_error_2 = (abs(eig_ms_FEM - lam_0_lam_1))
    print('------------------')
    print('eigen_error_2=', eigen_error_2)
    plot_eigen_error_vs_epsilon(eps, eigen_error_2, save_path='png/eig_error_ms_FEM_lam0_lam1.png', filename=r'error $\| \lambda_0^{TNN} + \epsilon \lambda_1^{TNN} - \lambda_{\epsilon}^{FEM} \|_{L^2(\Omega)}$ ~ $\epsilon$', alpha=2)

    eigen_error_3 = (abs(eig_ms_FEM - lam_0_lam_1_lam_2))
    print('------------------')
    print('eigen_error_3=', eigen_error_3)
    plot_eigen_error_vs_epsilon(eps, eigen_error_3, save_path='png/eig_error_ms_FEM_lam0_lam1_lam2.png', filename=r'error $\| \lambda_0^{TNN} + \epsilon \lambda_1^{TNN} + \epsilon^2 \lambda_2^{TNN} - \lambda_{\epsilon}^{FEM} \|_{L^2(\Omega)}$ ~ $\epsilon$', alpha=2)

    print('Done.')

