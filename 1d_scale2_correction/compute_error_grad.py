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
import gc
import numpy as np

# 内存监测函数
def print_mem(prefix=""):
    # CUDA
    if torch.cuda.is_available():
        print(f"{prefix} [CUDA] mem_allocated: {torch.cuda.memory_allocated()//1024**2} MB | mem_reserved: {torch.cuda.memory_reserved()//1024**2} MB | max_allocated: {torch.cuda.max_memory_allocated()//1024**2} MB | max_reserved: {torch.cuda.max_memory_reserved()//1024**2} MB")
    # 系统 RAM
    try:
        import psutil
        print(f"{prefix} [RAM] total: {psutil.virtual_memory().total//1024**2} MB | used: {psutil.virtual_memory().used//1024**2} MB | available: {psutil.virtual_memory().available//1024**2} MB")
    except Exception:
        pass

pi = 3.14159265358979323846
dtype = torch.double
device = 'cuda:0'

print_mem("程序启动后")

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
print_mem("加载 model_u0 后")
# model_u0 = torch.load('./model_u0/ex4_3_1_final_u0.pkl', map_location=device)

x1, w1 = composite_quadrature_1d(8, 0, 1, 200, device=device)
x2, w2 = composite_quadrature_1d(8, 0, 1, 200, device=device)

print_mem("生成 x1, w1, x2, w2 后")

w12 = torch.stack((w1, w2), dim=0)
N = len(x1)

dummy = torch.ones((1, 1), dtype=dtype, device=device)

x_error, w_error = composite_quadrature_1d(4, 0, 1, 10000, device=device, dtype=dtype)
N_error = len(x_error)

x_error_0 = torch.zeros(1, dtype=dtype, device=device)

print_mem("生成误差区间后")

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
print_mem("所有模型加载完毕后")

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

print_mem("chi1 和 A0 准备好后")

# 计算\int A0_TNN (\nabla u)^2 dx =\lambda0 \int u^2 dx 对应的lambda0
def criterion_0(model_u0):
    print_mem("进入criterion_0")
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

    print_mem("criterion_0 退出")
    return lam1, u1

lam, u = criterion_0(model_u0)
print('lam[:m]=', lam[:m])

gc.collect()
torch.cuda.empty_cache()
print_mem("计算初始lam, u之后")

with torch.no_grad():
    u0_norm_phi = model_u0(w1, x1, normed=False)
    u0_norm = torch.sqrt(torch.sum(w1 * u0_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u0_phi, u0_grad_phi, u0_grad_grad_phi = model_u0(dummy, x_error, need_grad=2, normed=False)
    u0_phi = u0_phi  / u0_norm
    u0_grad_phi = u0_grad_phi / u0_norm
    u0_grad_grad_phi = u0_grad_grad_phi / u0_norm

    u0 = torch.sum(torch.einsum('pm,pn->pnm',u[:,:m], u0_phi.squeeze(dim=0)),dim=0)
    u0_grad = torch.sum(torch.einsum('pm,pn->pnm',u[:,:m], u0_grad_phi.squeeze(dim=0)),dim=0)
    u0_grad_grad = torch.sum(torch.einsum('pm,pn->pnm',u[:,:m], u0_grad_grad_phi.squeeze(dim=0)),dim=0)

    chi1_x_norm_phi = model_chi1_x(w1, x1, normed=False)
    chi1_x_norm = torch.sqrt(torch.sum(w1 * chi1_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    chi1_x_phi, chi1_x_grad_phi = model_chi1_x(dummy, x_error, need_grad=1, normed=False)
    chi1_x_phi = chi1_x_phi / chi1_x_norm
    chi1_x_grad_phi = chi1_x_grad_phi / chi1_x_norm
    chi1_alpha = model_chi1_x.scaling_par()

    chi1_y1_norm_phi = model_chi1_y1(w2, x2, normed=False)
    chi1_y1_norm = torch.sqrt(torch.sum(w2 * chi1_y1_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    chi1_y1_shift = model_chi1_y1(dummy, x_error_0, need_grad=0, normed=False) #[d,p,1]
    chi1_y1_shift = chi1_y1_shift / chi1_y1_norm
    # chi1_y1_integral_term = (torch.sum(w2 * chi1_y1_norm_phi, dim=-1) / torch.sqrt(torch.sum(w2 * chi1_y1_norm_phi**2, dim=-1))).unsqueeze(dim=-1)

    w_norm_phi = model_w(w1, x1, normed=False)
    w_norm = torch.sqrt(torch.sum(w1 * w_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    w_phi, w_grad_phi = model_w(dummy, x_error, need_grad=1, normed=False)
    w_phi = w_phi / w_norm
    w_grad_phi = w_grad_phi / w_norm
    w_alpha = model_w.scaling_par()
    w = torch.sum(((w_phi.squeeze(dim=0)) * w_alpha.unsqueeze(dim=-1)).unsqueeze(dim=-1), dim=0)
    w_grad = torch.sum(((w_grad_phi.squeeze(dim=0)) * w_alpha.unsqueeze(dim=-1)).unsqueeze(dim=-1), dim=0)

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

print_mem("所有phi和norm等准备好后")

lam_0 = lam[:m].detach().cpu().numpy()

# 定义一系列 epsilon 值
epsilons = [3,5,8,10,12]
lam_0_lam_1 = np.empty((0, m))
lam_0_lam_1_lam_2 = np.empty((0, m))

l2_norm1_list = []
l2_norm2_list = []
l2_norm3_list = []

s = [1,-1,1,-1,1,-1]
for idx, eps in enumerate(epsilons):
    gc.collect()
    torch.cuda.empty_cache()
    print_mem(f"\n开始循环 idx={idx} eps={eps}")

    # 对于u_ep的一阶逼近
    y_error_eps = x_error * eps - (x_error * eps).floor()
    chi1_y1_phi, chi1_y1_grad_phi = model_chi1_y1(dummy, y_error_eps, need_grad=1, normed=False)
    chi1_y1_phi = chi1_y1_phi / chi1_y1_norm
    chi1_y1_grad_phi = chi1_y1_grad_phi / chi1_y1_norm
    chi1 = torch.sum((chi1_alpha.unsqueeze(dim=-1) * chi1_x_phi * (chi1_y1_phi - chi1_y1_shift)).squeeze(dim=0), dim=0)
    chi1_x = torch.sum((chi1_alpha.unsqueeze(dim=-1) * chi1_x_grad_phi * (chi1_y1_phi - chi1_y1_shift)).squeeze(dim=0), dim=0)
    chi1_y = torch.sum((chi1_alpha.unsqueeze(dim=-1) * chi1_x_phi * chi1_y1_grad_phi).squeeze(dim=0), dim=0)
    
    grad_one_order = torch.einsum('n,nm->nm', chi1_y + 1, u0_grad) 
    u1 = torch.einsum('n,nm->nm', chi1, u0_grad) 
    norm_ue = torch.sqrt(torch.einsum('n,nm->m', w_error, (u0 + u1/eps)**2))
    u0_new = u0
    u1_new = u1
    grad_one_order_new = grad_one_order
    # u0_new = u0 / norm_ue
    # u1_new = u1 / norm_ue
    # grad_one_order_new = grad_one_order / norm_ue

    # 对于u_ep的二阶逼近
    eta1_y1_phi, eta1_y1_grad_phi = model_eta1_y1(dummy, y_error_eps, need_grad=1, normed=False)
    eta1_y1_phi = eta1_y1_phi / eta1_y1_norm
    eta1_y1_grad_phi = eta1_y1_grad_phi / eta1_y1_norm
    
    eta2_y1_phi, eta2_y1_grad_phi = model_eta2_y1(dummy, y_error_eps, need_grad=1, normed=False)
    eta2_y1_phi = eta2_y1_phi / eta2_y1_norm
    eta2_y1_grad_phi = eta2_y1_grad_phi / eta2_y1_norm

    eta1 = torch.sum((eta1_alpha.unsqueeze(dim=-1) * eta1_x_phi * (eta1_y1_phi-eta1_y1_shift)).squeeze(dim=0), dim=0)
    eta2 = torch.sum((eta2_alpha.unsqueeze(dim=-1) * eta2_x_phi * (eta2_y1_phi-eta2_y1_shift)).squeeze(dim=0), dim=0)
    eta1_y = torch.sum((eta1_alpha.unsqueeze(dim=-1) * eta1_x_phi * eta1_y1_grad_phi).squeeze(dim=0), dim=0)
    eta2_y = torch.sum((eta2_alpha.unsqueeze(dim=-1) * eta2_x_phi * eta2_y1_grad_phi).squeeze(dim=0), dim=0)
    u2 = torch.einsum('n,nm->nm', eta1, u0_grad) + torch.einsum('n,nm->nm', eta2, u0_grad_grad)
    norm_ue_2 = torch.sqrt(torch.einsum('n,nm->m', w_error, (u0 + u1/eps + u2/(eps**2))**2))
    # 优化内存：分步计算，避免创建大型中间张量
    # 计算第一部分
    chi1_x_eta1_y = chi1_x + eta1_y  # (N_error,)
    part1 = torch.einsum('n,nm->nm', chi1_x_eta1_y, u0_grad)  # (N_error, m)
    del chi1_x_eta1_y  # 立即释放内存
    # 计算第二部分
    chi1_eta2_y = chi1 + eta2_y  # (N_error,)
    part2 = torch.einsum('n,nm->nm', chi1_eta2_y, u0_grad_grad)  # (N_error, m)
    del chi1_eta2_y  # 立即释放内存
    # 计算第三部分：确保w_grad形状正确
    if w_grad.dim() == 1:
        w_grad = w_grad.unsqueeze(dim=-1)  # (N_error,) -> (N_error, 1)
    eta1_y_plus_1 = chi1_y + 1  # (N_error,)
    part3 = (eta1_y_plus_1.unsqueeze(dim=-1) * w_grad)  # (N_error, 1)
    del eta1_y_plus_1  # 立即释放内存
    # 合并结果
    grad_secd_order = part1 + part2 + part3
    del part1, part2, part3  # 释放中间结果
    grad_secd_order_new = grad_secd_order
    # u0_new_2 = u0 / norm_ue_2
    # u1_new_2 = u1 / norm_ue_2
    # u2_new_2 = u2 / norm_ue_2
    # grad_secd_order_new = grad_secd_order / norm_ue_2

    print_mem(f"eps={eps} 所有新分量生成后")
    
    func_values_mat = sio.loadmat(f'data_Axy_plus/grad_values_{eps}.mat')
    func_values_matrix = func_values_mat['grad_values']  # 5000x16
    func_values_vector = func_values_matrix.reshape(-1)  # 按行（row-major）拼为一维
    # 根据s列表中对应位置的符号来决定是否加负号
    if idx < len(s) and s[idx] == -1:
        func_values_vector = -func_values_vector

    # 计算 func_values_vector-u0 和 func_values_vector-u0_plot - eps_value * u1_plot 的L2范数
    # 注意原来少定义了 eps_value (原本是 for idx, eps in enumerate ...)
    # eps_value = 1/eps
    # 将 Tensor 转换为 numpy 数组并展平，确保形状匹配
    grad_one_order_new_np = grad_one_order_new.detach().cpu().numpy().flatten()
    grad_secd_order_new_np = grad_secd_order_new.detach().cpu().numpy().flatten()
    diff1 = func_values_vector - grad_one_order_new_np
    diff2 = func_values_vector - (grad_one_order_new_np + grad_secd_order_new_np / eps)
    w_error_np = w_error.cpu().numpy()
    l2_norm1 = np.sqrt(np.sum(w_error_np * diff1 ** 2))
    l2_norm2 = np.sqrt(np.sum(w_error_np * diff2 ** 2))
    print(f'L2 norm of func_values_vector - u0 for eps={eps}:', l2_norm1)
    print(f'L2 norm of func_values_vector - (u0 + eps * u1) for eps={eps}:', l2_norm2)
    print('--------------------------------')
    print_mem(f"eps={eps} 计算L2后")
    l2_norm1_list.append(l2_norm1)
    l2_norm2_list.append(l2_norm2)
    gc.collect()
    torch.cuda.empty_cache()
    print_mem(f"eps={eps} 循环尾")

# 画L2范数收敛图
epsilons_np = np.array(epsilons)
l2_norm1_arr = np.array(l2_norm1_list)
l2_norm2_arr = np.array(l2_norm2_list)

plt.figure(figsize=(8, 6))
inv_epsilons = 1 / epsilons_np

# loglog画图
plt.loglog(inv_epsilons, l2_norm1_arr, 'o-', label=r'$\|\nabla u_\epsilon - (\nabla_x u_0-\nabla_y u_1)\|_{L^2}$')
plt.loglog(inv_epsilons, l2_norm2_arr, 's-', label=r'$\|\nabla u_\epsilon - (\nabla_x u_0-\nabla_y u_1) - e * (\nabla_x u_1 - \nabla_y u_2)\|_{L^2}$')

# 画参考线: O(ε) (斜率1) 和 O(ε^2) (斜率2)
# 选取两点用于画参考线
ref_x = np.array([inv_epsilons[0], inv_epsilons[-1]])
# 参考线起始点设置为对应误差的起点，幅值自适应可调：
ref1_y0 = l2_norm1_arr[0]
ref2_y0 = l2_norm2_arr[0]
ref1 = ref1_y0 * (ref_x / ref_x[0])**1  # 斜率1
ref2 = ref2_y0 * (ref_x / ref_x[0])**2  # 斜率2
ref3 = ref2_y0 * (ref_x / ref_x[0])**3  # 斜率3
plt.loglog(ref_x, ref1, 'k--', alpha=0.7, label=r'ref: slope=1')
plt.loglog(ref_x, ref2, 'k-.', alpha=0.7, label=r'ref: slope=2')

plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$H^1$ semi error')
plt.title(r'$H^1$ error convergence vs $\epsilon$')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig('png/h1_semi_error_vs_epsilon.png')
plt.close()
print_mem("绘图完成后")


if __name__ == '__main__':
    eps = [1 / x for x in epsilons]
    eps = np.array(eps)

    print('Done.')

    gc.collect()
    torch.cuda.empty_cache()
    print_mem("主程序结尾")
