import torch
torch.set_printoptions(precision=16, profile="full")
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io as sio
import torch.optim as optim

from tnn import *
from integration import *
from quadrature import *

import itertools
import os
import sys
import importlib.util
import gc

import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，不显示窗口，速度更快
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation

pi = 3.14159265358979323846
dtype = torch.double
# device = 'cpu'
device = 'cuda:5'
name = 'd2_ex3'

dim = 2

# p_A = 3
# p_chi = 20
# p_u0 = 20
# p_u1 = 20
# p_A0 = p_A + p_A * p_chi

a, b = 0, 1

x1, w1 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
N = len(x1)
x2, w2 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
x3, w3 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
x4, w4 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)

w1234 = torch.stack((w1, w2, w3, w4), dim=0)
w12 = torch.stack((w1, w2), dim=0)
w34 = torch.stack((w3, w4), dim=0)

# dummy variable, useless
dummy = torch.ones((1, 1), dtype=dtype, device=device)
x_error_0 = torch.zeros(1, dtype=dtype, device=device)


model_u0 = torch.load("./model_u0/d2_ex3_final_u0.pkl", map_location=device)
model_u1_x = torch.load('./model_u1/d2_ex3_final_u1_x.pkl', map_location=device)
model_u1_y = torch.load('./model_u1/d2_ex3_final_u1_y.pkl', map_location=device)
model_u2_x = torch.load('./model_u2/d2_ex3_final_u2_x.pkl', map_location=device)
model_u2_y = torch.load('./model_u2/d2_ex3_final_u2_y.pkl', map_location=device)

load_tilde_u1 = True
if load_tilde_u1:
    model_tilde_u1 = torch.load('./model_tilde_u1/d2_ex3_final_tilde_u1.pkl', map_location=device)
load_delta_u2 = True
if load_delta_u2:
    model_delta_u2_x = torch.load('./model_delta_u2/d2_ex3_final_delta_u2_x.pkl', map_location=device)
    model_delta_u2_y = torch.load('./model_delta_u2/d2_ex3_final_delta_u2_y.pkl', map_location=device)
load_u3 = True
if load_u3:
    model_u3_x = torch.load('./model_u3/d2_ex3_final_u3_x.pkl', map_location=device)
    model_u3_y = torch.load('./model_u3/d2_ex3_final_u3_y.pkl', map_location=device)
load_tilde_u2 = True
if load_tilde_u2:
    model_tilde_u2 = torch.load('./model_tilde_u2/d2_ex3_final_tilde_u2.pkl', map_location=device)

with torch.no_grad():
    u0_norm_phi = model_u0(w1, x1, need_grad=0, normed=False)
    u0_norm = torch.sqrt(torch.sum(w1 * u0_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u0_alpha = model_u0.scaling_par()
    u0_shift = model_u0(dummy, x_error_0, need_grad=0, normed=False)
    u0_shift = u0_shift / u0_norm

    u1_x_norm_phi = model_u1_x(w1, x1, need_grad=0, normed=False)
    u1_x_norm = torch.sqrt(torch.sum(w1 * u1_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u1_alpha = model_u1_x.scaling_par()
    u1_y_norm_phi = model_u1_y(w3, x3, need_grad=0, normed=False)
    u1_y_norm = torch.sqrt(torch.sum(w3 * u1_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u1_y_shift = model_u1_y(dummy, x_error_0, need_grad=0, normed=False)
    u1_y_shift = u1_y_shift / u1_y_norm

    u2_x_norm_phi = model_u2_x(w1, x1, need_grad=0, normed=False)
    u2_x_norm = torch.sqrt(torch.sum(w1 * u2_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u2_alpha = model_u2_x.scaling_par()
    u2_y_norm_phi = model_u2_y(w3, x3, need_grad=0, normed=False)
    u2_y_norm = torch.sqrt(torch.sum(w3 * u2_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u2_y_shift = model_u2_y(dummy, x_error_0, need_grad=0, normed=False)
    u2_y_shift = u2_y_shift / u2_y_norm

    # 计算其他不依赖于rank的norm
    if load_tilde_u1:
        tilde_u1_norm_phi = model_tilde_u1(w3, x3, need_grad=0, normed=False)
        tilde_u1_norm = torch.sqrt(torch.sum(w3 * tilde_u1_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        tilde_u1_alpha = model_tilde_u1.scaling_par()
        tilde_u1_shift = model_tilde_u1(dummy, x_error_0, need_grad=0, normed=False)
        tilde_u1_shift = tilde_u1_shift / tilde_u1_norm
    
    if load_delta_u2:
        delta_u2_x_norm_phi = model_delta_u2_x(w1, x1, need_grad=0, normed=False)
        delta_u2_x_norm = torch.sqrt(torch.sum(w1 * delta_u2_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        delta_u2_alpha = model_delta_u2_x.scaling_par()
        delta_u2_y_norm_phi = model_delta_u2_y(w3, x3, need_grad=0, normed=False)
        delta_u2_y_norm = torch.sqrt(torch.sum(w3 * delta_u2_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        delta_u2_y_shift = model_delta_u2_y(dummy, x_error_0, need_grad=0, normed=False)
        delta_u2_y_shift = delta_u2_y_shift / delta_u2_y_norm

    if load_u3:
        u3_x_norm_phi = model_u3_x(w1, x1, need_grad=0, normed=False)
        u3_x_norm = torch.sqrt(torch.sum(w1 * u3_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        u3_alpha = model_u3_x.scaling_par()
        u3_y_norm_phi = model_u3_y(w3, x3, need_grad=0, normed=False)
        u3_y_norm = torch.sqrt(torch.sum(w3 * u3_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        u3_y_shift = model_u3_y(dummy, x_error_0, need_grad=0, normed=False)
        u3_y_shift = u3_y_shift / u3_y_norm

    if load_tilde_u2:
        tilde_u2_norm_phi = model_tilde_u2(w1, x1, need_grad=0, normed=False)
        tilde_u2_norm = torch.sqrt(torch.sum(w1 * tilde_u2_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        tilde_u2_alpha = model_tilde_u2.scaling_par()
        tilde_u2_shift = model_tilde_u2(dummy, x_error_0, need_grad=0, normed=False)
        tilde_u2_shift = tilde_u2_shift / tilde_u2_norm

    inv_e = [3,5,8,10,12,15]
    error_e = torch.zeros((len(inv_e)), dtype=dtype, device=device)
    l2_error1_result_eps = torch.zeros(len(inv_e), dtype=dtype, device=device)
    l2_error2_result_eps = torch.zeros(len(inv_e), dtype=dtype, device=device)
    l2_error3_result_eps = torch.zeros(len(inv_e), dtype=dtype, device=device)
    l2_error0_result_eps = torch.zeros(len(inv_e), dtype=dtype, device=device)
    norm_ue_fem = torch.zeros(len(inv_e), dtype=dtype, device=device)
    
    # 用于保存绘图数据
    all_coords = []
    all_u0_values = []  # u0不依赖于epsilon，只保存一次
    all_u1_values = {}  # u1依赖于epsilon，为每个epsilon保存
    all_u2_values = {}  # u2依赖于epsilon，为每个epsilon保存
    all_u3_values = {}  # u2依赖于epsilon，为每个epsilon保存

    print("\n计算误差...")
    for i_rank in range(0,4):
        coord_x_np = np.loadtxt(f'./periodic_P2_R8/quad/coord_R8_rank{i_rank}.txt', unpack=True)
        coord_x = torch.from_numpy(coord_x_np).to(device)
        all_coords.append(coord_x)

        print(coord_x.shape)
        print(type(coord_x))
        
        
        w_np = np.loadtxt(f'./periodic_P2_R8/quad/w_R8_rank{i_rank}.txt', unpack=True)
        w = torch.from_numpy(w_np).to(device)

        u0_phi, u0_grad_phi = model_u0(dummy, coord_x, need_grad=1, normed=False)
        u0_phi = u0_phi / u0_norm
        u0_grad_phi = u0_grad_phi / u0_norm

        tnn_value_u0 = torch.sum(u0_alpha.unsqueeze(dim=-1) * (torch.prod(u0_phi, dim=0)-torch.prod(u0_shift, dim=0)), dim=0)
        all_u0_values.append(tnn_value_u0.cpu().numpy())

        u1_x_phi = model_u1_x(dummy, coord_x, need_grad=0, normed=False)
        u1_x_phi = u1_x_phi / u1_x_norm
        
        u2_x_phi = model_u2_x(dummy, coord_x, need_grad=0, normed=False)
        u2_x_phi = u2_x_phi / u2_x_norm
        
        if load_tilde_u1:
            tilde_u1_phi = model_tilde_u1(dummy, coord_x, need_grad=0, normed=False)
            tilde_u1_phi = tilde_u1_phi / tilde_u1_norm
            tnn_value_tilde_u1 = torch.sum(tilde_u1_alpha.unsqueeze(dim=-1) * (torch.prod(tilde_u1_phi, dim=0) - torch.prod(tilde_u1_shift, dim=0)), dim=0)
            
        if load_delta_u2:
            delta_u2_x_phi = model_delta_u2_x(dummy, coord_x, need_grad=0, normed=False)
            delta_u2_x_phi = delta_u2_x_phi / delta_u2_x_norm
        
        if load_u3:
            u3_x_phi = model_u3_x(dummy, coord_x, need_grad=0, normed=False)
            u3_x_phi = u3_x_phi / u3_x_norm

        if load_tilde_u2:
            tilde_u2_phi = model_tilde_u2(dummy, coord_x, need_grad=0, normed=False)
            tilde_u2_phi = tilde_u2_phi / tilde_u2_norm
            tnn_value_tilde_u2 = torch.sum(tilde_u2_alpha.unsqueeze(dim=-1) * (torch.prod(tilde_u2_phi, dim=0) - torch.prod(tilde_u2_shift, dim=0)), dim=0)
        
        for i_e in range(0, len(inv_e)):
            coord_y = coord_x * inv_e[i_e] - torch.floor(coord_x * inv_e[i_e])
            
            u1_y_phi = model_u1_y(dummy, coord_y, need_grad=0, normed=False)
            u1_y_phi = u1_y_phi / u1_y_norm
            tnn_value_u1 = torch.sum(u1_alpha.unsqueeze(dim=-1) * torch.prod(u1_x_phi, dim=0) * (torch.prod(u1_y_phi, dim=0) - torch.prod(u1_y_shift, dim=0)), dim=0)
            del u1_y_phi

            u2_y_phi = model_u2_y(dummy, coord_y, need_grad=0, normed=False)
            u2_y_phi = u2_y_phi / u2_y_norm
            tnn_value_u2 = torch.sum(u2_alpha.unsqueeze(dim=-1) * torch.prod(u2_x_phi, dim=0) * (torch.prod(u2_y_phi, dim=0) - torch.prod(u2_y_shift, dim=0)), dim=0)
            del u2_y_phi

            u3_y_phi = model_u3_y(dummy, coord_y, need_grad=0, normed=False)
            u3_y_phi = u3_y_phi / u3_y_norm
            tnn_value_u3 = torch.sum(u3_alpha.unsqueeze(dim=-1) * torch.prod(u3_x_phi, dim=0) * (torch.prod(u3_y_phi, dim=0) - torch.prod(u3_y_shift, dim=0)), dim=0)
            del u3_y_phi

            if load_delta_u2:
                delta_u2_y_phi = model_delta_u2_y(dummy, coord_y, need_grad=0, normed=False)
                delta_u2_y_phi = delta_u2_y_phi / delta_u2_y_norm
                tnn_value_delta_u2 = torch.sum(delta_u2_alpha.unsqueeze(dim=-1) * torch.prod(delta_u2_x_phi, dim=0) * (torch.prod(delta_u2_y_phi, dim=0) - torch.prod(delta_u2_y_shift, dim=0)), dim=0)
                del delta_u2_y_phi

            # 计算完整解（归一化）
            tnn_value0 = tnn_value_u0
            tnn_value1 = tnn_value_u0 + (tnn_value_u1 + tnn_value_tilde_u1) / inv_e[i_e]
            tnn_value2 = tnn_value_u0 + (tnn_value_u1 + tnn_value_tilde_u1) / inv_e[i_e] + (tnn_value_u2 + tnn_value_delta_u2 + tnn_value_tilde_u2) / ((inv_e[i_e])**2)
            tnn_value3 = tnn_value_u0 + (tnn_value_u1 + tnn_value_tilde_u1) / inv_e[i_e] + (tnn_value_u2 + tnn_value_delta_u2 + tnn_value_tilde_u2) / ((inv_e[i_e])**2) + tnn_value_u3 / ((inv_e[i_e])**3)

            # 加载FEM解并归一化
            fem_value_np = np.loadtxt(f'./periodic_P2_R8/e_{inv_e[i_e]}/u_fem_R8_rank{i_rank}.txt', unpack=True)
            fem_value = torch.from_numpy(fem_value_np).to(device)
            fem_value_normalized = fem_value 
            norm_ue_fem[i_e] += torch.sum(w * fem_value_normalized**2)

            # 保存用于绘图（每个epsilon分别保存）
            if i_e not in all_u1_values:
                all_u1_values[i_e] = []
            if i_e not in all_u2_values:
                all_u2_values[i_e] = []
            if i_e not in all_u3_values:
                all_u3_values[i_e] = []  
            all_u1_values[i_e].append((tnn_value_u1).cpu().numpy())
            all_u2_values[i_e].append((tnn_value_u2).cpu().numpy())
            all_u3_values[i_e].append(tnn_value_u3.cpu().numpy())

            # 计算误差（使用归一化后的解）
            current_error0 = torch.sum(w * (fem_value_normalized - tnn_value0)**2)
            l2_error0_result_eps[i_e] += current_error0

            current_error1 = torch.sum(w * (fem_value_normalized - tnn_value1)**2)
            l2_error1_result_eps[i_e] += current_error1
            print('l2 error square of 1 / %d from process rank %d : ' % (inv_e[i_e], i_rank), current_error1.item())

            current_error2 = torch.sum(w * (fem_value_normalized - tnn_value2)**2)
            l2_error2_result_eps[i_e] += current_error2
            print('l2 error square of 2 / %d from process rank %d : ' % (inv_e[i_e], i_rank), current_error2.item())

            current_error3 = torch.sum(w * (fem_value_normalized - tnn_value3)**2)
            l2_error3_result_eps[i_e] += current_error3
            print('l2 error square of 3 / %d from process rank %d : ' % (inv_e[i_e], i_rank), current_error3.item())

l2_error0_result_eps = torch.sqrt(l2_error0_result_eps)
print('inv_eps : ', inv_e)
print('l2 error0: ', l2_error0_result_eps)

l2_error1_result_eps = torch.sqrt(l2_error1_result_eps)
print('l2 error1: ', l2_error1_result_eps)

l2_error2_result_eps = torch.sqrt(l2_error2_result_eps)
print('l2 error2: ', l2_error2_result_eps)

l2_error3_result_eps = torch.sqrt(l2_error3_result_eps)
print('l2 error3: ', l2_error3_result_eps)

# print('norm_ue_fem: ', torch.sqrt(norm_ue_fem))


import matplotlib.pyplot as plt
import os

# 创建png文件夹如果不存在的话
if not os.path.exists('./png'):
    os.makedirs('./png')

# 横坐标为 epsilon = 1 / inv_e
epsilons = 1.0 / np.array(inv_e)

# 画l2误差随epsilon减小的loglog图
plt.figure(figsize=(8, 6))
plt.loglog(epsilons, l2_error0_result_eps.cpu().numpy(), marker='o', label=r'$||u_\varepsilon - u_0||_{L^2}$')
plt.loglog(epsilons, l2_error1_result_eps.cpu().numpy(), marker='o', label=r'$||u_\varepsilon - (u_0 + \varepsilon * (u_1+\tilde{u}_1))||_{L^2}$')
plt.loglog(epsilons, l2_error2_result_eps.cpu().numpy(), marker='o', label=r'$||u_\varepsilon - (u_0 + \varepsilon * (u_1+\tilde{u}_1) + \varepsilon^2 * (u_2 + \delta u_2 + \tilde{u}_2))||_{L^2}$')
plt.loglog(epsilons, l2_error3_result_eps.cpu().numpy(), marker='o', label=r'$||u_\varepsilon - (u_0 + \varepsilon * (u_1+\tilde{u}_1) + \varepsilon^2 * (u_2 + \delta u_2 + \tilde{u}_2) + \varepsilon^3 * (u_3))||_{L^2}$')

# 画参考线：O(ε) 和 O(ε^2)
ref_x = np.array([epsilons[0], epsilons[-1]])
ref1_y0 = l2_error0_result_eps.cpu().numpy()[0]
ref2_y0 = l2_error1_result_eps.cpu().numpy()[0]
ref3_y0 = l2_error2_result_eps.cpu().numpy()[0]
ref1 = ref1_y0 * (ref_x / ref_x[0])**1     # 斜率1
ref2 = ref2_y0 * (ref_x / ref_x[0])**2     # 斜率2
ref3 = ref3_y0 * (ref_x / ref_x[0])**3     # 斜率3
plt.loglog(ref_x, ref1, 'k--', alpha=0.7)# , label=r'ref: slope=1'
# plt.loglog(ref_x, ref2_hat, 'k--', alpha=0.7)# , label=r'ref: slope=2_hat')
plt.loglog(ref_x, ref2, 'k-.', alpha=0.7)# , label=r'ref: slope=2')
# plt.loglog(ref_x, ref3_hat, 'k-.', alpha=0.7)# , label=r'ref: slope=3_hat')
plt.loglog(ref_x, ref3, 'k:', alpha=0.7)# , label=r'ref: slope=3')
plt.xlabel('epsilon')
plt.ylabel('L2 Error')
plt.title(r'L2 Error vs $\epsilon$ (log-log)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig("./png/data_error_vs_epsilon_loglog_txt.png")
plt.show()




# # 创建png文件夹如果不存在的话
# if not os.path.exists('./png'):
#     os.makedirs('./png')
# # ------ 改进版：预读数据加速+避免反复从硬盘load -------


# num_ranks = 4

# # 合并所有rank的数据用于绘图（不需要每次合并，提前一次即可）
# with torch.no_grad():
#     all_coords_merged = np.concatenate(
#         [c.cpu().numpy() if isinstance(c, torch.Tensor) else c for c in all_coords], axis=1
#     )
#     u0_merged = np.concatenate([u.cpu().numpy() if isinstance(u, torch.Tensor) else u for u in all_u0_values])
#     u_full_all_eps = {}
#     for i_e in range(0, len(inv_e)):
#         u1_merged = np.concatenate(all_u1_values[i_e])
#         u2_merged = np.concatenate(all_u2_values[i_e])
#         u3_merged = np.concatenate(all_u3_values[i_e])
#         u_full_all_eps[i_e] = u3_merged

# x1_merge = all_coords_merged[0, :]
# x2_merge = all_coords_merged[1, :]

# n_rows, n_cols = 3, 3
# fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 12), constrained_layout=True)

# for i_e, inv_eps in enumerate(inv_e):
#     epsilon = 1.0 / inv_eps

    
#     fem_u_ranks = []
#     for i_rank in range(num_ranks):
#         fname = f'./periodic_P2_R8/e_{inv_e[i_e]}/u_fem_R8_rank{i_rank}.txt'
#         fem_value_np = np.loadtxt(fname, unpack=True)
#         fem_u_ranks.append(fem_value_np)
#     fem_u_merge = np.concatenate(fem_u_ranks)
#     # 主动释放临时变量
#     del fem_u_ranks
#     gc.collect()

#     tnn_u_merge = u_full_all_eps[i_e]
#     diff = tnn_u_merge

#     tri = Triangulation(x1_merge, x2_merge)
#     row = i_e // n_cols
#     col = i_e % n_cols
#     ax = axs[row, col]
#     contour = ax.tricontourf(tri, diff, levels=20, cmap='RdBu')
#     cbar = fig.colorbar(contour, ax=ax, shrink=0.8, aspect=15, pad=0.02)
#     ax.set_xlabel('$x_1$', fontsize=12)
#     ax.set_ylabel('$x_2$', fontsize=12)
#     ax.set_title(f'hat u_2 ($\\epsilon={epsilon:.4g}$)', fontsize=16)
#     ax.set_aspect('equal')

#     # 主动删除当前有限元解以释放内存
#     del fem_u_merge
#     gc.collect()

# for j in range(len(inv_e), n_rows * n_cols):
#     row = j // n_cols
#     col = j % n_cols
#     fig.delaxes(axs[row, col])

# plt.suptitle(r"Finite element solution minus TNN solution (for different $\epsilon$) two-dimensional contour plot", fontsize=20)
# plt.savefig('./png/u2_二维_多epsilon_子图_nobd.png', dpi=150, bbox_inches='tight')
# plt.close(fig)
# print('  ✓ 已保存: ./png/FEM减TNN_二维等高线图.png')


import numpy as np

# 计算误差曲线在 log-log 坐标下的斜率

# 横坐标 epsilon, 误差序列分别为 l2_error0_result_eps, l2_error1_result_eps, l2_error2_result_eps, l2_error3_result_eps
# 先转为 numpy 数组
eps_np = np.array(epsilons)
err0 = l2_error0_result_eps.cpu().numpy()
err1 = l2_error1_result_eps.cpu().numpy()
err2 = l2_error2_result_eps.cpu().numpy()
err3 = l2_error3_result_eps.cpu().numpy()

log_eps = np.log(eps_np)
log_err0 = np.log(err0)
log_err1 = np.log(err1)
log_err2 = np.log(err2)
log_err3 = np.log(err3)

slope0, intercept0 = np.polyfit(log_eps, log_err0, 1)
slope1, intercept1 = np.polyfit(log_eps, log_err1, 1)
slope2, intercept2 = np.polyfit(log_eps, log_err2, 1)
slope3, intercept3 = np.polyfit(log_eps, log_err3, 1)

print(f"L2 error0 (||u_eps - u0||_L2) log-log slope: {slope0:.4f}")
print(f"L2 error1 (||u_eps - (u0+eps*(u1+~u1))||_L2) log-log slope: {slope1:.4f}")
print(f"L2 error2 (||u_eps - (u0+eps*(u1+~u1)+eps^2*(u2+delta_u2+~u2))||_L2) log-log slope: {slope2:.4f}")
print(f"L2 error3 (||u_eps - (... + eps^3*u3)||_L2) log-log slope: {slope3:.4f}")