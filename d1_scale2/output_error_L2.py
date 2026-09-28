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
device = 'cuda:3'

dim = 1

p_A = 2
p_chi = 20
p_u0 = 20
p_u1 = 20
p_A0 = p_A + p_A * p_chi

a, b = 0, 1

x1, w1 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
N = len(x1)
x2, w2 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)

w12 = torch.stack((w1, w2), dim=0)

# dummy variable, useless
dummy = torch.ones((1, 1), dtype=dtype, device=device)
x_error_0 = torch.zeros(1, dtype=dtype, device=device)


def bd_u0(x):
    return (x - a) * (b - x)

def grad_bd_u0(x):
    return a + b - 2 * x

def grad_grad_bd_u0(x):
    return -2 * torch.ones_like(x)

def grad_grad_grad_bd_u0(x):
    # return -pi * pi * pi * torch.cos(pi * x)
    return 0 * torch.ones_like(x)

model_u0 = torch.load("./model_u0/d1_ex1_final_u0.pkl", map_location=device)
model_u1_x = torch.load('./model_u1/d1_ex1_final_u1_x.pkl', map_location=device)
model_u1_y = torch.load('./model_u1/d1_ex1_final_u1_y.pkl', map_location=device)
model_u2_x = torch.load('./model_u2/d1_ex1_final_u2_x.pkl', map_location=device)
model_u2_y = torch.load('./model_u2/d1_ex1_final_u2_y.pkl', map_location=device)

load_tilde_u1 = True
if load_tilde_u1:
    model_tilde_u1 = torch.load('./model_tilde_u1/d1_ex1_final_tilde_u1.pkl', map_location=device)
load_delta_u2 = True
if load_delta_u2:
    model_delta_u2_x = torch.load('./model_delta_u2/d1_ex1_final_delta_u2_x.pkl', map_location=device)
    model_delta_u2_y = torch.load('./model_delta_u2/d1_ex1_final_delta_u2_y.pkl', map_location=device)
load_u3 = True
if load_u3:
    model_u3_x = torch.load('./model_u3/d1_ex1_final_u3_x.pkl', map_location=device)
    model_u3_y = torch.load('./model_u3/d1_ex1_final_u3_y.pkl', map_location=device)
load_tilde_u2 = True
if load_tilde_u2:
    model_tilde_u2 = torch.load('./model_tilde_u2/d1_ex1_final_tilde_u2.pkl', map_location=device)

with torch.no_grad():
    u0_norm_phi = model_u0(w1, x1, need_grad=0, normed=False)
    u0_norm = torch.sqrt(torch.sum(w1 * u0_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u0_alpha = model_u0.scaling_par()

    u1_x_norm_phi = model_u1_x(w1, x1, need_grad=0, normed=False)
    u1_x_norm = torch.sqrt(torch.sum(w1 * u1_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u1_alpha = model_u1_x.scaling_par()

    u1_y_norm_phi = model_u1_y(w2, x2, need_grad=0, normed=False)
    u1_y_norm = torch.sqrt(torch.sum(w2 * u1_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u1_y_shift = model_u1_y(dummy, x_error_0, need_grad=0, normed=False)
    u1_y_shift = u1_y_shift / u1_y_norm

    u2_x_norm_phi = model_u2_x(w1, x1, need_grad=0, normed=False)
    u2_x_norm = torch.sqrt(torch.sum(w1 * u2_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u2_alpha = model_u2_x.scaling_par()

    u2_y_norm_phi = model_u2_y(w2, x2, need_grad=0, normed=False)
    u2_y_norm = torch.sqrt(torch.sum(w2 * u2_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u2_y_shift = model_u2_y(dummy, x_error_0, need_grad=0, normed=False)
    u2_y_shift = u2_y_shift / u2_y_norm

    # 计算其他不依赖于rank的norm
    if load_tilde_u1:
        tilde_u1_norm_phi = model_tilde_u1(w2, x2, need_grad=0, normed=False)
        tilde_u1_norm = torch.sqrt(torch.sum(w2 * tilde_u1_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        tilde_u1_alpha = model_tilde_u1.scaling_par()
    
    if load_delta_u2:
        delta_u2_x_norm_phi = model_delta_u2_x(w1, x1, need_grad=0, normed=False)
        delta_u2_x_norm = torch.sqrt(torch.sum(w1 * delta_u2_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        delta_u2_alpha = model_delta_u2_x.scaling_par()
        delta_u2_y_norm_phi = model_delta_u2_y(w2, x2, need_grad=0, normed=False)
        delta_u2_y_norm = torch.sqrt(torch.sum(w2 * delta_u2_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        delta_u2_y_shift = model_delta_u2_y(dummy, x_error_0, need_grad=0, normed=False)
        delta_u2_y_shift = delta_u2_y_shift / delta_u2_y_norm

    if load_u3:
        u3_x_norm_phi = model_u3_x(w1, x1, need_grad=0, normed=False)
        u3_x_norm = torch.sqrt(torch.sum(w1 * u3_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        u3_alpha = model_u3_x.scaling_par()
        u3_y_norm_phi = model_u3_y(w2, x2, need_grad=0, normed=False)
        u3_y_norm = torch.sqrt(torch.sum(w2 * u3_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        u3_y_shift = model_u3_y(dummy, x_error_0, need_grad=0, normed=False)
        u3_y_shift = u3_y_shift / u3_y_norm

    if load_tilde_u2:
        tilde_u2_norm_phi = model_tilde_u2(w1, x1, need_grad=0, normed=False)
        tilde_u2_norm = torch.sqrt(torch.sum(w1 * tilde_u2_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        tilde_u2_alpha = model_tilde_u2.scaling_par()

    inv_e = [3,5,8,10,12,15,20,30]
    error_e = torch.zeros((len(inv_e)), dtype=dtype, device=device)
    l2_error1_hat_result_eps = torch.zeros(len(inv_e), dtype=dtype, device=device)
    l2_error1_result_eps = torch.zeros(len(inv_e), dtype=dtype, device=device)
    l2_error2_hat_result_eps = torch.zeros(len(inv_e), dtype=dtype, device=device)
    l2_error2_result_eps = torch.zeros(len(inv_e), dtype=dtype, device=device)
    l2_error3_result_eps = torch.zeros(len(inv_e), dtype=dtype, device=device)
    l2_error0_result_eps = torch.zeros(len(inv_e), dtype=dtype, device=device)
    norm_ue_fem = torch.zeros(len(inv_e), dtype=dtype, device=device)
    
    # 用于保存绘图数据
    all_coords = []
    all_u0_values = []  # u0不依赖于epsilon，只保存一次
    all_u1_values = {}  # u1依赖于epsilon，为每个epsilon保存
    all_u2_values = {}  # u2依赖于epsilon，为每个epsilon保存
    all_u3_values = {}  # u3依赖于epsilon，为每个epsilon保存

    print("\n计算误差...")
    for i_rank in range(0,2):
        coord_x_np = np.loadtxt(f'./FEM_data/e_3/coord_R14_rank{i_rank}.txt', unpack=True)
        coord_x = torch.from_numpy(coord_x_np).to(device)
        all_coords.append(coord_x)
        
        w_np = np.loadtxt(f'./FEM_data/e_3/w_R14_rank{i_rank}.txt', unpack=True)
        w = torch.from_numpy(w_np).to(device)

        u0_phi, u0_grad_phi = model_u0(dummy, coord_x, need_grad=1, normed=False)
        u0_phi = u0_phi / u0_norm
        u0_grad_phi = u0_grad_phi / u0_norm

        tnn_value_u0 = torch.sum(u0_alpha.unsqueeze(dim=-1) * torch.prod(u0_phi, dim=0), dim=0)
        all_u0_values.append(tnn_value_u0.cpu().numpy())

        u1_x_phi = model_u1_x(dummy, coord_x, need_grad=0, normed=False)
        u1_x_phi = u1_x_phi / u1_x_norm
        
        u2_x_phi = model_u2_x(dummy, coord_x, need_grad=0, normed=False)
        u2_x_phi = u2_x_phi / u2_x_norm
        
        if load_tilde_u1:
            tilde_u1_phi = model_tilde_u1(dummy, coord_x, need_grad=0, normed=False)
            tilde_u1_phi = tilde_u1_phi / tilde_u1_norm
            tnn_value_tilde_u1 = torch.sum(tilde_u1_alpha.unsqueeze(dim=-1) * torch.prod(tilde_u1_phi, dim=0), dim=0)
            
        if load_delta_u2:
            delta_u2_x_phi = model_delta_u2_x(dummy, coord_x, need_grad=0, normed=False)
            delta_u2_x_phi = delta_u2_x_phi / delta_u2_x_norm
        
        if load_u3:
            u3_x_phi = model_u3_x(dummy, coord_x, need_grad=0, normed=False)
            u3_x_phi = u3_x_phi / u3_x_norm

        if load_tilde_u2:
            tilde_u2_phi = model_tilde_u2(dummy, coord_x, need_grad=0, normed=False)
            tilde_u2_phi = tilde_u2_phi / tilde_u2_norm
            tnn_value_tilde_u2 = torch.sum(tilde_u2_alpha.unsqueeze(dim=-1) * torch.prod(tilde_u2_phi, dim=0), dim=0)
        
        for i_e in range(0, len(inv_e)):
            coord_y = coord_x * inv_e[i_e] - torch.floor(coord_x * inv_e[i_e])
            
            u1_y_phi = model_u1_y(dummy, coord_y, need_grad=0, normed=False)
            u1_y_phi = u1_y_phi / u1_y_norm
            tnn_value_u1 = torch.sum(u1_alpha.unsqueeze(dim=-1) * torch.prod(u1_x_phi, dim=0) * (torch.prod(u1_y_phi, dim=0) - torch.prod(u1_y_shift, dim=0)), dim=0)

            u2_y_phi = model_u2_y(dummy, coord_y, need_grad=0, normed=False)
            u2_y_phi = u2_y_phi / u2_y_norm
            tnn_value_u2 = torch.sum(u2_alpha.unsqueeze(dim=-1) * torch.prod(u2_x_phi, dim=0) * (torch.prod(u2_y_phi, dim=0) - torch.prod(u2_y_shift, dim=0)), dim=0)
            
            u3_y_phi = model_u3_y(dummy, coord_y, need_grad=0, normed=False)
            u3_y_phi = u3_y_phi / u3_y_norm
            tnn_value_u3 = torch.sum(u3_alpha.unsqueeze(dim=-1) * torch.prod(u3_x_phi, dim=0) * (torch.prod(u3_y_phi, dim=0) - torch.prod(u3_y_shift, dim=0)), dim=0)
            
            if load_delta_u2:
                delta_u2_y_phi = model_delta_u2_y(dummy, coord_y, need_grad=0, normed=False)
                delta_u2_y_phi = delta_u2_y_phi / delta_u2_y_norm
                tnn_value_delta_u2 = torch.sum(delta_u2_alpha.unsqueeze(dim=-1) * torch.prod(delta_u2_x_phi, dim=0) * (torch.prod(delta_u2_y_phi, dim=0) - torch.prod(delta_u2_y_shift, dim=0)), dim=0)

            # 计算完整解（归一化）
            tnn_value0 = tnn_value_u0
            tnn_value1 = tnn_value_u0 + (tnn_value_u1 + tnn_value_tilde_u1) / inv_e[i_e]
            tnn_value2 = tnn_value_u0 + (tnn_value_u1 + tnn_value_tilde_u1) / inv_e[i_e] + (tnn_value_u2 + tnn_value_delta_u2 + tnn_value_tilde_u2) / ((inv_e[i_e])**2)
            tnn_value3 = tnn_value_u0 + (tnn_value_u1 + tnn_value_tilde_u1) / inv_e[i_e] + (tnn_value_u2 + tnn_value_delta_u2 + tnn_value_tilde_u2) / ((inv_e[i_e])**2) + tnn_value_u3 / (inv_e[i_e]**3)

            # 加载FEM解并归一化
            fem_value_np = np.loadtxt(f'./FEM_data/e_{inv_e[i_e]}/u_fem_R14_rank{i_rank}.txt', unpack=True)
            fem_value = torch.from_numpy(fem_value_np).to(device)
            fem_value_normalized = fem_value 
            norm_ue_fem[i_e] += torch.sum(w * fem_value_normalized**2)

            # 保存用于绘图（每个epsilon分别保存）
            if i_e not in all_u1_values:
                all_u1_values[i_e] = []
            if i_e not in all_u2_values:
                all_u2_values[i_e] = []
            all_u1_values[i_e].append((tnn_value_u1 + tnn_value_tilde_u1).cpu().numpy())
            all_u2_values[i_e].append((tnn_value_u2 + tnn_value_delta_u2 + tnn_value_tilde_u2).cpu().numpy())
            if i_e not in all_u3_values:
                all_u3_values[i_e] = []
            all_u3_values[i_e].append(tnn_value_u3.cpu().numpy())
       

            # 计算误差（使用归一化后的解）
            current_error0 = torch.sum(w * (fem_value_normalized - tnn_value0)**2)
            l2_error0_result_eps[i_e] += current_error0
            print('l2 error square of 0 / %d from process rank %d : ' % (inv_e[i_e], i_rank), current_error0.item())

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

print('norm_ue_fem: ', torch.sqrt(norm_ue_fem))


import matplotlib.pyplot as plt
import os

# 创建png文件夹如果不存在的话
if not os.path.exists('./png'):
    os.makedirs('./png')

# 横坐标为 epsilon = 1 / inv_e
epsilons = 1.0 / np.array(inv_e)

# 画l2误差随epsilon减小的loglog图
plt.figure(figsize=(8, 6))
plt.loglog(epsilons, l2_error0_result_eps.cpu().numpy(), marker='o', label=' $|| u^{fem}_\epsilon-U_0 ||_{L_2}$')
plt.loglog(epsilons, l2_error1_result_eps.cpu().numpy(), marker='o', label=' $|| u^{fem}_\epsilon-U_1 ||_{L_2}$')
plt.loglog(epsilons, l2_error2_result_eps.cpu().numpy(), marker='o', label=' $|| u^{fem}_\epsilon-U_2 ||_{L_2}$')
plt.loglog(epsilons, l2_error3_result_eps.cpu().numpy(), marker='o', label=' $|| u^{fem}_\epsilon-U^*_3||_{L_2}$')

# 画参考线：O(ε) 和 O(ε^2)
ref_x = np.array([epsilons[0], epsilons[-1]])
ref0_y0 = l2_error0_result_eps.cpu().numpy()[0]
ref1_y0 = l2_error1_result_eps.cpu().numpy()[0]
ref2_y0 = l2_error2_result_eps.cpu().numpy()[0]
# ref3_y0 = l2_error3_result_eps.cpu().numpy()[0]
ref0 = ref0_y0 * (ref_x / ref_x[0])**1     # 斜率1
ref1 = ref1_y0 * (ref_x / ref_x[0])**2     # 斜率2
ref2 = ref2_y0 * (ref_x / ref_x[0])**3     # 斜率3
# ref3 = ref3_y0 * (ref_x / ref_x[0])**4     # 斜率4

plt.loglog(ref_x, ref0, 'k--', alpha=0.7)# , label=r'ref: slope=1'
plt.loglog(ref_x, ref1, 'k--', alpha=0.7)# , label=r'ref: slope=2'
plt.loglog(ref_x, ref2, 'k-.', alpha=0.7)# , label=r'ref: slope=3')
# plt.loglog(ref_x, ref3, 'k:', alpha=0.7)# , label=r'ref: slope=3')
plt.xlabel(r'$\epsilon$', fontsize=15)
plt.ylabel('$L^2$ Error', fontsize=15)
plt.title(r'$L^2$ Error vs $\epsilon$', fontsize=17)
plt.tick_params(axis='both', which='both', labelsize=12)
plt.grid(True, which="both", ls="--")
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("./png/L2_error.png")
plt.show()




# # # ------ 改进版：预读数据加速+避免反复从硬盘load -------


num_ranks = 2

# 合并所有rank的数据用于绘图（不需要每次合并，提前一次即可）
with torch.no_grad():
    all_coords_merged = np.concatenate(
        [c.cpu().numpy() if isinstance(c, torch.Tensor) else c for c in all_coords], axis=0
    )
    u0_merged = np.concatenate([u.cpu().numpy() if isinstance(u, torch.Tensor) else u for u in all_u0_values])
    u_full_all_eps = {}
    for i_e in range(0, len(inv_e)):
        u1_merged = np.concatenate(all_u1_values[i_e])
        u2_merged = np.concatenate(all_u2_values[i_e])
        u3_merged = np.concatenate(all_u3_values[i_e])

x_merge = all_coords_merged


import matplotlib.pyplot as plt

# eps=10 时画 2x2 子图
eps_plot = 30
# 找到 inv_e 中对应的索引
if eps_plot in inv_e:
    idx_plot = inv_e.index(eps_plot)
else:
    raise ValueError(f"inv_e 中没有 {eps_plot} 对应的数据")

print(f"\n绘制 eps={eps_plot} 的 2x2 子图...")

# 加载 FEM 解
fem_u_ranks = []
for i_rank in range(num_ranks):
    fname = f'./FEM_data/e_{eps_plot}/u_fem_R14_rank{i_rank}.txt'
    fem_value_np = np.loadtxt(fname, unpack=True)
    fem_u_ranks.append(fem_value_np)
fem_u_merge = np.concatenate(fem_u_ranks)
del fem_u_ranks
gc.collect()

# u0 按顺序合并
u0_merged = np.concatenate([u.cpu().numpy() if isinstance(u, torch.Tensor) else u for u in all_u0_values])
u0_merge_plot = u0_merged.copy()

# u1/u2/u3 (在idx_plot对应epsilon下)
u1_merge_plot = np.concatenate(all_u1_values[idx_plot])
u2_merge_plot = np.concatenate(all_u2_values[idx_plot])
u3_merge_plot = np.concatenate(all_u3_values[idx_plot])

# 计算4个近似解
u01_plot = u0_merge_plot + u1_merge_plot / eps_plot
u012_plot = u01_plot + u2_merge_plot / (eps_plot ** 2)
u0123_plot = u012_plot + u3_merge_plot / (eps_plot ** 3)

# 误差
err0_plot = fem_u_merge - u0_merge_plot
err01_plot = fem_u_merge - u01_plot
err012_plot = fem_u_merge - u012_plot
err0123_plot = fem_u_merge - u0123_plot

# 合并坐标
x_plot = np.concatenate(
    [c.cpu().numpy() if isinstance(c, torch.Tensor) else c for c in all_coords], axis=0
)

# 按x排序，保证曲线平滑
sort_idx = np.argsort(x_plot)
x_sorted = x_plot[sort_idx]
u012_sorted = u012_plot[sort_idx]
err0_sorted = err0_plot[sort_idx]
err01_sorted = err01_plot[sort_idx]
err012_sorted = err012_plot[sort_idx]
err0123_sorted = err0123_plot[sort_idx]

fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

# 设置字体大小
label_fontsize = 18
tick_fontsize = 14
title_fontsize = 20
legend_fontsize = 18
title_pad = 14  # 调大pad使标题不贴紧

# 左上：画出u_fem, u0, u0+u1/eps, u0+u1/eps+u2/eps^2的图像
axs[0, 0].plot(x_sorted, fem_u_merge[sort_idx], 'k', linewidth=1.5, label=r'$u^{{FEM}}_\epsilon$')
axs[0, 0].plot(x_sorted, u0_merge_plot[sort_idx], 'b--', linewidth=1.5, label=r'$U_0$')
axs[0, 0].plot(x_sorted, (u0_merge_plot + u1_merge_plot / eps_plot)[sort_idx], 'g-.', linewidth=1.5, label=r'$U_1$')
axs[0, 0].plot(x_sorted, u012_sorted, 'r', linewidth=1.5, label=r'$U_2$')
axs[0, 0].set_title(r'Solution comparison, $\epsilon=1/{}$'.format(eps_plot), fontsize=title_fontsize, pad=title_pad)
axs[0, 0].set_xlabel('$x$', fontsize=label_fontsize)
axs[0, 0].set_ylabel('value', fontsize=label_fontsize)
axs[0, 0].legend(fontsize=legend_fontsize)
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].tick_params(axis='both', labelsize=tick_fontsize)

# 右上：FEM-u0
axs[0, 1].plot(x_sorted, err0_sorted, 'r', linewidth=1.5)
axs[0, 1].set_title(r'Error:$u^{{FEM}}_\epsilon - U_0$, $\epsilon=1/{}$'.format(eps_plot), fontsize=title_fontsize, pad=title_pad)
axs[0, 1].set_xlabel('$x$', fontsize=label_fontsize)
axs[0, 1].set_ylabel('error', fontsize=label_fontsize)
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].tick_params(axis='both', labelsize=tick_fontsize)

# 左下：FEM-(u0+u1/eps)
axs[1, 0].plot(x_sorted, err01_sorted, 'g', linewidth=1.5)
axs[1, 0].set_title(r'Error:$u^{{FEM}}_\epsilon - U_1$, $\epsilon=1/{}$'.format(eps_plot), fontsize=title_fontsize, pad=title_pad)
axs[1, 0].set_xlabel('$x$', fontsize=label_fontsize)
axs[1, 0].set_ylabel('error', fontsize=label_fontsize)
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].tick_params(axis='both', labelsize=tick_fontsize)

# 右下：FEM-(u0+u1/eps+u2/eps^2+u3/eps^3) 和 FEM-(u0+u1/eps+u2/eps^2) 两条线
axs[1, 1].plot(x_sorted, err0123_sorted, color='purple', linewidth=1.5, label=r'$u^{{FEM}}_\epsilon - U_3^*$')
axs[1, 1].plot(x_sorted, err012_sorted, color='orange', linestyle='--', linewidth=1.5, label=r'$u^{{FEM}}_\epsilon - U_2$')
axs[1, 1].set_title(r'Error: high order approximations, $\epsilon=1/{}$'.format(eps_plot), fontsize=title_fontsize, pad=title_pad)
axs[1, 1].set_xlabel('$x$', fontsize=label_fontsize)
axs[1, 1].set_ylabel('error', fontsize=label_fontsize)
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].legend(fontsize=legend_fontsize)
axs[1, 1].tick_params(axis='both', labelsize=tick_fontsize)

plt.savefig('./png/eps{}_2x2_comparison_1d.png'.format(eps_plot), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  ✓ 已保存: ./png/eps{eps_plot}_2x2_comparison_1d.png')


# n_rows, n_cols = 3, 3
# fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 12), constrained_layout=True)

# for i_e, inv_eps in enumerate(inv_e):
#     epsilon = 1.0 / inv_eps

    
#     fem_u_ranks = []
#     for i_rank in range(num_ranks):
#         fname = f'./FEM_data/e_{inv_e[i_e]}/u_fem_R14_rank{i_rank}.txt'
#         fem_value_np = np.loadtxt(fname, unpack=True)
#         fem_u_ranks.append(fem_value_np)
#     fem_u_merge = np.concatenate(fem_u_ranks)
#     # 主动释放临时变量
#     del fem_u_ranks
#     gc.collect()

#     tnn_u_merge = u_full_all_eps[i_e]
#     diff = fem_u_merge - tnn_u_merge

#     # 按 x 坐标排序，保证绘图线条正确
#     sort_idx = np.argsort(x_merge)
#     x_sorted = x_merge[sort_idx]
#     diff_sorted = diff[sort_idx]

#     row = i_e // n_cols
#     col = i_e % n_cols
#     ax = axs[row, col]
#     ax.plot(x_sorted, diff_sorted, 'b-', linewidth=0.5, alpha=0.8)
#     ax.set_xlabel('$x$', fontsize=12)
#     ax.set_ylabel('FEM - TNN', fontsize=12)
#     ax.set_title(f'$\\epsilon={epsilon:.4g}$', fontsize=14)
#     ax.grid(True, alpha=0.3)

#     # 主动删除当前有限元解以释放内存
#     del fem_u_merge
#     gc.collect()

# for j in range(len(inv_e), n_rows * n_cols):
#     row = j // n_cols
#     col = j % n_cols
#     fig.delaxes(axs[row, col])

# plt.suptitle(r"Finite element solution minus TNN solution (for different $\epsilon$) 1D plot", fontsize=20)
# plt.savefig('./png/FEM减TNN_1D_多epsilon_子图_nobd.png', dpi=150, bbox_inches='tight')
# plt.close(fig)
# print('  ✓ 已保存: ./png/FEM减TNN_1D_多epsilon_子图_nobd.png')

# # =========================================================================
# # 画出 eps=10 时，u_fem, u0, u0+u1/epsilon, u0+u1/epsilon+u2/epsilon^2,
# #        u0+u1/epsilon+u2/epsilon^2+u3/epsilon^3 的图像（含局部放大）
# # =========================================================================
# eps_plot = 10

# # 找到 inv_e 中对应的索引
# if eps_plot in inv_e:
#     idx_plot = inv_e.index(eps_plot)
# else:
#     raise ValueError(f"inv_e 中没有 {eps_plot} 对应的数据")

# # 读取 FEM 解
# fem_u_ranks_plot = []
# for i_rank in range(num_ranks):
#     fname_plot = f'./FEM_data/e_{eps_plot}/u_fem_R14_rank{i_rank}.txt'
#     fem_value_np_plot = np.loadtxt(fname_plot, unpack=True)
#     fem_u_ranks_plot.append(fem_value_np_plot)
# fem_u_merge_plot = np.concatenate(fem_u_ranks_plot)

# # 坐标和 FEM 解排序
# x_plot = x_merge.copy()
# sort_idx_plot = np.argsort(x_plot)
# x_plot_sorted = x_plot[sort_idx_plot]
# fem_u_sorted = fem_u_merge_plot[sort_idx_plot]

# # u0 (已排序)
# u0_sorted = u0_merged[sort_idx_plot]

# # u0 + u1/eps
# u1_merged_plot = np.concatenate(all_u1_values[idx_plot])
# u01_sorted = (u0_merged + u1_merged_plot / eps_plot)[sort_idx_plot]

# # u0 + u1/eps + u2/eps^2
# u2_merged_plot = np.concatenate(all_u2_values[idx_plot])
# u012_sorted = (u0_merged + u1_merged_plot / eps_plot + u2_merged_plot / (eps_plot ** 2))[sort_idx_plot]

# # u0 + u1/eps + u2/eps^2 + u3/eps^3
# u3_merged_plot = np.concatenate(all_u3_values[idx_plot])
# u0123_sorted = (u0_merged + u1_merged_plot / eps_plot + u2_merged_plot / (eps_plot ** 2) + u3_merged_plot / (eps_plot ** 3))[sort_idx_plot]

# # 局部放大区间
# zoom_xmin, zoom_xmax = 0.47, 0.48
# zoom_mask = (x_plot_sorted >= zoom_xmin) & (x_plot_sorted <= zoom_xmax)

# # 绘图：左图全局，右图局部放大
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)

# ax1.plot(x_plot_sorted, fem_u_sorted, 'k--', linewidth=1.5, label=r'$u^{fem}_\epsilon$', alpha=0.9)
# ax1.plot(x_plot_sorted, u0_sorted, 'b-', linewidth=1.2, label=r'$u_0$', alpha=0.8)
# ax1.plot(x_plot_sorted, u01_sorted, 'r-.', linewidth=1.2, label=r'$u_0 + u_1/\epsilon$', alpha=0.8)
# ax1.plot(x_plot_sorted, u012_sorted, 'g:', linewidth=1.5, label=r'$u_0 + u_1/\epsilon + u_2/\epsilon^2$', alpha=0.8)
# ax1.plot(x_plot_sorted, u0123_sorted, 'm-.', linewidth=1.5, label=r'$u_0 + u_1/\epsilon + u_2/\epsilon^2 + u_3/\epsilon^3$', alpha=0.8)
# ax1.set_xlabel(r'$x$', fontsize=14)
# ax1.set_ylabel(r'$u$', fontsize=14)
# ax1.set_title(rf'Full domain, $\epsilon = {eps_plot}$', fontsize=16)
# ax1.legend(fontsize=10, loc='best')
# ax1.grid(True, alpha=0.3)
# ax1.axvspan(zoom_xmin, zoom_xmax, alpha=0.15, color='orange', label='zoom region')

# ax2.plot(x_plot_sorted[zoom_mask], fem_u_sorted[zoom_mask], 'k--', linewidth=1.5, label=r'$u^{fem}_\epsilon$', alpha=0.9)
# ax2.plot(x_plot_sorted[zoom_mask], u0_sorted[zoom_mask], 'b-', linewidth=1.2, label=r'$u_0$', alpha=0.8)
# ax2.plot(x_plot_sorted[zoom_mask], u01_sorted[zoom_mask], 'r-.', linewidth=1.2, label=r'$u_0 + u_1/\epsilon$', alpha=0.8)
# ax2.plot(x_plot_sorted[zoom_mask], u012_sorted[zoom_mask], 'g:', linewidth=1.5, label=r'$u_0 + u_1/\epsilon + u_2/\epsilon^2$', alpha=0.8)
# ax2.plot(x_plot_sorted[zoom_mask], u0123_sorted[zoom_mask], 'm-.', linewidth=1.5, label=r'$u_0 + u_1/\epsilon + u_2/\epsilon^2 + u_3/\epsilon^3$', alpha=0.8)
# ax2.set_xlabel(r'$x$', fontsize=14)
# ax2.set_ylabel(r'$u$', fontsize=14)
# ax2.set_title(rf'Zoomed: $x \in [{zoom_xmin}, {zoom_xmax}]$, $\epsilon = {eps_plot}$', fontsize=16)
# ax2.legend(fontsize=10, loc='best')
# ax2.grid(True, alpha=0.3)

# plt.savefig('./png/u_fem_vs_tnn_eps10_zoom.png', dpi=150, bbox_inches='tight')
# plt.close(fig)
# print(f'  ✓ 已保存: ./png/u_fem_vs_tnn_eps10_zoom.png')


# # =========================================================================
# # 画出所有 eps 对应的图像，每个 eps 一个子图
# # =========================================================================
# n_eps = len(inv_e)
# n_rows, n_cols = 3, 3
# fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 15), constrained_layout=True)

# for i_e, inv_eps in enumerate(inv_e):
#     eps_val = 1.0 / inv_eps

#     # 读取 FEM 解
#     fem_u_ranks = []
#     for i_rank in range(num_ranks):
#         fname = f'./FEM_data/e_{inv_eps}/u_fem_R14_rank{i_rank}.txt'
#         fem_value_np = np.loadtxt(fname, unpack=True)
#         fem_u_ranks.append(fem_value_np)
#     fem_u_merge = np.concatenate(fem_u_ranks)
#     del fem_u_ranks
#     gc.collect()

#     # 排序
#     sort_idx = np.argsort(x_merge)
#     x_sorted = x_merge[sort_idx]
#     fem_sorted = fem_u_merge[sort_idx]
#     u0_sorted = u0_merged[sort_idx]

#     # u1, u2, u3
#     u1_merged = np.concatenate(all_u1_values[i_e])
#     u2_merged = np.concatenate(all_u2_values[i_e])
#     u3_merged = np.concatenate(all_u3_values[i_e])

#     u01_sorted = (u0_merged + u1_merged / inv_eps)[sort_idx]
#     u012_sorted = (u0_merged + u1_merged / inv_eps + u2_merged / (inv_eps ** 2))[sort_idx]
#     u0123_sorted = (u0_merged + u1_merged / inv_eps + u2_merged / (inv_eps ** 2) + u3_merged / (inv_eps ** 3))[sort_idx]

#     row = i_e // n_cols
#     col = i_e % n_cols
#     ax = axs[row, col]

#     ax.plot(x_sorted, fem_sorted, 'k-', linewidth=1.2, label=r'$u^{fem}_\epsilon$', alpha=0.9)
#     ax.plot(x_sorted, u0_sorted, 'b--', linewidth=1.0, label=r'$u_0$', alpha=0.8)
#     ax.plot(x_sorted, u01_sorted, 'r-.', linewidth=1.0, label=r'$u_0+u_1/\epsilon$', alpha=0.8)
#     ax.plot(x_sorted, u012_sorted, 'g:', linewidth=1.2, label=r'$u_0+u_1/\epsilon+u_2/\epsilon^2$', alpha=0.8)
#     ax.plot(x_sorted, u0123_sorted, 'm-', linewidth=1.2, label=r'$u_0+u_1/\epsilon+u_2/\epsilon^2+u_3/\epsilon^3$', alpha=0.8)

#     ax.set_xlabel(r'$x$', fontsize=10)
#     ax.set_ylabel(r'$u$', fontsize=10)
#     ax.set_title(rf'$\epsilon = {eps_val:.4g}$', fontsize=12)
#     ax.legend(fontsize=7, loc='best', framealpha=0.5)
#     ax.grid(True, alpha=0.3)

#     del fem_u_merge, fem_sorted, u1_merged, u2_merged, u3_merged
#     gc.collect()

# for j in range(n_eps, n_rows * n_cols):
#     row = j // n_cols
#     col = j % n_cols
#     fig.delaxes(axs[row, col])

# plt.suptitle(rf"$u_{{fem}}$ vs TNN approximations for different $\epsilon$", fontsize=18)
# plt.savefig('./png/u_fem_vs_tnn_all_eps_subplots.png', dpi=150, bbox_inches='tight')
# plt.close(fig)
# print(f'  ✓ 已保存: ./png/u_fem_vs_tnn_all_eps_subplots.png')


# 计算三条对数误差曲线的斜率（对数尺度下斜率即为 log-log 拟合的指数）

# epsilons: 横轴 eps，l2_error*_result_eps: 纵轴误差
import numpy as np

log_eps = np.log(epsilons)
log_err0 = np.log(l2_error0_result_eps.cpu().numpy())
log_err1 = np.log(l2_error1_result_eps.cpu().numpy())
log_err2 = np.log(l2_error2_result_eps.cpu().numpy())

# 拟合一阶多项式，返回斜率和截距
slope0, intercept0 = np.polyfit(log_eps, log_err0, 1)
slope1, intercept1 = np.polyfit(log_eps, log_err1, 1)
slope2, intercept2 = np.polyfit(log_eps, log_err2, 1)

print(f"Error0 \\| u_fem-u_0 \\|_L2 的 log-log 斜率: {slope0:.4f}")
print(f"Error1 \\| u_fem-u_0-εu_1 \\|_L2 的 log-log 斜率: {slope1:.4f}")
print(f"Error2 \\| u_fem-u_0-εu_1-ε^2u_2 \\|_L2 的 log-log 斜率: {slope2:.4f}")