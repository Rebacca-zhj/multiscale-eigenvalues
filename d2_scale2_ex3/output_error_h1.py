from typing import Any

import torch
import numpy as np
import scipy.io as sio

from tnn import *
from integration import *
from quadrature import *
import matplotlib.pyplot as plt

pi = 3.14159265358979323846
dtype = torch.double
device = 'cpu'
 
dim = 2
p_A = 1
p_chi = 20
p_u0 = 20
p_u1 = 20
p_u2 = 20

a, b = 0, 1

x1, w1 = composite_quadrature_1d(4, 0, 1, 200, device=device)
N = len(x1)
x2, w2 = composite_quadrature_1d(4, 0, 1, 200, device=device)
x3, w3 = composite_quadrature_1d(4, 0, 1, 200, device=device)
x4, w4 = composite_quadrature_1d(4, 0, 1, 200, device=device)

w1234 = torch.stack((w1, w2, w3, w4), dim=0)
w12 = torch.stack((w1, w2), dim=0)
w34 = torch.stack((w3, w4), dim=0)

N = len(x1)
dummy = torch.ones((1, 1), dtype=dtype, device=device)

x_error_0 = torch.zeros(1, dtype=dtype, device=device)

def bd_u0(x):
    return (x - a) * (b - x)

def grad_bd_u0(x):
    return a + b - 2 * x

def grad_grad_bd_u0(x):
    return -2 * torch.ones_like(x)

def grad_grad_grad_bd_u0(x):
    return 0 * torch.ones_like(x)

model_u0 = torch.load("./model_u0/d2_ex4_final_u0.pkl", map_location=device)
model_u1_x = torch.load('./model_u1/d2_ex4_final_u1_x.pkl', map_location=device)
model_u1_y = torch.load('./model_u1/d2_ex4_final_u1_y.pkl', map_location=device)
model_u2_x = torch.load('./model_u2/d2_ex4_final_u2_x.pkl', map_location=device)
model_u2_y = torch.load('./model_u2/d2_ex4_final_u2_y.pkl', map_location=device)

load_tilde_u1 = True
if load_tilde_u1:
    model_tilde_u1 = torch.load('./model_tilde_u1/d2_ex4_final_tilde_u1.pkl', map_location=device)
load_delta_u2 = True
if load_delta_u2:
    model_delta_u2_x = torch.load('./model_delta_u2/d2_ex4_final_delta_u2_x.pkl', map_location=device)
    model_delta_u2_y = torch.load('./model_delta_u2/d2_ex4_final_delta_u2_y.pkl', map_location=device)
load_tilde_u2 = True
if load_tilde_u2:
    model_tilde_u2 = torch.load('./model_tilde_u2/d2_ex4_autograd_final_tilde_u2.pkl', map_location=device)

with torch.no_grad():
    # 归一化及函数分量准备
    u0_norm_phi = model_u0(w1, x1, normed=False)
    u0_norm = torch.sqrt(torch.sum(w1 * u0_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    alpha_u0 = model_u0.scaling_par()
    
    u1_x_norm_phi = model_u1_x(w1, x1, normed=False)
    u1_x_norm = torch.sqrt(torch.sum(w1 * u1_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u1_alpha = model_u1_x.scaling_par()
    u1_y_norm_phi = model_u1_y(w3, x3, normed=False)
    u1_y_norm = torch.sqrt(torch.sum(w3 * u1_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u1_y_shift = model_u1_y(dummy, x_error_0, need_grad=0, normed=False)
    u1_y_shift = u1_y_shift / u1_y_norm
    
    u2_x_norm_phi = model_u2_x(w1, x1, normed=False)
    u2_x_norm = torch.sqrt(torch.sum(w1 * u2_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u2_alpha = model_u2_x.scaling_par()
    u2_y_norm_phi = model_u2_y(w3, x3, normed=False)
    u2_y_norm = torch.sqrt(torch.sum(w3 * u2_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u2_y_shift = model_u2_y(dummy, x_error_0, need_grad=0, normed=False)
    u2_y_shift = u2_y_shift / u2_y_norm

    # 计算其他不依赖于rank的norm
    if load_tilde_u1:
        tilde_u1_norm_phi = model_tilde_u1(w1, x1, normed=False)
        tilde_u1_norm = torch.sqrt(torch.sum(w1 * tilde_u1_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        alpha_tilde_u1 = model_tilde_u1.scaling_par()
    
    if load_delta_u2:
        delta_u2_x_norm_phi = model_delta_u2_x(w1, x1, normed=False)
        delta_u2_x_norm = torch.sqrt(torch.sum(w1 * delta_u2_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        delta_u2_alpha = model_delta_u2_x.scaling_par()
        delta_u2_y_norm_phi = model_delta_u2_y(w3, x3, normed=False)
        delta_u2_y_norm = torch.sqrt(torch.sum(w3 * delta_u2_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        delta_u2_y_shift = model_delta_u2_y(dummy, x_error_0, need_grad=0, normed=False)
        delta_u2_y_shift = delta_u2_y_shift / delta_u2_y_norm

    if load_tilde_u2:
        tilde_u2_norm_phi = model_tilde_u2(w1, x1, need_grad=0, normed=False)
        tilde_u2_norm = torch.sqrt(torch.sum(w1 * tilde_u2_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        tilde_u2_alpha = model_tilde_u2.scaling_par()

    epsilons = [3,5,8,10,12,15]
    # 初始化累加器（每个epsilon一个值）
    h1_seminorm1_sq = np.zeros(len(epsilons))
    h1_seminorm2_sq = np.zeros(len(epsilons))
    
    # 用于保存绘图数据
    all_coords = []
    

    for i_rank in range(0,4):
        coord_x_np = np.loadtxt(f'./periodic_P2_R9_260825/quad/coord_R9_rank{i_rank}.txt', unpack=True)
        coord_x = torch.from_numpy(coord_x_np).to(device)
        
        w_np = np.loadtxt(f'./periodic_P2_R9_260825/quad/w_R9_rank{i_rank}.txt', unpack=True)
        w = torch.from_numpy(w_np).to(device)

        u0_phi, u0_grad_phi = model_u0(dummy, coord_x, need_grad=1, normed=False)
        u0_phi = u0_phi / u0_norm
        u0_grad_phi = u0_grad_phi / u0_norm
        u0_grad_x1_phi = torch.cat((u0_grad_phi[0:1, :, :], u0_phi[1:2, :, :]), dim=0)
        u0_grad_x2_phi = torch.cat((u0_phi[0:1, :, :], u0_grad_phi[1:2, :, :]), dim=0)
        u0 = torch.einsum('p,pn->n', alpha_u0, torch.prod(u0_phi, dim=0))
        u0_grad_x1 = torch.einsum('p,pn->n', alpha_u0, torch.prod(u0_grad_x1_phi, dim=0))
        u0_grad_x2 = torch.einsum('p,pn->n', alpha_u0, torch.prod(u0_grad_x2_phi, dim=0))

        u1_x_phi, u1_x_grad_phi = model_u1_x(dummy, coord_x, need_grad=1, normed=False)
        u1_x_phi = u1_x_phi / u1_x_norm
        u1_x_grad_phi = u1_x_grad_phi / u1_x_norm

        u2_x_phi, u2_x_grad_phi = model_u2_x(dummy, coord_x, need_grad=1, normed=False)
        u2_x_phi = u2_x_phi / u2_x_norm
        u2_x_grad_phi = u2_x_grad_phi / u2_x_norm

        if load_tilde_u1:
            tilde_u1_phi, tilde_u1_grad_phi = model_tilde_u1(dummy, coord_x, need_grad=1, normed=False)
            tilde_u1_phi = tilde_u1_phi / tilde_u1_norm
            tilde_u1_grad_phi = tilde_u1_grad_phi / tilde_u1_norm
            tilde_u1_grad_x1_phi = torch.cat((tilde_u1_grad_phi[0:1, :, :], tilde_u1_phi[1:2, :, :]), dim=0)
            tilde_u1_grad_x2_phi = torch.cat((tilde_u1_phi[0:1, :, :], tilde_u1_grad_phi[1:2, :, :]), dim=0)
            tilde_u1_grad_x1 = torch.einsum('p,pn->n', alpha_tilde_u1, torch.prod(tilde_u1_grad_x1_phi, dim=0))
            tilde_u1_grad_x2 = torch.einsum('p,pn->n', alpha_tilde_u1, torch.prod(tilde_u1_grad_x2_phi, dim=0))
        else:
            tilde_u1 = torch.zeros_like(u0)
            tilde_u1_grad_x1 = torch.zeros_like(u0_grad_x1)
            tilde_u1_grad_x2 = torch.zeros_like(u0_grad_x2)

        if load_delta_u2:
            delta_u2_x_phi, delta_u2_x_grad_phi = model_delta_u2_x(dummy, coord_x, need_grad=1, normed=False)
            delta_u2_x_phi = delta_u2_x_phi / delta_u2_x_norm
            delta_u2_x_grad_phi = delta_u2_x_grad_phi / delta_u2_x_norm

        if load_tilde_u2:
            tilde_u2_phi, tilde_u2_grad_phi = model_tilde_u2(dummy, coord_x, need_grad=1, normed=False)
            tilde_u2_phi = tilde_u2_phi / tilde_u2_norm
            tilde_u2_grad_phi = tilde_u2_grad_phi / tilde_u2_norm
            tilde_u2_grad_x1_phi = torch.cat((tilde_u2_grad_phi[0:1, :, :], tilde_u2_phi[1:2, :, :]), dim=0)
            tilde_u2_grad_x2_phi = torch.cat((tilde_u2_phi[0:1, :, :], tilde_u2_grad_phi[1:2, :, :]), dim=0)
            tilde_u2_grad_x1 = torch.sum(tilde_u2_alpha.unsqueeze(dim=-1) * torch.prod(tilde_u2_grad_x1_phi, dim=0), dim=0)
            tilde_u2_grad_x2 = torch.sum(tilde_u2_alpha.unsqueeze(dim=-1) * torch.prod(tilde_u2_grad_x2_phi, dim=0), dim=0)
        else:
            tilde_u2 = torch.zeros_like(u0)
            tilde_u2_grad_x1 = torch.zeros_like(u0_grad_x1)
            tilde_u2_grad_x2 = torch.zeros_like(u0_grad_x2)

        for idx, eps in enumerate(epsilons):
            # 计算一阶展开项
            coord_y = coord_x * eps - torch.floor(coord_x * eps)
            u1_y_phi, u1_y_grad_phi = model_u1_y(dummy, coord_y, need_grad=1, normed=False)
            u1_y_phi = u1_y_phi / u1_y_norm
            u1_y_grad_phi = u1_y_grad_phi / u1_y_norm
            u1_grad_x1_phi = torch.cat((u1_x_grad_phi[0:1, :, :], u1_x_phi[1:2, :, :]), dim=0)
            u1_grad_x2_phi = torch.cat((u1_x_phi[0:1, :, :], u1_x_grad_phi[1:2, :, :]), dim=0)
            u1_grad_y1_phi = torch.cat((u1_y_grad_phi[0:1, :, :], u1_y_phi[1:2, :, :]), dim=0)
            u1_grad_y2_phi = torch.cat((u1_y_phi[0:1, :, :], u1_y_grad_phi[1:2, :, :]), dim=0)
            u1_grad_x1 = torch.sum((u1_alpha.unsqueeze(dim=-1) * torch.prod(u1_grad_x1_phi, dim=0) * (torch.prod(u1_y_phi, dim=0) - torch.prod(u1_y_shift, dim=0))), dim=0)
            u1_grad_x2 = torch.sum((u1_alpha.unsqueeze(dim=-1) * torch.prod(u1_grad_x2_phi, dim=0) * (torch.prod(u1_y_phi, dim=0) - torch.prod(u1_y_shift, dim=0))), dim=0)
            u1_grad_y1 = torch.sum((u1_alpha.unsqueeze(dim=-1) * torch.prod(u1_x_phi, dim=0) * torch.prod(u1_grad_y1_phi, dim=0)), dim=0)
            u1_grad_y2 = torch.sum((u1_alpha.unsqueeze(dim=-1) * torch.prod(u1_x_phi, dim=0) * torch.prod(u1_grad_y2_phi, dim=0)), dim=0)
            del u1_grad_x1_phi,u1_grad_x2_phi,u1_grad_y1_phi,u1_grad_y2_phi
            
            u2_y_phi, u2_y_grad_phi = model_u2_y(dummy, coord_y, need_grad=1, normed=False)
            u2_y_phi = u2_y_phi / u2_y_norm
            u2_y_grad_phi = u2_y_grad_phi / u2_y_norm
            u2_grad_x1_phi = torch.cat((u2_x_grad_phi[0:1, :, :], u2_x_phi[1:2, :, :]), dim=0)
            u2_grad_x2_phi = torch.cat((u2_x_phi[0:1, :, :], u2_x_grad_phi[1:2, :, :]), dim=0)
            u2_grad_y1_phi = torch.cat((u2_y_grad_phi[0:1, :, :], u2_y_phi[1:2, :, :]), dim=0)
            u2_grad_y2_phi = torch.cat((u2_y_phi[0:1, :, :], u2_y_grad_phi[1:2, :, :]), dim=0)
            u2_grad_x1 = torch.sum((u2_alpha.unsqueeze(dim=-1) * torch.prod(u2_grad_x1_phi, dim=0) * (torch.prod(u2_y_phi, dim=0) - torch.prod(u2_y_shift, dim=0))), dim=0)
            u2_grad_x2 = torch.sum((u2_alpha.unsqueeze(dim=-1) * torch.prod(u2_grad_x2_phi, dim=0) * (torch.prod(u2_y_phi, dim=0) - torch.prod(u2_y_shift, dim=0))), dim=0)
            u2_grad_y1 = torch.sum((u2_alpha.unsqueeze(dim=-1) * torch.prod(u2_x_phi, dim=0) * torch.prod(u2_grad_y1_phi, dim=0)), dim=0)
            u2_grad_y2 = torch.sum((u2_alpha.unsqueeze(dim=-1) * torch.prod(u2_x_phi, dim=0) * torch.prod(u2_grad_y2_phi, dim=0)), dim=0)
            del u2_grad_x1_phi,u2_grad_x2_phi,u2_grad_y1_phi,u2_grad_y2_phi

            if load_delta_u2:
                delta_u2_y_phi, delta_u2_y_grad_phi = model_delta_u2_y(dummy, coord_y, need_grad=1, normed=False)
                delta_u2_y_phi = delta_u2_y_phi / delta_u2_y_norm
                delta_u2_y_grad_phi = delta_u2_y_grad_phi / delta_u2_y_norm

                delta_u2_grad_x1_phi = torch.cat((delta_u2_x_grad_phi[0:1, :, :], delta_u2_x_phi[1:2, :, :]), dim=0)
                delta_u2_grad_x2_phi = torch.cat((delta_u2_x_phi[0:1, :, :], delta_u2_x_grad_phi[1:2, :, :]), dim=0)
                delta_u2_grad_y1_phi = torch.cat((delta_u2_y_grad_phi[0:1, :, :], delta_u2_y_phi[1:2, :, :]), dim=0)
                delta_u2_grad_y2_phi = torch.cat((delta_u2_y_phi[0:1, :, :], delta_u2_y_grad_phi[1:2, :, :]), dim=0)
                delta_u2_grad_x1 = torch.sum((delta_u2_alpha.unsqueeze(dim=-1) * torch.prod(delta_u2_grad_x1_phi, dim=0) * (torch.prod(delta_u2_y_phi, dim=0) - torch.prod(delta_u2_y_shift, dim=0))), dim=0)
                delta_u2_grad_x2 = torch.sum((delta_u2_alpha.unsqueeze(dim=-1) * torch.prod(delta_u2_grad_x2_phi, dim=0) * (torch.prod(delta_u2_y_phi, dim=0) - torch.prod(delta_u2_y_shift, dim=0))), dim=0)
                delta_u2_grad_y1 = torch.sum((delta_u2_alpha.unsqueeze(dim=-1) * torch.prod(delta_u2_x_phi, dim=0) * torch.prod(delta_u2_grad_y1_phi, dim=0)), dim=0)
                delta_u2_grad_y2 = torch.sum((delta_u2_alpha.unsqueeze(dim=-1) * torch.prod(delta_u2_x_phi, dim=0) * torch.prod(delta_u2_grad_y2_phi, dim=0)), dim=0)
                del delta_u2_grad_x1_phi,delta_u2_grad_x2_phi,delta_u2_grad_y1_phi,delta_u2_grad_y2_phi
            else:
                delta_u2 = torch.zeros_like(u0)
                delta_u2_grad_x1 = torch.zeros_like(u0_grad_x1)
                delta_u2_grad_x2 = torch.zeros_like(u0_grad_x2)
                delta_u2_grad_y1 = torch.zeros_like(u0_grad_x1)
                delta_u2_grad_y2 = torch.zeros_like(u0_grad_x2)
                
            # 计算梯度（未归一化）
            grad_one_order_x1_1 = (u0_grad_x1 + u1_grad_y1).cpu().numpy()
            grad_one_order_x2_1 = (u0_grad_x2 + u1_grad_y2).cpu().numpy()
            
            grad_second_order_x1_1 = (u1_grad_x1 + tilde_u1_grad_x1).cpu().numpy()
            grad_second_order_x2_1 = (u1_grad_x2 + tilde_u1_grad_x2).cpu().numpy()
            
            grad_one_order_x1_2 = (u0_grad_x1 + u1_grad_y1).cpu().numpy()
            grad_one_order_x2_2 = (u0_grad_x2 + u1_grad_y2).cpu().numpy()
            
            grad_second_order_x1_2 = (u1_grad_x1 + tilde_u1_grad_x1 + u2_grad_y1 + delta_u2_grad_y1).cpu().numpy()
            grad_second_order_x2_2 = (u1_grad_x2 + tilde_u1_grad_x2 + u2_grad_y2 + delta_u2_grad_y2).cpu().numpy()
            
            grad_third_order_x1_2 = (u2_grad_x1 + delta_u2_grad_x1 + tilde_u2_grad_x1).cpu().numpy()
            grad_third_order_x2_2 = (u2_grad_x2 + delta_u2_grad_x2 + tilde_u2_grad_x2).cpu().numpy()
            
            # 加载FEM参考解
            func_values_matrix_x1 = np.loadtxt(f'./periodic_P2_R9_260825/e_{eps}/u_d10_R9_rank{i_rank}.txt')
            func_values_matrix_x2 = np.loadtxt(f'./periodic_P2_R9_260825/e_{eps}/u_d01_R9_rank{i_rank}.txt')
            func_values_vector_x1 = func_values_matrix_x1.reshape(-1) 
            func_values_vector_x2 = func_values_matrix_x2.reshape(-1) 

            eps_value = 1 / eps
            diff1_x1 = func_values_vector_x1 - (grad_one_order_x1_1 + eps_value * grad_second_order_x1_1)
            diff1_x2 = func_values_vector_x2 - (grad_one_order_x2_1 + eps_value * grad_second_order_x2_1)
            diff2_x1 = func_values_vector_x1 - (grad_one_order_x1_2 + eps_value * grad_second_order_x1_2 + eps_value ** 2 * grad_third_order_x1_2)
            diff2_x2 = func_values_vector_x2 - (grad_one_order_x2_2 + eps_value * grad_second_order_x2_2 + eps_value ** 2 * grad_third_order_x2_2)
            # 计算 H1 半范数误差的平方（梯度项），累加到对应epsilon
            h1_seminorm1_sq[idx] += np.sum(w.cpu().numpy() * (diff1_x1 ** 2 + diff1_x2 ** 2))
            h1_seminorm2_sq[idx] += np.sum(w.cpu().numpy() * (diff2_x1 ** 2 + diff2_x2 ** 2))
            
            # 删除for idx, eps in enumerate(epsilons):过程中的中间变量
            # 清理无用的中间变量，回收内存
            del grad_one_order_x1_1
            del grad_one_order_x2_1
            del grad_one_order_x1_2
            del grad_one_order_x2_2
            del grad_second_order_x1_1
            del grad_second_order_x2_1
            del grad_second_order_x1_2
            del grad_second_order_x2_2
            del grad_third_order_x1_2
            del grad_third_order_x2_2
            del func_values_matrix_x1
            del func_values_matrix_x2
            del func_values_vector_x1
            del func_values_vector_x2
            del diff1_x1
            del diff1_x2
            del diff2_x1
            del diff2_x2
    
    # 计算最终的H1半范数误差（所有rank累加后取平方根）
    h1_seminorm1_arr = np.sqrt(h1_seminorm1_sq)
    h1_seminorm2_arr = np.sqrt(h1_seminorm2_sq)
    
    # 打印结果
    epsilons_np = np.array(epsilons)
    inv_epsilons = 1 / epsilons_np
    for i, eps_value in enumerate(inv_epsilons):
        print(f'|u_{{fem}} - U_1|_H^1 for eps={eps_value}:', h1_seminorm1_arr[i])
        print(f'|u_{{fem}} - U_2|_H^1 for eps={eps_value}:', h1_seminorm2_arr[i])
        print('--------------------------------')


plt.figure(figsize=(8, 6))
epsilons_np = np.array(epsilons)
inv_epsilons = 1 / epsilons_np

plt.loglog(inv_epsilons, h1_seminorm1_arr, 's-', label=r'$|u_\epsilon - U_1^*|_{H^1}$')
plt.loglog(inv_epsilons, h1_seminorm2_arr, '^-', label=r'$|u_\epsilon - U_2^*|_{H^1}$')

# 画参考线
ref_x = np.array([inv_epsilons[0], inv_epsilons[-1]])

# 1阶参考线（斜率1）
ref1_y0 = h1_seminorm1_arr[0]
ref1 = ref1_y0 * (ref_x / ref_x[0]) ** 1
plt.loglog(ref_x, ref1, 'k--', alpha=0.7)# , label=r'ref: slope=1'


# 2阶参考线（斜率2）
ref2_y0 = h1_seminorm2_arr[0]
ref2 = ref2_y0 * (ref_x / ref_x[0]) ** 2
plt.loglog(ref_x, ref2, 'k--', alpha=0.7)# , label=r'ref: slope=2'


plt.xlabel(r'$\epsilon$', fontsize=15)
plt.ylabel(r'$H^1$ error', fontsize=15)
plt.title(r'$H^1$ error convergence vs $\epsilon$', fontsize=17)
plt.tick_params(axis='both', which='both', labelsize=13)  # 增大横纵坐标刻度数字的字体大小
plt.grid(True, which="both", ls="--")
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig('png/h1_error.png')
plt.close()
