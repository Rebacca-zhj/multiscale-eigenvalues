import torch
import numpy as np
import scipy.io as sio
import os
# 【关键1】解决CUDA显存碎片化，必须放在导入torch之前
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from tnn import *
from integration import *
from quadrature import *
import matplotlib.pyplot as plt
import gc

pi = 3.14159265358979323846
dtype = torch.double
device = 'cuda:5'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

dim = 1
p_A = 2
p_chi = 20
p_u0 = 20
p_u1 = 20
p_u2 = 20

a, b = 0, 1

x1, w1 = composite_quadrature_1d(4, 0, 1, 200, device=device)
N = len(x1)
x2, w2 = composite_quadrature_1d(4, 0, 1, 200, device=device)

w12 = torch.stack((w1, w2), dim=0)

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

model_u0 = torch.load("./model_u0/d1_ex1_final_u0.pkl", map_location=device).eval()  
model_u1_x = torch.load('./model_u1/d1_ex1_final_u1_x.pkl', map_location=device).eval()  
model_u1_y = torch.load('./model_u1/d1_ex1_final_u1_y.pkl', map_location=device).eval()  
model_u2_x = torch.load('./model_u2/d1_ex1_final_u2_x.pkl', map_location=device).eval()  
model_u2_y = torch.load('./model_u2/d1_ex1_final_u2_y.pkl', map_location=device).eval()  

load_tilde_u1 = True
if load_tilde_u1:
    model_tilde_u1 = torch.load('./model_tilde_u1/d1_ex1_final_tilde_u1.pkl', map_location=device).eval()  
load_delta_u2 = True
if load_delta_u2:
    model_delta_u2_x = torch.load('./model_delta_u2/d1_ex1_final_delta_u2_x.pkl', map_location=device).eval()  
    model_delta_u2_y = torch.load('./model_delta_u2/d1_ex1_final_delta_u2_y.pkl', map_location=device).eval()  
load_u3 = True
if load_u3:
    model_u3_x = torch.load('./model_u3/d1_ex1_final_u3_x.pkl', map_location=device).eval()  
    model_u3_y = torch.load('./model_u3/d1_ex1_final_u3_y.pkl', map_location=device).eval()  
load_tilde_u2 = True
if load_tilde_u2:
    model_tilde_u2 = torch.load('./model_tilde_u2/d1_ex1_final_tilde_u2.pkl', map_location=device).eval()  

with torch.no_grad():
    # 归一化及函数分量准备
    u0_norm_phi = model_u0(w1, x1, normed=False)
    u0_norm = torch.sqrt(torch.sum(w1 * u0_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    alpha_u0 = model_u0.scaling_par()
    
    u1_x_norm_phi = model_u1_x(w1, x1, normed=False)
    u1_x_norm = torch.sqrt(torch.sum(w1 * u1_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u1_alpha = model_u1_x.scaling_par()
    u1_y_norm_phi = model_u1_y(w2, x2, normed=False)
    u1_y_norm = torch.sqrt(torch.sum(w2 * u1_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u1_y_shift = model_u1_y(dummy, x_error_0, need_grad=0, normed=False)
    u1_y_shift = u1_y_shift / u1_y_norm
    
    u2_x_norm_phi = model_u2_x(w1, x1, normed=False)
    u2_x_norm = torch.sqrt(torch.sum(w1 * u2_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u2_alpha = model_u2_x.scaling_par()
    u2_y_norm_phi = model_u2_y(w2, x2, normed=False)
    u2_y_norm = torch.sqrt(torch.sum(w2 * u2_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
    u2_y_shift = model_u2_y(dummy, x_error_0, need_grad=0, normed=False)
    u2_y_shift = u2_y_shift / u2_y_norm

    del u0_norm_phi, u1_x_norm_phi, u1_y_norm_phi, u2_x_norm_phi, u2_y_norm_phi
    # 计算其他不依赖于rank的norm
    if load_tilde_u1:
        tilde_u1_norm_phi = model_tilde_u1(w1, x1, normed=False)
        tilde_u1_norm = torch.sqrt(torch.sum(w1 * tilde_u1_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        alpha_tilde_u1 = model_tilde_u1.scaling_par()
        del tilde_u1_norm_phi
    
    if load_delta_u2:
        delta_u2_x_norm_phi = model_delta_u2_x(w1, x1, normed=False)
        delta_u2_x_norm = torch.sqrt(torch.sum(w1 * delta_u2_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        delta_u2_alpha = model_delta_u2_x.scaling_par()
        delta_u2_y_norm_phi = model_delta_u2_y(w2, x2, normed=False)
        delta_u2_y_norm = torch.sqrt(torch.sum(w2 * delta_u2_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        delta_u2_y_shift = model_delta_u2_y(dummy, x_error_0, need_grad=0, normed=False)
        delta_u2_y_shift = delta_u2_y_shift / delta_u2_y_norm
        del delta_u2_x_norm_phi, delta_u2_y_norm_phi

    if load_u3:
        u3_x_norm_phi = model_u3_x(w1, x1, normed=False)
        u3_x_norm = torch.sqrt(torch.sum(w1 * u3_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        u3_alpha = model_u3_x.scaling_par()
        u3_y_norm_phi = model_u3_y(w2, x2, normed=False)
        u3_y_norm = torch.sqrt(torch.sum(w2 * u3_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        u3_y_shift = model_u3_y(dummy, x_error_0, need_grad=0, normed=False)
        u3_y_shift = u3_y_shift / u3_y_norm
        del u3_x_norm_phi, u3_y_norm_phi

    if load_tilde_u2:
        tilde_u2_norm_phi = model_tilde_u2(w1, x1, need_grad=0, normed=False)
        tilde_u2_norm = torch.sqrt(torch.sum(w1 * tilde_u2_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        tilde_u2_alpha = model_tilde_u2.scaling_par()
        del tilde_u2_norm_phi

    # 【关键5】清空GPU缓存，释放已删除张量的显存
    # torch.cuda.empty_cache()
    epsilons = [3,5,8,10,12,15,20,30,50]
    # 初始化累加器（每个epsilon一个值）
    h1_seminorm0_sq = np.zeros(len(epsilons))
    h1_seminorm1_sq = np.zeros(len(epsilons))
    h1_seminorm2_sq = np.zeros(len(epsilons))
    h1_seminorm3_sq = np.zeros(len(epsilons))
    
    # 用于保存绘图数据
    all_coords = []
    

    for i_rank in range(0,2):
        coord_x_np = np.loadtxt(f'./FEM_data/e_3/coord_R14_rank{i_rank}.txt', unpack=True)
        coord_x = torch.from_numpy(coord_x_np).to(device)
        del coord_x_np
        w_np = np.loadtxt(f'./FEM_data/e_3/w_R14_rank{i_rank}.txt', unpack=True)
        w = torch.from_numpy(w_np).to(device)

        u0_phi, u0_grad_phi = model_u0(dummy, coord_x, need_grad=1, normed=False)
        u0_phi = u0_phi / u0_norm
        u0_grad_phi = u0_grad_phi / u0_norm
        u0 = torch.einsum('p,pn->n', alpha_u0, torch.prod(u0_phi, dim=0))
        u0_grad_x = torch.einsum('p,pn->n', alpha_u0, torch.prod(u0_grad_phi, dim=0))
        del u0_phi, u0_grad_phi

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
            tilde_u1 = torch.einsum('p,pn->n', alpha_tilde_u1, torch.prod(tilde_u1_phi, dim=0))
            tilde_u1_grad_x = torch.einsum('p,pn->n', alpha_tilde_u1, torch.prod(tilde_u1_grad_phi, dim=0))
            del tilde_u1_phi, tilde_u1_grad_phi
        else:
            tilde_u1 = torch.zeros_like(u0)
            tilde_u1_grad_x = torch.zeros_like(u0_grad_x)

        if load_delta_u2:
            delta_u2_x_phi, delta_u2_x_grad_phi = model_delta_u2_x(dummy, coord_x, need_grad=1, normed=False)
            delta_u2_x_phi = delta_u2_x_phi / delta_u2_x_norm
            delta_u2_x_grad_phi = delta_u2_x_grad_phi / delta_u2_x_norm

        if load_u3:
            u3_x_phi = model_u3_x(dummy, coord_x, normed=False)
            u3_x_phi = u3_x_phi / u3_x_norm

        if load_tilde_u2:
            tilde_u2_phi, tilde_u2_grad_phi = model_tilde_u2(dummy, coord_x, need_grad=1, normed=False)
            tilde_u2_phi = tilde_u2_phi / tilde_u2_norm
            tilde_u2_grad_phi = tilde_u2_grad_phi / tilde_u2_norm
            tilde_u2 = torch.sum(tilde_u2_alpha.unsqueeze(dim=-1) * torch.prod(tilde_u2_phi, dim=0), dim=0)
            tilde_u2_grad_x = torch.sum(tilde_u2_alpha.unsqueeze(dim=-1) * torch.prod(tilde_u2_grad_phi, dim=0), dim=0)
            del tilde_u2_phi, tilde_u2_grad_phi
        else:
            tilde_u2 = torch.zeros_like(u0)
            tilde_u2_grad_x = torch.zeros_like(u0_grad_x)

        for idx, eps in enumerate(epsilons):
            # 计算一阶展开项
            coord_y = coord_x * eps - torch.floor(coord_x * eps)
            u1_y_phi, u1_y_grad_phi = model_u1_y(dummy, coord_y, need_grad=1, normed=False)
            u1_y_phi = u1_y_phi / u1_y_norm
            u1_y_grad_phi = u1_y_grad_phi / u1_y_norm
            u1 = torch.sum((u1_alpha.unsqueeze(dim=-1) * torch.prod(u1_x_phi, dim=0) * (torch.prod(u1_y_phi, dim=0) - torch.prod(u1_y_shift, dim=0))), dim=0)
            u1_grad_x = torch.sum((u1_alpha.unsqueeze(dim=-1) * torch.prod(u1_x_grad_phi, dim=0) * (torch.prod(u1_y_phi, dim=0) - torch.prod(u1_y_shift, dim=0))), dim=0)
            u1_grad_y = torch.sum((u1_alpha.unsqueeze(dim=-1) * torch.prod(u1_x_phi, dim=0) * torch.prod(u1_y_grad_phi, dim=0)), dim=0)

            u2_y_phi, u2_y_grad_phi = model_u2_y(dummy, coord_y, need_grad=1, normed=False)
            u2_y_phi = u2_y_phi / u2_y_norm
            u2_y_grad_phi = u2_y_grad_phi / u2_y_norm
            u2 = torch.sum((u2_alpha.unsqueeze(dim=-1) * torch.prod(u2_x_phi, dim=0) * (torch.prod(u2_y_phi, dim=0) - torch.prod(u2_y_shift, dim=0))), dim=0)
            u2_grad_x = torch.sum((u2_alpha.unsqueeze(dim=-1) * torch.prod(u2_x_grad_phi, dim=0) * (torch.prod(u2_y_phi, dim=0) - torch.prod(u2_y_shift, dim=0))), dim=0)
            u2_grad_y = torch.sum((u2_alpha.unsqueeze(dim=-1) * torch.prod(u2_x_phi, dim=0) * torch.prod(u2_y_grad_phi, dim=0)), dim=0)

            if load_delta_u2:
                delta_u2_y_phi, delta_u2_y_grad_phi = model_delta_u2_y(dummy, coord_y, need_grad=1, normed=False)
                delta_u2_y_phi = delta_u2_y_phi / delta_u2_y_norm
                delta_u2_y_grad_phi = delta_u2_y_grad_phi / delta_u2_y_norm

                delta_u2 = torch.sum((delta_u2_alpha.unsqueeze(dim=-1) * torch.prod(delta_u2_x_phi, dim=0) * (torch.prod(delta_u2_y_phi, dim=0) - torch.prod(delta_u2_y_shift, dim=0))), dim=0)
                delta_u2_grad_x = torch.sum((delta_u2_alpha.unsqueeze(dim=-1) * torch.prod(delta_u2_x_grad_phi, dim=0) * (torch.prod(delta_u2_y_phi, dim=0) - torch.prod(delta_u2_y_shift, dim=0))), dim=0)
                delta_u2_grad_y = torch.sum((delta_u2_alpha.unsqueeze(dim=-1) * torch.prod(delta_u2_x_phi, dim=0) * torch.prod(delta_u2_y_grad_phi, dim=0)), dim=0)
            else:
                delta_u2 = torch.zeros_like(u0)
                delta_u2_grad_x = torch.zeros_like(u0_grad_x)
                delta_u2_grad_y = torch.zeros_like(u0_grad_x)

            if load_u3:
                u3_y_phi, u3_y_grad_phi = model_u3_y(dummy, coord_y, need_grad=1, normed=False)
                u3_y_phi = u3_y_phi / u3_y_norm
                u3_y_grad_phi = u3_y_grad_phi / u3_y_norm
                u3 = torch.sum((u3_alpha.unsqueeze(dim=-1) * torch.prod(u3_x_phi, dim=0) * (torch.prod(u3_y_phi, dim=0) - torch.prod(u3_y_shift, dim=0))), dim=0)
                u3_grad_y = torch.sum((u3_alpha.unsqueeze(dim=-1) * torch.prod(u3_x_phi, dim=0) * torch.prod(u3_y_grad_phi, dim=0)), dim=0)
            else:
                u3 = torch.zeros_like(u0)
                u3_grad_y = torch.zeros_like(u0_grad_x)
                
            # 计算梯度
            grad_one_order_x = (u0_grad_x).cpu().numpy()
            
            grad_second_order_x_1 = (u0_grad_x + u1_grad_y).cpu().numpy()
            grad_second_order_x_2 = (u1_grad_x + tilde_u1_grad_x).cpu().numpy()
            
            grad_third_order_x_1 = (u0_grad_x + u1_grad_y).cpu().numpy()
            grad_third_order_x_2 = (u1_grad_x + tilde_u1_grad_x + u2_grad_y + delta_u2_grad_y).cpu().numpy()
            grad_third_order_x_3 = (u2_grad_x + delta_u2_grad_x + tilde_u2_grad_x).cpu().numpy()
            grad_third_order_x_4 = (u2_grad_x + delta_u2_grad_x + tilde_u2_grad_x + u3_grad_y).cpu().numpy()
            
            # 加载FEM参考解
            func_values_matrix_x = np.loadtxt(f'./FEM_data/e_{eps}/u_d1_R14_rank{i_rank}.txt')
            func_values_vector_x = func_values_matrix_x.reshape(-1) 

            eps_value = 1 / eps
            diff0_x = func_values_vector_x - grad_one_order_x
            diff1_x = func_values_vector_x - (grad_second_order_x_1 + eps_value * grad_second_order_x_2)
            diff2_x = func_values_vector_x - (grad_third_order_x_1 + eps_value * grad_third_order_x_2 + eps_value ** 2 * grad_third_order_x_3)
            diff3_x = func_values_vector_x - (grad_third_order_x_1 + eps_value * grad_third_order_x_2 + eps_value ** 2 * grad_third_order_x_4)

            # 计算 H1 半范数误差的平方（梯度项），累加到对应epsilon
            h1_seminorm0_sq[idx] += np.sum(w.cpu().numpy() * (diff0_x ** 2))
            h1_seminorm1_sq[idx] += np.sum(w.cpu().numpy() * (diff1_x ** 2))
            h1_seminorm2_sq[idx] += np.sum(w.cpu().numpy() * (diff2_x ** 2))
            h1_seminorm3_sq[idx] += np.sum(w.cpu().numpy() * (diff3_x ** 2))
            
            # 每个 epsilon 循环结束后释放临时 GPU 张量
            gc.collect()
            torch.cuda.empty_cache()
        
        # # 当前 rank 处理完毕，清理不必要的大张量
        # del coord_x, w
        # del u0, u0_grad_x
        # del u1_x_phi
        # del u2_x_phi, u2_x_grad_phi
        # del u1_grad_x, u1_grad_y
        # del u2_grad_x, u2_grad_y
        # if load_tilde_u1:
        #     del tilde_u1, tilde_u1_grad_x
        # if load_delta_u2:
        #     del delta_u2_x_phi, delta_u2_x_grad_phi
        # if load_u3:
        #     del u3_x_phi
        # if load_tilde_u2:
        #     del tilde_u2, tilde_u2_grad_x
        # gc.collect()
        torch.cuda.empty_cache()
        print(f"Rank {i_rank} 处理完成")
    
    # 计算最终的H1半范数误差（所有rank累加后取平方根）
    h1_seminorm0_arr = np.sqrt(h1_seminorm0_sq)
    h1_seminorm1_arr = np.sqrt(h1_seminorm1_sq)
    h1_seminorm2_arr = np.sqrt(h1_seminorm2_sq)
    h1_seminorm3_arr = np.sqrt(h1_seminorm3_sq)
    
    # 打印结果
    epsilons_np = np.array(epsilons)
    inv_epsilons = 1 / epsilons_np
    for i, eps_value in enumerate(inv_epsilons):
        print(f'|u_e - u_0|_H^1 for eps={eps_value}:', h1_seminorm0_arr[i])
        print(f'|u_e - (u_0+e* u_1)|_H^1 for eps={eps_value}:', h1_seminorm1_arr[i])
        print(f'|u_e - (u_0+e*u_1+e^2 * u_2)|_H^1 for eps={eps_value}:', h1_seminorm2_arr[i])
        print(f'|u_e - (u_0+e * u_1+e^2 * u_2 + e^3 * u_3|_H^1 for eps={eps_value}:', h1_seminorm3_arr[i])
        print('--------------------------------')


plt.figure(figsize=(8, 6))
epsilons_np = np.array(epsilons)
inv_epsilons = 1 / epsilons_np

plt.loglog(inv_epsilons, h1_seminorm0_arr, 'o-', label=r'$|\nabla u_\epsilon - u_0|_{H^1}$')
plt.loglog(inv_epsilons, h1_seminorm1_arr, 's-', label=r'$|\nabla u_\epsilon - (u_0 + u_1)|_{H^1}$')
plt.loglog(inv_epsilons, h1_seminorm2_arr, '^-', label=r'$|\nabla u_\epsilon - (u_0 + u_1 + u_2)|_{H^1}$')

# 画参考线
ref_x = np.array([inv_epsilons[0], inv_epsilons[-1]])

# 1阶参考线（斜率1）
ref1_y0 = h1_seminorm0_arr[0]
ref1 = ref1_y0 * (ref_x / ref_x[0]) ** 0
plt.loglog(ref_x, ref1, 'k--', alpha=0.7, label=r'ref: slope=0')

ref2_y0 = h1_seminorm1_arr[0]
ref2 = ref2_y0 * (ref_x / ref_x[0]) ** 1
plt.loglog(ref_x, ref2, 'k--', alpha=0.7, label=r'ref: slope=1')

# 2阶参考线（斜率2）
ref3_y0 = h1_seminorm2_arr[0]
ref3 = ref3_y0 * (ref_x / ref_x[0]) ** 2
plt.loglog(ref_x, ref3, 'k:', alpha=0.7, label=r'ref: slope=2')

plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$H^1$ error')
plt.title(r'$H^1$ error convergence vs $\epsilon$')
plt.legend(fontsize=8)  # 让label更小
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig('png/H1_error_1.png')
plt.close()

# 计算误差曲线在 log-log 坐标下的斜率（对数尺度下斜率即为误差阶）

# 取倒数，因为横坐标为 inv_epsilons = 1/epsilon
log_inv_eps = np.log(inv_epsilons)
log_h1_0 = np.log(h1_seminorm0_arr)
log_h1_1 = np.log(h1_seminorm1_arr)
log_h1_2 = np.log(h1_seminorm2_arr)
log_h1_3 = np.log(h1_seminorm3_arr)

# 拟合一次多项式，返回斜率和截距
slope0, intercept0 = np.polyfit(log_inv_eps, log_h1_0, 1)
slope1, intercept1 = np.polyfit(log_inv_eps, log_h1_1, 1)
slope2, intercept2 = np.polyfit(log_inv_eps, log_h1_2, 1)
slope3, intercept3 = np.polyfit(log_inv_eps, log_h1_3, 1)

print(f"H1 Error0 (||u_e - u_0||_H1) log-log 斜率: {slope0:.4f}")
print(f"H1 Error1 (||u_e - (u_0+e*û1)||_H1) log-log 斜率: {slope1:.4f}")
print(f"H1 Error2 (||u_e - (u_0+e*û1+e^2*û2)||_H1) log-log 斜率: {slope2:.4f}")
print(f"H1 Error3 (||u_e - (u_0+e*û1+e^2*û2)+e^3*u_3||_H1) log-log 斜率: {slope3:.4f}")