import torch
import math
import numpy as np

torch.set_printoptions(profile="full", precision=16)
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import torch.optim as optim
import gc
from tnn import *
from integration import *
from quadrature import *

import itertools
import os
import sys
import time
import importlib.util

pi = 3.14159265358979323846
dtype = torch.double
device = 'cuda:6'
name = 'd2_ex4_test24'

# sizes_periodic = [1, 2, 50, 50, 50, 40] #test18 
# sizes_periodic = [1, 2, 50, 50, 50, 50, 40] #test19 
# sizes_periodic = [1, 5, 80, 80, 40] #test20 
# sizes_periodic = [1, 5, 80, 80, 80, 40] #test21 
sizes_periodic = [1, 5, 80, 80, 80, 80, 40] #test22 0.0010397320004159939（✅）
# sizes_periodic = [1, 8, 100, 100, 40] #test23 
# sizes_periodic = [1, 8, 100, 100, 100, 40] #test24 
# sizes_periodic = [1, 8, 100, 100, 100, 100, 40] #test25 
# sizes_periodic = [1, 8, 150, 150, 40] #test26 
# sizes_periodic = [1, 8, 150, 150, 150, 40] #test27 
# sizes_periodic = [1, 8, 150, 150, 150, 150, 40] #test28 
# sizes_periodic = [1, 2, 50, 50, 50, 50, 50, 40] #test29 

print(sizes_periodic)

p_u0 = 20
p_u1 = 20
p_u2 = 40
dim = 2
a, b = 0, 1

x1, w1 = composite_quadrature_1d(4, 0, 1, 400, dtype=dtype, device=device)
N = len(x1)
x2, w2 = composite_quadrature_1d(4, 0, 1, 400, dtype=dtype, device=device)
x3, w3 = composite_quadrature_1d(4, 0, 1, 400, dtype=dtype, device=device)
x4, w4 = composite_quadrature_1d(4, 0, 1, 400, dtype=dtype, device=device)

w1234 = torch.stack((w1, w2, w3, w4), dim=0)
w12 = torch.stack((w1, w2), dim=0)
w34 = torch.stack((w3, w4), dim=0)

dummy = torch.ones((1, 1), dtype=dtype, device=device)
x_error_0 = torch.zeros(1, dtype=dtype, device=device)

F = torch.ones((2, 2, N), dtype=dtype, device=device)
F[0, 0, :] = torch.sin(2 * pi * x1)
F[1, 1, :] = torch.sin(2 * pi * x2)
alpha_F = torch.tensor([30, 20], dtype=dtype, device=device)
norm_F = Int2TNN(w12, alpha_F, F, alpha_F, F, if_sum=True)

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


# 加载已训练好的模型
model_u0 = torch.load('./model_u0/d2_ex4_final_u0.pkl', map_location=device)
model_u1_x = torch.load('./model_u1/d2_ex4_final_u1_x.pkl', map_location=device)
model_u1_y = torch.load('./model_u1/d2_ex4_final_u1_y.pkl', map_location=device)


with torch.no_grad():
    expand_F = torch.cat((F, torch.ones((2, 2, N), dtype=dtype, device=device)), dim=0)

    u0_phi, u0_grad_phi, u0_grad_grad_phi = model_u0(w1, x1, need_grad=2)
    alpha_u0 = model_u0.scaling_par()
    
    # expand_u0_phi = torch.cat((u0_phi, torch.ones((dim, p_u0, N), dtype=dtype, device=device)), dim=0)
    expand_u0_grad_x1 = torch.cat((u0_grad_phi[0:1, :, :], u0_phi[1:2, :, :], torch.ones((dim, p_u0, N), dtype=dtype, device=device)), dim=0)
    expand_u0_grad_x2 = torch.cat((u0_phi[0:1, :, :], u0_grad_phi[1:2, :, :], torch.ones((dim, p_u0, N), dtype=dtype, device=device)), dim=0)
    expand_u0_grad_grad_x1 = torch.cat((u0_grad_grad_phi[0:1, :, :], u0_phi[1:2, :, :], torch.ones((dim, p_u0, N), dtype=dtype, device=device)), dim=0)
    expand_u0_grad_grad_x2 = torch.cat((u0_phi[0:1, :, :], u0_grad_grad_phi[1:2, :, :], torch.ones((dim, p_u0, N), dtype=dtype, device=device)), dim=0)
    
    # u1 的值和导数
    u1_phi_x, u1_grad_phi_x = model_u1_x(w1, x1, need_grad=1)
    u1_phi_y, u1_grad_phi_y = model_u1_y(w3, x3, need_grad=1)
    alpha_u1 = model_u1_x.scaling_par()
    u1_norm_phi_y = model_u1_y(w3, x3, normed=False)
    u1_norm = torch.sqrt(torch.sum(w3 * u1_norm_phi_y**2, dim=-1)).unsqueeze(dim=-1)
    u1_y_shift = model_u1_y(dummy, x_error_0, need_grad=0, normed=False)
    u1_y_shift = u1_y_shift / u1_norm

    expand_u1_grad_x1 = torch.cat((u1_grad_phi_x[0:1, :, :], u1_phi_x[1:2, :, :], u1_phi_y), dim=0)
    expand_u1_grad_x1_shift = torch.cat((u1_grad_phi_x[0:1, :, :], u1_phi_x[1:2, :, :], u1_y_shift.expand(2, p_u1, N)), dim=0)
    expand_u1_grad_x2 = torch.cat((u1_phi_x[0:1, :, :], u1_grad_phi_x[1:2, :, :], u1_phi_y), dim=0)
    expand_u1_grad_x2_shift = torch.cat((u1_phi_x[0:1, :, :], u1_grad_phi_x[1:2, :, :], u1_y_shift.expand(2, p_u1, N)), dim=0)
    expand_u1_grad_y1 = torch.cat((u1_phi_x, u1_grad_phi_y[0:1, :, :], u1_phi_y[1:2, :, :]), dim=0)
    expand_u1_grad_y2 = torch.cat((u1_phi_x, u1_phi_y[0:1, :, :], u1_grad_phi_y[1:2, :, :]), dim=0)
    expand_u1_grad_x1y1 = torch.cat((u1_grad_phi_x[0:1, :, :], u1_phi_x[1:2, :, :], u1_grad_phi_y[0:1, :, :], u1_phi_y[1:2, :, :]), dim=0)
    expand_u1_grad_x2y2 = torch.cat((u1_phi_x[0:1, :, :], u1_grad_phi_x[1:2, :, :], u1_phi_y[0:1, :, :], u1_grad_phi_y[1:2, :, :]), dim=0)
    
    # 构建 RHS 的各项    
    # 1. ∇_{x_1}A · ∇_{x_1}u_0
    part1_phi = torch.einsum('dpn, dqn->dpqn', A_grad_x1, expand_u0_grad_x1).flatten(start_dim=1, end_dim=2)
    alpha_part1 = torch.outer(alpha_A_grad_x1, alpha_u0).view(-1)
    
    # 2. A∇_{x_1x_1}u_0
    part2_phi = torch.einsum('dpn, dqn->dpqn', A, expand_u0_grad_grad_x1).flatten(start_dim=1, end_dim=2)
    alpha_part2 = torch.outer(alpha_A, alpha_u0).view(-1)
    
    # 3. ∇_{x_1}A∇_{y_1}u_1
    part3_phi = torch.einsum('dpn, dqn->dpqn', A_grad_x1, expand_u1_grad_y1).flatten(start_dim=1, end_dim=2)
    alpha_part3 = torch.outer(alpha_A_grad_x1, alpha_u1).view(-1)
    
    # 4. ∇_{y_1}A∇_{x_1}u_1
    part4_phi = torch.einsum('dpn, dqn->dpqn', A_grad_y1, expand_u1_grad_x1).flatten(start_dim=1, end_dim=2)
    part4_shift = torch.einsum('dpn, dqn->dpqn', A_grad_y1, expand_u1_grad_x1_shift).flatten(start_dim=1, end_dim=2)
    alpha_part4 = torch.outer(alpha_A_grad_y1, alpha_u1).view(-1)
    
    # 5. 2A∇_{x_1y_1}u_1
    part5_phi = torch.einsum('dpn, dqn->dpqn', A, expand_u1_grad_x1y1).flatten(start_dim=1, end_dim=2)
    alpha_part5 = 2 * torch.outer(alpha_A, alpha_u1).view(-1)
    
    # 6. ∇_{x_2}A ∇_{x_2}u_0
    part6_phi = torch.einsum('dpn, dqn->dpqn', A_grad_x2, expand_u0_grad_x2).flatten(start_dim=1, end_dim=2)
    alpha_part6 = torch.outer(alpha_A_grad_x2, alpha_u0).view(-1)
    
    # 7. A∇_{x_2x_2}u_0
    part7_phi = torch.einsum('dpn, dqn->dpqn', A, expand_u0_grad_grad_x2).flatten(start_dim=1, end_dim=2)
    alpha_part7 = torch.outer(alpha_A, alpha_u0).view(-1)
    
    # 8. ∇_{x_2}A ∇_{y_2}u_1
    part8_phi = torch.einsum('dpn, dqn->dpqn', A_grad_x2, expand_u1_grad_y2).flatten(start_dim=1, end_dim=2)
    alpha_part8 = torch.outer(alpha_A_grad_x2, alpha_u1).view(-1)
    
    # 9. ∇_{y_2}A∇_{x_2}u_1
    part9_phi = torch.einsum('dpn, dqn->dpqn', A_grad_y2, expand_u1_grad_x2).flatten(start_dim=1, end_dim=2)
    part9_shift = torch.einsum('dpn, dqn->dpqn', A_grad_y2, expand_u1_grad_x2_shift).flatten(start_dim=1, end_dim=2)
    alpha_part9 = torch.outer(alpha_A_grad_y2, alpha_u1).view(-1)
    
    # 10. 2A∇_{x_2y_2}u_1
    part10_phi = torch.einsum('dpn, dqn->dpqn', A, expand_u1_grad_x2y2).flatten(start_dim=1, end_dim=2)
    alpha_part10 = 2 * torch.outer(alpha_A, alpha_u1).view(-1)
    
    # 组合所有 RHS 项
    rhs_phi = torch.cat((expand_F, part1_phi, part2_phi, part3_phi, part4_phi, part4_shift, part5_phi, 
                         part6_phi, part7_phi, part8_phi, part9_phi, part9_shift, part10_phi), dim=1)
    alpha_rhs = torch.cat((alpha_F, alpha_part1, alpha_part2, alpha_part3, alpha_part4, -alpha_part4, alpha_part5,
                           alpha_part6, alpha_part7, alpha_part8, alpha_part9, -alpha_part9, alpha_part10))
    norm_rhs = Int2TNN(w1234, alpha_rhs, rhs_phi, alpha_rhs, rhs_phi)
    print('norm_rhs =', norm_rhs.item())
    print('rhs_phi',Int1TNN(w1234, alpha_rhs, rhs_phi))
    print('alpha_rhs',alpha_rhs.shape)

def criterion_u2(model_u2_x, model_u2_y):
    """
    损失函数：强形式
    ||-∇_y · (A∇_y u_2) - rhs||_{L2}^2
    
    其中 LHS = ∇_{y_1}A · ∇_{y_1}u_2 + A∇_{y_1y_1}u_2 + ∇_{y_2}A · ∇_{y_2}u_2 + A∇_{y_2y_2}u_2
    """
    u2_phi_x = model_u2_x(w1, x1, need_grad=0)
    u2_phi_y, u2_grad_phi_y, u2_grad_grad_phi_y = model_u2_y(w3, x3, need_grad=2)
    
    # 构建 LHS 的各项
    # 1. A∇_{y_1y_1}u_2
    u2_grad_grad_y1 = torch.cat((u2_phi_x, u2_grad_grad_phi_y[0:1], u2_phi_y[1:2]), dim=0)
    operand_A_u2_grad_grad_y1 = torch.einsum('dpn, dqn->dpqn', A, u2_grad_grad_y1)
    
    # 2. ∇_{y_1}A · ∇_{y_1}u_2
    u2_grad_y1 = torch.cat((u2_phi_x, u2_grad_phi_y[0:1], u2_phi_y[1:2]), dim=0)
    operand_A_grad_y1_u2_grad_y1 = torch.einsum('dpn, dqn->dpqn', A_grad_y1, u2_grad_y1)
    
    # 3. A∇_{y_2y_2}u_2
    u2_grad_grad_y2 = torch.cat((u2_phi_x, u2_phi_y[0:1], u2_grad_grad_phi_y[1:2]), dim=0)
    operand_A_u2_grad_grad_y2 = torch.einsum('dpn, dqn->dpqn', A, u2_grad_grad_y2)
    
    # 4. ∇_{y_2}A · ∇_{y_2}u_2
    u2_grad_y2 = torch.cat((u2_phi_x, u2_phi_y[0:1], u2_grad_phi_y[1:2]), dim=0)
    operand_A_grad_y2_u2_grad_y2 = torch.einsum('dpn, dqn->dpqn', A_grad_y2, u2_grad_y2)
    
    # 组合 LHS 的所有项
    my_operand_phi = torch.cat((operand_A_u2_grad_grad_y1, operand_A_grad_y1_u2_grad_y1,
                                operand_A_u2_grad_grad_y2, operand_A_grad_y2_u2_grad_y2), dim=1)
    my_operand_alpha = torch.cat((alpha_A, alpha_A_grad_y1, alpha_A, alpha_A_grad_y2))
    
    # 计算矩阵：my_matrix = ∫ (my_operand_phi, my_operand_phi)
    # 使用自定义 autograd.Function (替代原始的 einsum + prod 实现，约 6 倍加速)
    my_matrix = torch.sum(
        torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(
            torch.einsum('dn, dpqn, djkn->dpjqk', w1234, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    
    # 计算向量：my_vector = ∫ (my_operand_phi, rhs)
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_rhs).unsqueeze(dim=-1) * torch.prod(
        torch.einsum('dn, dpqn, dkn->dpkq', w1234, my_operand_phi, rhs_phi), dim=0), dim=(0, 1))
    
    # 如果矩阵病态，使用 lstsq 替代 linalg.solve
    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    
    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_rhs)
    
    model_u2_x.scaling_par().data.copy_(my_alpha.detach())
    
    return loss


def train_u2(model_u2_x, model_u2_y, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        
        for e in range(epochs):
            loss = criterion_u2(model_u2_x, model_u2_y)
            if e == 0:
                print('\n Step u2 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                print('lr: ', scheduler.get_last_lr()[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_u2(model_u2_x, model_u2_y)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])
                    torch.save(model_u2_x, "./model_u2/{}_final_u2_x.pkl".format(name))
                    torch.save(model_u2_y, "./model_u2/{}_final_u2_y.pkl".format(name))

    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_u2(model_u2_x, model_u2_y)
            optimizer.zero_grad()
            loss.backward()
            return loss

        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_u2(model_u2_x, model_u2_y)
                    print('\n Step u2 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_u2(model_u2_x, model_u2_y)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())




if __name__ == '__main__':
    activation = TNN_Sin

    if not os.path.exists("model_u2"):
        os.mkdir("model_u2")

    # model_u2_x = torch.load("./model_u2/d2_ex4_final_u2_x.pkl", map_location=device)
    # model_u2_y = torch.load("./model_u2/d2_ex4_final_u2_y.pkl", map_location=device)
    # loss = criterion_u2(model_u2_x, model_u2_y)
    # print('loss =', loss.item())


    model_u2_x = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=True).to(dtype).to(device)
    model_u2_y = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=False).to(dtype).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, itertools.chain(model_u2_x.parameters(), model_u2_y.parameters())),
        lr=1e-2)

    epochs = 50000
    begin_time = time.time()
    train_u2(model_u2_x, model_u2_y, optimizer, 'Adam', epochs)
    end_time = time.time()
    print('\n')
    print('*' * 40)
    print('Training done!')
    print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    torch.save(model_u2_x, "./model_u2/{}_final_u2_x.pkl".format(name))
    torch.save(model_u2_y, "./model_u2/{}_final_u2_y.pkl".format(name))

    # 可选：使用 LBFGS 进一步优化
    print('*' * 20, 'LBFGS', '*' * 20)
    lr = 0.1
    epochs = 5000
    optimizer = torch.optim.LBFGS(
        filter(lambda p: p.requires_grad, itertools.chain(model_u2_x.parameters(), model_u2_y.parameters())),
        lr=lr)
    train_u2(model_u2_x, model_u2_y, optimizer, 'LBFGS', epochs)
    
    torch.save(model_u2_x, "./model_u2/{}_final_u2_x.pkl".format(name))
    torch.save(model_u2_y, "./model_u2/{}_final_u2_y.pkl".format(name))

    # INSERT_YOUR_CODE
    # ========== 画u2关于x1, x2的2维图像 ==========
    import matplotlib.pyplot as plt

    # 设尺度参数
    eps = 10.0

    with torch.no_grad():
        # 均匀网格
        x1_plot = torch.linspace(a, b, 100, dtype=dtype, device=device)
        x2_plot = torch.linspace(a, b, 100, dtype=dtype, device=device)
        X1, X2 = torch.meshgrid(x1_plot, x2_plot, indexing='ij')
        X1_flat = X1.flatten()
        X2_flat = X2.flatten()

        # 小尺度坐标
        y1_flat = X1_flat * eps - (X1_flat * eps).floor()
        y2_flat = X2_flat * eps - (X2_flat * eps).floor()
        xy_grid = torch.stack([X1_flat, X2_flat], dim=0)  # [2, n_points]
        y1y2_grid = torch.stack([y1_flat, y2_flat], dim=0)  # [2, n_points]
        n_points = xy_grid.shape[1]
        dummy_plot = torch.ones((1, n_points), dtype=dtype, device=device)

        # 模型：假定 model_u2_x, model_u2_y 已经加载/训练完毕，并且有 scaling 参数  
        u2_phi_x_plot = model_u2_x(dummy_plot, xy_grid, need_grad=0, normed=False)  # [d, p, n_points]
        u2_phi_y_plot = model_u2_y(dummy_plot, y1y2_grid, need_grad=0, normed=False)  # [d, p, n_points]
        # 归一化因子（与u2_x, u2_y的norm有关，这里假定等价于u1版本，如有不同可适当修改）
        u2_x_norm_phi = model_u2_x(w1, x1, normed=False)
        u2_x_norm = torch.sqrt(torch.sum(w1 * u2_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        u2_y_norm_phi = model_u2_y(w3, x3, normed=False)
        u2_y_norm = torch.sqrt(torch.sum(w3 * u2_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1)
        u2_phi_x_plot = u2_phi_x_plot / u2_x_norm
        u2_phi_y_plot = u2_phi_y_plot / u2_y_norm

        # 归一化的shift
        u2_y_shift = model_u2_y(dummy, x_error_0, normed=False)
        u2_y_shift = u2_y_shift / u2_y_norm

        # 获取alpha
        alpha_u2 = model_u2_x.scaling_par()  # [p]

        # u2 = sum(alpha * prod(phi_x) * prod(phi_y))
        u2_plot = torch.sum(
            alpha_u2.unsqueeze(-1) *
            torch.prod(u2_phi_x_plot, dim=0) *
            (torch.prod(u2_phi_y_plot, dim=0) - torch.prod(u2_y_shift, dim=0)),
            dim=0
        )  # [n_points]

        # 变为2D形状
        u2_plot_2d = u2_plot.reshape(100, 100)
        u2_value_2d = u2_plot_2d.detach().cpu().numpy()
        X1_np = X1.detach().cpu().numpy()
        X2_np = X2.detach().cpu().numpy()

        # 绘制 u2(x1, x2) 的二维热力图
        fig = plt.figure(figsize=(8, 6))
        im = plt.imshow(
            u2_value_2d.T,
            extent=(a, b, a, b),
            origin='lower',
            cmap='viridis',
            aspect='auto'
        )
        plt.xlabel('x1', fontsize=12)
        plt.ylabel('x2', fontsize=12)
        plt.title(r'$u_2(x_1, x_2)$ 2D Heatmap ($\epsilon={}$)'.format(eps), fontsize=14)
        cbar = plt.colorbar(im, shrink=0.8, pad=0.03)
        cbar.set_label('u2', fontsize=12)
        plt.tight_layout()
        plt.savefig('png/u2_x1_x2_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("2D heatmap saved to: png/u2_x1_x2_heatmap.png")
