import torch
torch.set_printoptions(profile="full", precision=16)
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import torch.optim as optim
import importlib.util

from tnn import *
from integration import *
from quadrature import *

import itertools
import os
import sys
import time
import math

pi = 3.14159265358979323846
dtype = torch.double
device = 'cuda:7'
name = 'd2_ex4_test9'

# sizes_periodic = [1, 2, 50, 50, 20] # test1 0.005107542292444961
# sizes_periodic = [1, 2, 50, 50, 50, 20] # test2 0.004279055165279568
# sizes_periodic = [1, 2, 50, 50, 50, 50, 20] # test3 0.004174401816984543
sizes_periodic = [1, 5, 80, 80, 20] # test4 0.003514706081699982（✅）
# sizes_periodic = [1, 5, 80, 80, 80, 20] # test5 0.01214180136842775
# sizes_periodic = [1, 5, 80, 80, 80, 80, 20] # test6 0.009954727098715109
# sizes_periodic = [1, 8, 100, 100, 20] # test7 0.004099068772830159
# sizes_periodic = [1, 8, 100, 100, 100, 20] # test8 0.009961200863225524
# sizes_periodic = [1, 8, 100, 100, 100, 100, 20] # test9 0.36523216722002266

p = 20

dim = 2
a, b = 0, 1

x1, w1 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
N = len(x1)
x2, w2 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
x3, w3 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
x4, w4 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)

w1234 = torch.stack((w1, w2, w3, w4), dim=0)
w12 = torch.stack((w1, w2), dim=0)

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
A_grad_y1 = A_dict['A_grad_y1']
alpha_A_grad_y1 = A_dict['alpha_A_grad_y1']
A_grad_y2 = A_dict['A_grad_y2']
alpha_A_grad_y2 = A_dict['alpha_A_grad_y2']


norm_square_A_grad_y1 = Int2TNN(w1234, alpha_A_grad_y1, A_grad_y1, alpha_A_grad_y1, A_grad_y1)
print('norm_square_A_grad_y1 =', norm_square_A_grad_y1.item())
norm_square_A_grad_y2 = Int2TNN(w1234, alpha_A_grad_y2, A_grad_y2, alpha_A_grad_y2, A_grad_y2)
print('norm_square_A_grad_y2 =', norm_square_A_grad_y2.item())

# \int\int \left[  \frac{\grad A}{\grad y_i}
#         + \nabla_y A \cdot \nabla_y \chi_i
#         + A \nabla_y^2 \chi_i
def criterion_chi1(model_chi1_x, model_chi1_y):
    chi1_phi_x = model_chi1_x(w1, x1, need_grad=0)
    chi1_phi_y, chi1_grad_phi_y, chi1_grad_grad_phi_y = model_chi1_y(w3, x3, need_grad=2)

    chi1_grad_y1 = torch.cat((chi1_phi_x, chi1_grad_phi_y[0:1], chi1_phi_y[1:2]), dim=0)
    chi1_grad_y2 = torch.cat((chi1_phi_x, chi1_phi_y[0:1], chi1_grad_phi_y[1:2]), dim=0)
    chi1_grad2_y1 = torch.cat((chi1_phi_x, chi1_grad_grad_phi_y[0:1], chi1_phi_y[1:2]), dim=0)
    chi1_grad2_y2 = torch.cat((chi1_phi_x, chi1_phi_y[0:1], chi1_grad_grad_phi_y[1:2]), dim=0)

    operand_A_grad_y1_chi1_grad_y1 = torch.einsum('dpn, dqn->dpqn', A_grad_y1, chi1_grad_y1)
    operand_A_grad_y2_chi1_grad_y2 = torch.einsum('dpn, dqn->dpqn', A_grad_y2, chi1_grad_y2)
    operand_A_chi1_grad2_y1 = torch.einsum('dpn, dqn->dpqn', A, chi1_grad2_y1)
    operand_A_chi1_grad2_y2 = torch.einsum('dpn, dqn->dpqn', A, chi1_grad2_y2)

    my_operand_phi = torch.cat((operand_A_grad_y1_chi1_grad_y1, operand_A_grad_y2_chi1_grad_y2, operand_A_chi1_grad2_y1, operand_A_chi1_grad2_y2), dim=1)
    my_operand_alpha = torch.cat((alpha_A_grad_y1, alpha_A_grad_y2, alpha_A, alpha_A))
    # p * p matrix
    my_matrix = torch.sum(torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(torch.einsum('dn, dpqn, djkn->dpjqk', w1234, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    # p vector
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_A_grad_y1).unsqueeze(dim=-1) * torch.prod(torch.einsum('dn, dpqn, dkn->dpkq', w1234, my_operand_phi, A_grad_y1), dim=0), dim=(0, 1))

    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_chi1_x.scaling_par().data.copy_(my_alpha)

    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_square_A_grad_y1)

    return loss

def train_chi1(model_chi1_x, model_chi1_y, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_chi1(model_chi1_x, model_chi1_y)
            if e == 0:
                print('\n For j=1, Step 1 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                print('lr: ', scheduler.get_last_lr()[0])
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_chi1(model_chi1_x, model_chi1_y)
                    print('\n [ epoch %d ]' % (e+1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])
                
    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_chi1(model_chi1_x, model_chi1_y)
            optimizer.zero_grad()
            loss.backward()
            return loss
        
        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_chi1(model_chi1_x, model_chi1_y)
                    print('\n Step 1 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_chi1(model_chi1_x, model_chi1_y)
                    print('\n [ epoch %d ]' % (e+1))
                    print('loss: ', loss.item())
        
        # torch.save(model_chi1_x, "model_chi1/{}_final_chi1_x.pkl".format(name))
        # torch.save(model_chi1_y, "model_chi1/{}_final_chi1_y.pkl".format(name))


if __name__ == '__main__':

    

    print(sizes_periodic)
    activation = TNN_Sin

    if not os.path.exists("model_chi1"):
        os.mkdir("model_chi1")
    
    model_chi1_x = torch.load("./model_chi1/d2_ex4_final_chi1_x.pkl", map_location=device)
    model_chi1_y = torch.load("./model_chi1/d2_ex4_final_chi1_y.pkl", map_location=device)
    loss = criterion_chi1(model_chi1_x, model_chi1_y)
    print('loss: ', loss.item())
    
    # model_chi1_x = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=True).to(dtype).to(device)
    # model_chi1_y = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=False).to(dtype).to(device)

    # optimizer = optim.Adam(filter(lambda p:p.requires_grad, itertools.chain(model_chi1_x.parameters(), model_chi1_y.parameters())), lr=1e-2)

    # epochs = 50000

    # begin_time = time.time()
    # train_chi1(model_chi1_x, model_chi1_y, optimizer, 'Adam', epochs)
    # end_time = time.time()
    # print('\n')
    # print('*' * 40)
    # print('Training done!')
    # print('Elapsed time is {} seconds.'.format(end_time - begin_time))
    # torch.save(model_chi1_x, "./model_chi1/{}_final_chi1_x.pkl".format(name))
    # torch.save(model_chi1_y, "./model_chi1/{}_final_chi1_y.pkl".format(name))
    

    # optimizer = optim.LBFGS(filter(lambda p:p.requires_grad, itertools.chain(model_chi1_x.parameters(), model_chi1_y.parameters())), lr=1, \
    #                         tolerance_grad=0, tolerance_change=0)

    # epochs = 4000
    # train_chi1(model_chi1_x, model_chi1_y, optimizer, 'LBFGS', epochs)

    # torch.save(model_chi1_x, "./model_chi1/{}_final_chi1_x.pkl".format(name))
    # torch.save(model_chi1_y, "./model_chi1/{}_final_chi1_y.pkl".format(name))


    #     # INSERT_YOUR_CODE
    # # ========== 画chi1关于x1, x2的三维图像 ==========
    # import matplotlib.pyplot as plt

    # # 设尺度参数
    # eps = 5.0

    # with torch.no_grad():
    #     # 均匀网格
    #     x1_plot = torch.linspace(a, b, 100, dtype=dtype, device=device)
    #     x2_plot = torch.linspace(a, b, 100, dtype=dtype, device=device)
    #     X1, X2 = torch.meshgrid(x1_plot, x2_plot, indexing='ij')
    #     X1_flat = X1.flatten()
    #     X2_flat = X2.flatten()

    #     # 小尺度坐标
    #     y1_flat = X1_flat * eps - (X1_flat * eps).floor()
    #     y2_flat = X2_flat * eps - (X2_flat * eps).floor()
    #     xy_grid = torch.stack([X1_flat, X2_flat], dim=0)  # [2, n_points]
    #     y1y2_grid = torch.stack([y1_flat, y2_flat], dim=0)  # [2, n_points]
    #     n_points = xy_grid.shape[1]
    #     dummy_plot = torch.ones((1, n_points), dtype=dtype, device=device)

    #     #归一化
    #     chi1_x_norm_phi = model_chi1_x(w1, x1, need_grad=0, normed=False)
    #     chi1_x_norm = torch.sqrt(torch.sum(w1 * chi1_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    #     chi1_y_norm_phi = model_chi1_y(w3, x3, need_grad=0, normed=False)
    #     chi1_y_norm = torch.sqrt(torch.sum(w3 * chi1_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    #     chi1_y_shift = model_chi1_y(dummy, x_error_0, need_grad=0, normed=False) #[d,p,1]
    #     chi1_y_shift = chi1_y_shift / chi1_y_norm

    #     # chi1_phi_x (大尺度), chi1_phi_y (小尺度)
    #     chi1_phi_x_plot = model_chi1_x(dummy_plot, xy_grid, need_grad=0, normed=False)  # [d, p, n_points]
    #     chi1_phi_x_plot = chi1_phi_x_plot / chi1_x_norm
    #     chi1_phi_y_plot = model_chi1_y(dummy_plot, y1y2_grid, need_grad=0, normed=False)  # [d, p, n_points]
    #     chi1_phi_y_plot = chi1_phi_y_plot / chi1_y_norm

    #     # 获取alpha
    #     alpha_chi1 = model_chi1_x.scaling_par()  # [p]

    #     # chi1 = sum(alpha * prod(chi1_phi_x) * prod(chi1_phi_y - chi1_y_shift))
    #     chi1_plot = torch.sum(
    #         alpha_chi1.unsqueeze(dim=-1) *
    #         torch.prod(chi1_phi_x_plot, dim=0) *
    #         (torch.prod(chi1_phi_y_plot, dim=0) - torch.prod(chi1_y_shift, dim=0)),
    #         dim=0
    #     )  # [n_points]

    #     # 转成二维形状
    #     chi1_plot_2d = chi1_plot.reshape(100, 100)
    #     chi1_value_2d = chi1_plot_2d.detach().cpu().numpy()
    #     X1_np = X1.detach().cpu().numpy()
    #     X2_np = X2.detach().cpu().numpy()

    #     # 绘图
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     surf = ax.plot_surface(X1_np, X2_np, chi1_value_2d, cmap='viridis',
    #                            edgecolor='none', alpha=0.9, linewidth=0, antialiased=True)
    #     ax.set_xlabel('x1', fontsize=12)
    #     ax.set_ylabel('x2', fontsize=12)
    #     ax.set_zlabel('chi1(x1, x2)', fontsize=12)
    #     ax.set_title(r'$\chi_1(x_1, x_2)$ 3D Plot ($\epsilon={}$)'.format(eps), fontsize=14)
    #     fig.colorbar(surf, shrink=0.5, aspect=20, pad=0.1, label='chi1')
    #     ax.view_init(elev=30, azim=45)
    #     plt.tight_layout()
    #     plt.savefig('png/chi1_x1_x2_surface_minus_y_shift.png', dpi=300, bbox_inches='tight')
    #     plt.close(fig)
    #     print("3D plot saved to: png/chi1_x1_x2_surface_minus_y_shift.png")


    
        # # INSERT_YOUR_CODE

        # # ========== 画固定 x1 时 chi1 关于 x2 的曲线（同时固定 y1），x1取[0.2, 0.5, 1]，所有画在同一张图上 ==========

        # x1_fixed_values = [0.12, 0.4,0.67]
        # eps = 5.0  # 保持与前面一致
        # n_x2_points = 200

        # with torch.no_grad():
        #     fig_line, ax_line = plt.subplots(figsize=(8, 5))
        #     for x1_fixed_value in x1_fixed_values:
        #         # 生成 x2 网格
        #         x2_plot = torch.linspace(a, b, n_x2_points, dtype=dtype, device=device)
        #         # 固定 x1，拼成坐标对
        #         x1_fixed_tensor = torch.full_like(x2_plot, fill_value=x1_fixed_value)
        #         xy_grid_line = torch.stack([x1_fixed_tensor, x2_plot], dim=0)  # [2, n_points]

        #         # 计算对应的小尺度坐标（y1固定，y2随x2变化）
        #         y1_fixed = x1_fixed_tensor * eps - (x1_fixed_tensor * eps).floor()
        #         y2 = x2_plot * eps - (x2_plot * eps).floor()
        #         y1y2_grid_line = torch.stack([y1_fixed, y2], dim=0)  # [2, n_points]

        #         n_points_line = xy_grid_line.shape[1]
        #         dummy_plot_line = torch.ones((1, n_points_line), dtype=dtype, device=device)

        #         # 归一化
        #         chi1_phi_x_line = model_chi1_x(dummy_plot_line, xy_grid_line, need_grad=0, normed=False)
        #         chi1_phi_x_line = chi1_phi_x_line / chi1_x_norm
        #         chi1_phi_y_line = model_chi1_y(dummy_plot_line, y1y2_grid_line, need_grad=0, normed=False)
        #         chi1_phi_y_line = chi1_phi_y_line / chi1_y_norm

        #         chi1_plot_line = torch.sum(
        #             alpha_chi1.unsqueeze(dim=-1) *
        #             torch.prod(chi1_phi_x_line, dim=0) *
        #             (torch.prod(chi1_phi_y_line, dim=0) - torch.prod(chi1_y_shift, dim=0)),
        #             dim=0
        #         )  # [n_points_line]

        #         chi1_plot_line_np = chi1_plot_line.detach().cpu().numpy()
        #         x2_plot_np = x2_plot.detach().cpu().numpy()

        #         # 在同一张图上画
        #         ax_line.plot(x2_plot_np, chi1_plot_line_np, lw=2, label=r'$x_1={}$'.format(x1_fixed_value))

        #     # 总图设置
        #     ax_line.set_xlabel('x2', fontsize=12)
        #     ax_line.set_ylabel(r'$\chi_1(x_1, x_2)$', fontsize=12)
        #     ax_line.set_title(r'$\chi_1(x_1, x_2)$ with different $x_1$ , $\epsilon={}$'.format(eps), fontsize=14)
        #     ax_line.legend()
        #     ax_line.grid(True)
        #     plt.tight_layout()
        #     plt.savefig('png/chi1_x2_multi_x1_bd_minus_y_shift.png', dpi=300, bbox_inches='tight')
        #     plt.close(fig_line)
        #     print("chi1(x1, x2) for x1 in {} saved to: png/chi1_x2_multi_x1_bd.png".format(x1_fixed_values))

            # # INSERT_YOUR_CODE

            # # ========== 画固定 x2 时 chi1 关于 x1 的曲线（同时固定 y2），x2取[0.12, 0.4, 0.67], 所有画在同一张图上 ==========
            # x2_fixed_values = [0.12, 0.4, 0.67]
            # eps = 5.0  # 保持与前面一致
            # n_x1_points = 200

            # with torch.no_grad():
            #     fig_line2, ax_line2 = plt.subplots(figsize=(8, 5))
            #     for x2_fixed_value in x2_fixed_values:
            #         # 生成 x1 网格
            #         x1_plot = torch.linspace(a, b, n_x1_points, dtype=dtype, device=device)
            #         # 固定 x2，拼成坐标对
            #         x2_fixed_tensor = torch.full_like(x1_plot, fill_value=x2_fixed_value)
            #         xy_grid_line2 = torch.stack([x1_plot, x2_fixed_tensor], dim=0)  # [2, n_points]

            #         # 计算对应的小尺度坐标（y2固定，y1随x1变化）
            #         y2_fixed = x2_fixed_tensor * eps - (x2_fixed_tensor * eps).floor()
            #         y1 = x1_plot * eps - (x1_plot * eps).floor()
            #         y1y2_grid_line2 = torch.stack([y1, y2_fixed], dim=0)  # [2, n_points]

            #         n_points_line2 = xy_grid_line2.shape[1]
            #         dummy_plot_line2 = torch.ones((1, n_points_line2), dtype=dtype, device=device)

            #         # 归一化
            #         chi1_phi_x_line2 = model_chi1_x(dummy_plot_line2, xy_grid_line2, need_grad=0, normed=False)
            #         chi1_phi_x_line2 = chi1_phi_x_line2 / chi1_x_norm
            #         chi1_phi_y_line2 = model_chi1_y(dummy_plot_line2, y1y2_grid_line2, need_grad=0, normed=False)
            #         chi1_phi_y_line2 = chi1_phi_y_line2 / chi1_y_norm

            #         chi1_plot_line2 = torch.sum(
            #             alpha_chi1.unsqueeze(dim=-1) *
            #             torch.prod(chi1_phi_x_line2, dim=0) *
            #             (torch.prod(chi1_phi_y_line2, dim=0) - torch.prod(chi1_y_shift, dim=0)),
            #             dim=0
            #         )  # [n_points_line2]

            #         chi1_plot_line2_np = chi1_plot_line2.detach().cpu().numpy()
            #         x1_plot_np = x1_plot.detach().cpu().numpy()

            #         # 在同一张图上画
            #         ax_line2.plot(x1_plot_np, chi1_plot_line2_np, lw=2, label=r'$x_2={}$'.format(x2_fixed_value))

            #     # 总图设置
            #     ax_line2.set_xlabel('x1', fontsize=12)
            #     ax_line2.set_ylabel(r'$\chi_1(x_1, x_2)$', fontsize=12)
            #     ax_line2.set_title(r'$\chi_1(x_1, x_2)$ with different $x_2$ , $\epsilon={}$'.format(eps), fontsize=14)
            #     ax_line2.legend()
            #     ax_line2.grid(True)
            #     plt.tight_layout()
            #     plt.savefig('png/chi1_x1_multi_x2_bd_shifted.png', dpi=300, bbox_inches='tight')
            #     plt.close(fig_line2)
            #     print("chi1(x1, x2) for x2 in {} saved to: png/chi1_x1_multi_x2_bd_shifted.png".format(x2_fixed_values))
