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
device = 'cuda:6'
name = 'd2_ex3'

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
A_grad_x1 = A_dict['A_grad_x1']
alpha_A_grad_x1 = A_dict['alpha_A_grad_x1']
A_grad_x2 = A_dict['A_grad_x2']
alpha_A_grad_x2 = A_dict['alpha_A_grad_x2']
A_grad_y1 = A_dict['A_grad_y1']
alpha_A_grad_y1 = A_dict['alpha_A_grad_y1']
A_grad_y2 = A_dict['A_grad_y2']
alpha_A_grad_y2 = A_dict['alpha_A_grad_y2']


norm_square_A_grad_y1 = Int2TNN(w1234, alpha_A_grad_y1, A_grad_y1, alpha_A_grad_y1, A_grad_y1)
print('norm_square_A_grad_y1 =', norm_square_A_grad_y1.item())
norm_square_A_grad_y2 = Int2TNN(w1234, alpha_A_grad_y2, A_grad_y2, alpha_A_grad_y2, A_grad_y2)
print('norm_square_A_grad_y2 =', norm_square_A_grad_y2.item())

# alpha_A0_exact = (1.5 * math.sqrt(2)) * torch.ones(1, dtype=dtype, device=device)

# \int\int \left[  \frac{\grad A}{\grad y_i}
#         + \nabla_y A \cdot \nabla_y \chi_i
#         + A \nabla_y^2 \chi_i
def criterion_12(model_chi2_x, model_chi2_y):
    chi2_phi_x = model_chi2_x(w1, x1, need_grad=0)
    chi2_phi_y, chi2_grad_phi_y, chi2_grad_grad_phi_y = model_chi2_y(w3, x3, need_grad=2)

    chi2_grad_y1 = torch.cat((chi2_phi_x, chi2_grad_phi_y[0:1], chi2_phi_y[1:2]), dim=0)
    chi2_grad_y2 = torch.cat((chi2_phi_x, chi2_phi_y[0:1], chi2_grad_phi_y[1:2]), dim=0)
    chi2_grad2_y1 = torch.cat((chi2_phi_x, chi2_grad_grad_phi_y[0:1], chi2_phi_y[1:2]), dim=0)
    chi2_grad2_y2 = torch.cat((chi2_phi_x, chi2_phi_y[0:1], chi2_grad_grad_phi_y[1:2]), dim=0)

    operand_A_grad_y1_chi2_grad_y1 = torch.einsum('dpn, dqn->dpqn', A_grad_y1, chi2_grad_y1)
    operand_A_grad_y2_chi2_grad_y2 = torch.einsum('dpn, dqn->dpqn', A_grad_y2, chi2_grad_y2)
    operand_A_chi2_grad2_y1 = torch.einsum('dpn, dqn->dpqn', A, chi2_grad2_y1)
    operand_A_chi2_grad2_y2 = torch.einsum('dpn, dqn->dpqn', A, chi2_grad2_y2)

    my_operand_phi = torch.cat((operand_A_grad_y1_chi2_grad_y1, operand_A_grad_y2_chi2_grad_y2, operand_A_chi2_grad2_y1, operand_A_chi2_grad2_y2), dim=1)
    my_operand_alpha = torch.cat((alpha_A_grad_y1, alpha_A_grad_y2, alpha_A, alpha_A))
    # p * p matrix
    my_matrix = torch.sum(torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(torch.einsum('dn, dpqn, djkn->dpjqk', w1234, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    # p vector
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_A_grad_y2).unsqueeze(dim=-1) * torch.prod(torch.einsum('dn, dpqn, dkn->dpkq', w1234, my_operand_phi, A_grad_y2), dim=0), dim=(0, 1))
    
    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_chi2_x.scaling_par().data.copy_(my_alpha)

    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_square_A_grad_y2)
    # loss = torch.sqrt(torch.sum(torch.outer(my_alpha, my_alpha) * my_matrix) - 2 * torch.sum(my_alpha * my_vector) + norm_square_A1_grad_y)

    return loss

def train_12(model_chi2_x, model_chi2_y, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_12(model_chi2_x, model_chi2_y)
            if e == 0:
                print('\n For j=2, Step 1 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                print('lr: ', scheduler.get_last_lr()[0])
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_12(model_chi2_x, model_chi2_y)
                    print('\n [ epoch %d ]' % (e+1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])

                
    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_12(model_chi2_x, model_chi2_y)
            optimizer.zero_grad()
            loss.backward()
            return loss
        
        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_12(model_chi2_x, model_chi2_y)
                    print('\n Step 1 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_12(model_chi2_x, model_chi2_y)
                    print('\n [ epoch %d ]' % (e+1))
                    print('loss: ', loss.item())



if __name__ == '__main__':
    p_net = 20
    sizes_periodic = [1, 2, p_net, p_net, p_net, p] 
    activation = TNN_Sin

    if not os.path.exists("model_chi2"):
        os.mkdir("model_chi2")
    

    model_chi2_x = torch.load("./model_chi2/d2_ex3_final_chi2_x.pkl", map_location=device)
    model_chi2_y = torch.load("./model_chi2/d2_ex3_final_chi2_y.pkl", map_location=device)
    loss = criterion_12(model_chi2_x, model_chi2_y)
    print('loss: ', loss.item())

    # model_chi2_x = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=True).to(dtype).to(device)
    # model_chi2_y = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=False).to(dtype).to(device)

    # optimizer = optim.Adam(filter(lambda p:p.requires_grad, itertools.chain(model_chi2_x.parameters(), model_chi2_y.parameters())), lr=1e-2)

    # epochs = 50000

    # begin_time = time.time()
    # train_12(model_chi2_x, model_chi2_y, optimizer, 'Adam', epochs)
    # end_time = time.time()
    # print('\n')
    # print('*' * 40)
    # print('Training done!')
    # print('Elapsed time is {} seconds.'.format(end_time - begin_time))

    # torch.save(model_chi2_x, "./model_chi2/{}_final_chi2_x.pkl".format(name))
    # torch.save(model_chi2_y, "./model_chi2/{}_final_chi2_y.pkl".format(name))

    
    # optimizer = optim.LBFGS(filter(lambda p:p.requires_grad, itertools.chain(model_chi2_x.parameters(), model_chi2_y.parameters())), lr=0.1, \
    #                         tolerance_grad=0, tolerance_change=0)

    # epochs = 2000
    # train_12(model_chi2_x, model_chi2_y, optimizer, 'LBFGS', epochs)
    # torch.save(model_chi2_x, "./model_chi2/{}_final_chi2_x.pkl".format(name))
    # torch.save(model_chi2_y, "./model_chi2/{}_final_chi2_y.pkl".format(name))

    #     # INSERT_YOUR_CODE
    # # ========== 画chi2关于x1, x2的三维图像 ==========
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
    #     chi2_x_norm_phi = model_chi2_x(w1, x1, need_grad=0, normed=False)
    #     chi2_x_norm = torch.sqrt(torch.sum(w1 * chi2_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    #     chi2_y_norm_phi = model_chi2_y(w3, x3, need_grad=0, normed=False)
    #     chi2_y_norm = torch.sqrt(torch.sum(w3 * chi2_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]

    #     # chi2_phi_x (大尺度), chi2_phi_y (小尺度)
    #     chi2_phi_x_plot = model_chi2_x(dummy_plot, xy_grid, need_grad=0, normed=False)  # [d, p, n_points]
    #     chi2_phi_x_plot = chi2_phi_x_plot / chi2_x_norm
    #     chi2_phi_y_plot = model_chi2_y(dummy_plot, y1y2_grid, need_grad=0, normed=False)  # [d, p, n_points]
    #     chi2_phi_y_plot = chi2_phi_y_plot / chi2_y_norm
    #     chi2_y_shift = model_chi2_y(dummy, x_error_0, need_grad=0, normed=False) #[d,p,1]
    #     chi2_y_shift = chi2_y_shift / chi2_y_norm

    #     # 获取alpha
    #     alpha_chi2 = model_chi2_x.scaling_par()  # [p]

    #     # chi2 = sum(alpha * prod(chi2_phi_x) * (prod(chi2_phi_y) - prod(chi2_y_shift)))
    #     chi2_plot = torch.sum(
    #         alpha_chi2.unsqueeze(dim=-1) *
    #         torch.prod(chi2_phi_x_plot, dim=0) *
    #         (torch.prod(chi2_phi_y_plot, dim=0) - torch.prod(chi2_y_shift, dim=0)),
    #         dim=0
    #     )  # [n_points]

    #     # 转成二维形状
    #     chi2_plot_2d = chi2_plot.reshape(100, 100)
    #     chi2_value_2d = chi2_plot_2d.detach().cpu().numpy()
    #     X1_np = X1.detach().cpu().numpy()
    #     X2_np = X2.detach().cpu().numpy()

    #     # 绘图
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     surf = ax.plot_surface(X1_np, X2_np, chi2_value_2d, cmap='viridis',
    #                            edgecolor='none', alpha=0.9, linewidth=0, antialiased=True)
    #     ax.set_xlabel('x1', fontsize=12)
    #     ax.set_ylabel('x2', fontsize=12)
    #     ax.set_zlabel('chi2(x1, x2)', fontsize=12)
    #     ax.set_title(r'$\chi_2(x_1, x_2)$ 3D Plot ($\epsilon={}$)'.format(eps), fontsize=14)
    #     fig.colorbar(surf, shrink=0.5, aspect=20, pad=0.1, label='chi2')
    #     ax.view_init(elev=30, azim=45)
    #     plt.tight_layout()
    #     plt.savefig('png/chi2_x1_x2_surface.png', dpi=300, bbox_inches='tight')
    #     plt.close(fig)
    #     print("3D plot saved to: png/chi2_x1_x2_surface.png")