import torch
torch.set_printoptions(profile="full", precision=16)
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import torch.optim as optim

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
name = 'd1_ex1'

# 求前m个特征值
# m = 1
dim=1

a, b = 0, 1

p_A = 2
p_chi1 = 20
p_u0 = 20
p_u1 = 20
p_u2 = 20
p_A0 = p_A + p_A * p_chi1
x1, w1 = composite_quadrature_1d(4, 0, 1, 200, device=device)
x2, w2 = composite_quadrature_1d(4, 0, 1, 200, device=device)

dummy = torch.ones((1, 1), dtype=dtype, device=device)
x_error_0 = torch.zeros(1, dtype=dtype, device=device)

N = len(x1)
w12 = torch.stack((w1, w2), dim=0)

def bd_u0(x):
    # return torch.sin(pi * x)
    return (x - a) * (b - x)

def grad_bd_u0(x):
    # return pi * torch.cos(pi * x)
    return a + b - 2 * x

def grad_grad_bd_u0(x):
    # return -pi * pi * torch.sin(pi * x)
    return -2 * torch.ones_like(x)

def grad_grad_grad_bd_u0(x):
    # return -pi * pi * pi * torch.cos(pi * x)
    return 0 * torch.ones_like(x)

A = torch.ones((2, 2, N), dtype=dtype, device=device)
A[0, 0, :] = 2 + x1
A[1, 1, :] = torch.sin(2 * pi * x2)
alpha_A = torch.ones(2, dtype=dtype, device=device)

A_partial_x = torch.ones((2, 2, N), dtype=dtype, device=device)
A_partial_x[0, 0, :] = 1
A_partial_x[1, 1, :] = 0
alpha_A_partial_x = torch.ones(2, dtype=dtype, device=device)

A_partial_y = torch.ones((2, 2, N), dtype=dtype, device=device)
A_partial_y[0, 0, :] = 0
A_partial_y[1, 1, :] = 2 * pi * torch.cos(2 * pi * x2)
alpha_A_partial_y = torch.ones(2, dtype=dtype, device=device)

F = x1.view(1, 1, -1)
alpha_F = 3 * torch.ones(1, dtype=dtype, device=device)
expand_F = torch.cat((F, torch.ones((1, 1, N), dtype=dtype, device=device)), dim=0)

model_u0_x = torch.load('./model_u0/d1_ex1_final_u0.pkl', map_location=device)
model_u1_x = torch.load('./model_u1/d1_ex1_final_u1_x.pkl', map_location=device)
model_u1_y = torch.load('./model_u1/d1_ex1_final_u1_y.pkl', map_location=device)

with torch.no_grad():
    u0_phi, u0_grad_phi, u0_grad_grad_phi = model_u0_x(w1, x1, need_grad=2)
    alpha_u0 = model_u0_x.scaling_par()

    expand_u0_grad_x = torch.cat((u0_grad_phi, torch.ones((1, p_u0, N), dtype=dtype, device=device)), dim=0)
    expand_u0_grad2_x = torch.cat((u0_grad_grad_phi, torch.ones((1, p_u0, N), dtype=dtype, device=device)), dim=0)

    u1_x_phi, u1_x_grad_phi = model_u1_x(w1, x1, need_grad=1)
    u1_y_phi, u1_y_grad_phi = model_u1_y(w2, x2, need_grad=1)
    alpha_u1 = model_u1_x.scaling_par()
    
    u1_y_norm_phi = model_u1_y(w2, x2, normed=False)
    u1_y_norm = torch.sqrt(torch.sum(w2 * u1_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    u1_y_shift = model_u1_y(dummy, x_error_0, need_grad=0, normed=False) #[d,p,1]
    u1_y_shift = u1_y_shift / u1_y_norm

    u1_grad_x = torch.cat((u1_x_grad_phi, u1_y_phi), dim=0)
    u1_grad_x_shift = torch.cat((u1_x_grad_phi, u1_y_shift.expand(1, p_u1, N)), dim=0)
    u1_grad_y = torch.cat((u1_x_phi, u1_y_grad_phi), dim=0)
    u1_grad_xy = torch.cat((u1_x_grad_phi, u1_y_grad_phi), dim=0)

    # \nabla_y A \nabla_x u1
    part1 = torch.einsum('dpn,dqn->dpqn', A_partial_y, u1_grad_x).flatten(start_dim=1, end_dim=2) 
    part1_shift = torch.einsum('dpn,dqn->dpqn', A_partial_y, u1_grad_x_shift).flatten(start_dim=1, end_dim=2) 
    alpha_part1 = torch.outer(alpha_A, alpha_u1).view(-1)

    # \nabla_x A \nabla_y u1
    part2 = torch.einsum('dpn,dqn->dpqn', A_partial_x, u1_grad_y).flatten(start_dim=1, end_dim=2)
    alpha_part2 = alpha_part1

    # 2 A \nabla_x \nabla_y u1
    part3 = torch.einsum('dpn,dqn->dpqn', A, u1_grad_xy).flatten(start_dim=1, end_dim=2) 
    alpha_part3 = 2 * alpha_part1

    # \nabla_x A \nabla_x u0
    part4 = torch.einsum('dpn,dqn->dpqn', A_partial_x, expand_u0_grad_x).flatten(start_dim=1, end_dim=2)
    alpha_part4 = torch.outer(alpha_A, alpha_u0).view(-1)
    
    # A \nabla_x \nabla_x u0
    part5 = torch.einsum('dpn,dqn->dpqn', A, expand_u0_grad2_x).flatten(start_dim=1, end_dim=2)
    alpha_part5 = alpha_part4
    
    rhs_term = torch.cat((expand_F, part1, part1_shift, part2, part3, part4, part5), dim=1)
    alpha_rhs_term = torch.cat((alpha_F, alpha_part1, -alpha_part1, alpha_part2, alpha_part3, alpha_part4, alpha_part5))

    norm_square_rhs_term = Int2TNN(w12, alpha_rhs_term, rhs_term, alpha_rhs_term, rhs_term)

def criterion_1(model_u2_x, model_u2_y):
    u2_phi_x = model_u2_x(w1, x1, need_grad=0)
    _, u2_grad_phi_y, u2_grad_grad_phi_y = model_u2_y(w2, x2, need_grad=2)

    u2_partial_y = torch.cat((u2_phi_x, u2_grad_phi_y), dim=0)
    u2_partial2_y = torch.cat((u2_phi_x, u2_grad_grad_phi_y), dim=0)

    operand_A_partial_y_u2_partial_y = torch.einsum('dpn, dqn->dpqn', A_partial_y, u2_partial_y)
    operand_A_u2_partial2_y = torch.einsum('dpn, dqn->dpqn', A, u2_partial2_y)

    my_operand_phi = torch.cat((operand_A_partial_y_u2_partial_y, operand_A_u2_partial2_y), dim=1)
    my_operand_alpha = torch.cat((alpha_A_partial_y, alpha_A))

    # p * p matrix
    my_matrix = torch.sum(
        torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(
            torch.einsum('dn, dpqn, djkn->dpjqk', w12, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    # p vector
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_rhs_term).unsqueeze(dim=-1) * torch.prod(
        torch.einsum('dn, dpqn, dkn->dpkq', w12, my_operand_phi, rhs_term), dim=0), dim=(0, 1))

    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    model_u2_x.scaling_par().data.copy_(my_alpha)

    # my_alpha = model_u2_x.scaling_par()

    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_square_rhs_term)
    # loss = torch.sqrt(torch.sum(torch.outer(my_alpha, my_alpha) * my_matrix) - 2 * torch.sum(my_alpha * my_vector) + norm_square_A1_partial_y)

    return loss


def train_1(model_u2_x, model_u2_y, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_1(model_u2_x, model_u2_y)
            if e == 0:
                print('\n Step 5 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                print('lr: ', scheduler.get_last_lr()[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_1(model_u2_x, model_u2_y)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])

        

    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_1(model_u2_x, model_u2_y)
            optimizer.zero_grad()
            loss.backward()
            return loss

        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_1(model_u2_x, model_u2_y)
                    print('\n Step 1 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_1(model_u2_x, model_u2_y)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
    return loss

            


if __name__ == '__main__':
    # ===================== 自动参数测试配置 =====================
    # 你要测试的所有参数组合：(p_net, p_net_periodic)
    param_combinations = [
        (20, 2),
        (50, 2),
        (50, 5),
        (80, 5),
        (100, 8),
    ]
    # 存储结果：[ (p_net, p_per, final_loss), ... ]
    results = []
    # ===========================================================

    activation = TNN_Sin

    if not os.path.exists("model_u2"):
        os.mkdir("model_u2")
 
    # ===================== 自动遍历所有参数组合 =====================
    for idx, (p_net, p_net_periodic) in enumerate(param_combinations):
        print("\n" + "="*60)
        print(f" 开始测试第 {idx+1}/{len(param_combinations)} 组参数 ")
        print(f" p_net = {p_net}, p_net_periodic = {p_net_periodic} ")
        print("="*60)
        
        # 网络结构
        sizes = [1, p_net, p_net, p_net, p_net, p_u2]
        sizes_periodic = [1, p_net_periodic, p_net, p_net, p_net, p_u2]

        # 初始化模型
        model_u2_x = TNN(1, sizes, activation, scaling=True).to(dtype).to(device)
        model_u2_y = TNN_periodic(1, sizes_periodic, activation, period=1, scaling=False).to(dtype).to(device)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, itertools.chain(model_u2_x.parameters(), model_u2_y.parameters())),
            lr=1e-2)

        # 第一次训练 Adam
        epochs = 40000
        begin_time = time.time()
        final_loss = train_1(model_u2_x, model_u2_y, optimizer, 'Adam', epochs)
        end_time = time.time()
        print(f'\nAdam 训练完成，耗时：{end_time - begin_time:.2f}s')

        # 第二次训练 LBFGS
        epochs = 4000
        begin_time = time.time()
        final_loss = train_1(model_u2_x, model_u2_y, optimizer, 'LBFGS', epochs)
        end_time = time.time()
        print(f'LBFGS 精调完成，耗时：{end_time - begin_time:.2f}s')

        # 保存模型（带参数标识，不覆盖）
        save_name = f"{name}_p{p_net}_pper{p_net_periodic}"
        torch.save(model_u2_x, f"./model_u2/{save_name}_final_u2_x.pkl")
        torch.save(model_u2_y, f"./model_u2/{save_name}_final_u2_y.pkl")
        
        # 记录结果
        results.append( (p_net, p_net_periodic, final_loss) )
        print(f"\n✅ 参数组 (p_net={p_net}, p_per={p_net_periodic}) 最终 loss = {final_loss:.8e}")

    # ===================== 自动输出最优结果 =====================
    print("\n" + "*"*80)
    print(" 🎯 所有参数组合测试完成！结果汇总：")
    print("-"*80)
    for p, p_per, loss in results:
        print(f"p_net={p:3d}, p_per={p_per:2d}  |  loss = {loss:.8e}")
    
    # 找到 loss 最小的一组
    best = min(results, key=lambda x: x[2])
    print("-"*80)
    print(f"🏆 最优参数组合：p_net = {best[0]}, p_net_periodic = {best[1]}")
    print(f"🎯 最小 loss 值：{best[2]:.8e}")
    print("*"*80)
