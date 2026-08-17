from tnn import *
from integration import *
from quadrature import *
import torch.optim as optim
import importlib.util

import itertools
import os
import sys
import time

pi = 3.14159265358979323846
dtype = torch.double
device = 'cuda:0'
name = 'd2_ex2'
dim = 2

p_A = 1
p_chi = 20
p_u0 = 20
p_u1 = 20
p_A0 = p_A + p_A * p_chi

a, b = 0, 1

x1, w1 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
N = len(x1)
x2, w2 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
x3, w3 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
x4, w4 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)

w12 = torch.stack((w1, w2), dim=0)
w34 = torch.stack((w3, w4), dim=0)
w1234 = torch.stack((w1, w2, w3, w4), dim=0)

dummy = torch.ones((1, 1), dtype=dtype, device=device)
x_error_0 = torch.zeros(1, dtype=dtype, device=device)

# 从0step_A.py导入A矩阵信息（使用importlib因为模块名不能以数字开头）
spec = importlib.util.spec_from_file_location("step_0_A", os.path.join(os.path.dirname(__file__), "0step_A.py"))
if spec is None or spec.loader is None:
    raise ImportError("Could not load 0step_A.py for module import.")
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

model_u0 = torch.load("./model_u0/d2_ex2_final_u0.pkl", map_location=device)

# 准备右端项
with torch.no_grad():
    # u0 的值和导数
    u0_phi, u0_grad_phi, u0_grad_grad_phi = model_u0(w1, x1, need_grad=2)
    alpha_u0 = model_u0.scaling_par()
    
    expand_u0_grad_x1 = torch.cat((u0_grad_phi[0:1, :, :], u0_phi[1:2, :, :], torch.ones((2, p_u0, N), dtype=dtype, device=device)), dim=0)
    expand_u0_grad_x2 = torch.cat((u0_phi[0:1, :, :], u0_grad_phi[1:2, :, :], torch.ones((2, p_u0, N), dtype=dtype, device=device)), dim=0)
    u0_grad_grad_x1_2d = torch.cat((u0_grad_grad_phi[0:1, :, :], u0_phi[1:2, :, :]), dim=0)
    u0_grad_grad_x2_2d = torch.cat((u0_phi[0:1, :, :], u0_grad_grad_phi[1:2, :, :]), dim=0)
    
    # 第一项：∇_{y_1} A · ∇_{x_1} u_0
    part1_phi = torch.einsum('dpn, dqn->dpqn', A_grad_y1, expand_u0_grad_x1).flatten(start_dim=1, end_dim=2)
    alpha_part1 = torch.outer(alpha_A_grad_y1, alpha_u0).view(-1)
    
    # 第二项：∇_{y_2} A · ∇_{x_2} u_0
    part2_phi = torch.einsum('dpn, dqn->dpqn', A_grad_y2, expand_u0_grad_x2).flatten(start_dim=1, end_dim=2)
    alpha_part2 = torch.outer(alpha_A_grad_y2, alpha_u0).view(-1)
    
    # 组合 RHS
    rhs_phi = torch.cat((part1_phi, part2_phi), dim=1)
    alpha_rhs = torch.cat((alpha_part1, alpha_part2))
    
    norm_rhs = Int2TNN(w1234, alpha_rhs, rhs_phi, alpha_rhs, rhs_phi)
    print('norm_rhs =', norm_rhs.item())


def criterion_u1(model_u1_x, model_u1_y):
    """
    损失函数：强形式
    ||-∇_{y_1} · (A∇_{y_1} u_1) - ∇_{y_2} · (A∇_{y_2} u_1) - rhs||_{L2}^2
    
    
    其中 LHS = -A∇_{y_1y_1}u_1 - ∇_{y_1}A · ∇_{y_1}u_1 - A∇_{y_2y_2}u_1 - ∇_{y_2}A · ∇_{y_2}u_1
    RHS = ∇_{y_1} A · ∇_{x_1} u_0 + ∇_{y_2} A · ∇_{x_2} u_0
    """
    # 获取 u1 的值和导数
    u1_phi_x, u1_phi_grad_x = model_u1_x(w1, x1, need_grad=1)
    u1_phi_y, u1_grad_phi_y, u1_grad_grad_phi_y = model_u1_y(w3, x3, need_grad=2)
    u1_grad_grad_y1 = torch.cat((u1_phi_x, u1_grad_grad_phi_y[0:1], u1_phi_y[1:2]), dim=0)
    u1_grad_y1 = torch.cat((u1_phi_x, u1_grad_phi_y[0:1], u1_phi_y[1:2]), dim=0)
    u1_grad_grad_y2 = torch.cat((u1_phi_x, u1_phi_y[0:1], u1_grad_grad_phi_y[1:2]), dim=0)
    u1_grad_y2 = torch.cat((u1_phi_x, u1_phi_y[0:1], u1_grad_phi_y[1:2]), dim=0)

    # 1. -A∇_{y_1y_1}u_1
    operand_A_u1_grad_grad_y1 = torch.einsum('dpn, dqn->dpqn', A, u1_grad_grad_y1)
    # 2. -∇_{y_1}A · ∇_{y_1}u_1
    operand_A_grad_y1_u1_grad_y1 = torch.einsum('dpn, dqn->dpqn', A_grad_y1, u1_grad_y1)
    # 3. -A∇_{y_2y_2}u_1
    operand_A_u1_grad_grad_y2 = torch.einsum('dpn, dqn->dpqn', A, u1_grad_grad_y2)
    # 4. -∇_{y_2}A · ∇_{y_2}u_1
    operand_A_grad_y2_u1_grad_y2 = torch.einsum('dpn, dqn->dpqn', A_grad_y2, u1_grad_y2)
    # my_alpha = model_u1_x.scaling_par()
    # 组合 LHS
    my_operand_phi = torch.cat((operand_A_u1_grad_grad_y1, operand_A_grad_y1_u1_grad_y1,operand_A_u1_grad_grad_y2, operand_A_grad_y2_u1_grad_y2), dim=1)
    my_operand_alpha = torch.cat((alpha_A, alpha_A_grad_y1, alpha_A, alpha_A_grad_y2))
    # 计算矩阵：my_matrix = ∫ (my_operand_phi, my_operand_phi)
    my_matrix = torch.sum(
        torch.outer(my_operand_alpha, my_operand_alpha).unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.prod(
            torch.einsum('dn, dpqn, djkn->dpjqk', w1234, my_operand_phi, my_operand_phi), dim=0), dim=(0, 1))
    
    # 计算向量：my_vector = ∫ (my_operand_phi, -rhs)
    my_vector = torch.sum(torch.outer(my_operand_alpha, -alpha_rhs).unsqueeze(dim=-1) * torch.prod(
        torch.einsum('dn, dpqn, dkn->dpkq', w1234, my_operand_phi, rhs_phi), dim=0), dim=(0, 1))
    
    my_alpha = torch.linalg.solve(my_matrix, my_vector)
    
    loss = torch.sqrt(torch.sum(-my_alpha * my_vector) + norm_rhs)
    
    # 更新参数（在计算完损失后，使用 detach 避免计算图问题）
    model_u1_x.scaling_par().data.copy_(my_alpha.detach())

    return loss


def train_u1(model_u1_x, model_u1_y, optimizer, optimizer_flag, epochs):
    if optimizer_flag == 'Adam':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
        for e in range(epochs):
            loss = criterion_u1(model_u1_x, model_u1_y)
            if e == 0:
                print('\n Step 4 {} Initial:'.format(optimizer_flag))
                print('loss: ', loss.item())
                print('lr: ', scheduler.get_last_lr()[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_u1(model_u1_x, model_u1_y)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    print('lr: ', scheduler.get_last_lr()[0])
                    torch.save(model_u1_x, "./model_u1/{}_final_u1_x.pkl".format(name))
                    torch.save(model_u1_y, "./model_u1/{}_final_u1_y.pkl".format(name))

    if optimizer_flag == 'LBFGS':
        def closure():
            loss = criterion_u1(model_u1_x, model_u1_y)
            optimizer.zero_grad()
            loss.backward()
            return loss

        for e in range(epochs):
            if e == 0:
                with torch.no_grad():
                    loss = criterion_u1(model_u1_x, model_u1_y)
                    print('\n Step u1 {} Initial:'.format(optimizer_flag))
                    print('loss: ', loss.item())

            loss = optimizer.step(closure)

            if (e + 1) % 1000 == 0:
                with torch.no_grad():
                    loss = criterion_u1(model_u1_x, model_u1_y)
                    print('\n [ epoch %d ]' % (e + 1))
                    print('loss: ', loss.item())
                    # torch.save(model_u1_x, "./model_u1/{}_final_u1_x.pkl".format(name))
                    # torch.save(model_u1_y, "./model_u1/{}_final_u1_y.pkl".format(name))


if __name__ == '__main__':
    activation = TNN_Sin

    if not os.path.exists("model_u1"):
        os.mkdir("model_u1")

    # model_u1_x = torch.load("./model_u1/d2_ex2_final_u1_x.pkl", map_location=device)
    # model_u1_y = torch.load("./model_u1/d2_ex2_final_u1_y.pkl", map_location=device)
    # loss = criterion_u1(model_u1_x, model_u1_y)
    # print('loss =', loss.item())

    sizes_periodic = [1, 5, 80, 80, 80, p_u1] #test3 0.0002453737469624951
    print(sizes_periodic)

    # 初始化模型
    model_u1_x = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=True).to(dtype).to(device)
    model_u1_y = TNN_periodic(dim, sizes_periodic, activation, period=1, scaling=False).to(dtype).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, itertools.chain(model_u1_x.parameters(), model_u1_y.parameters())),
        lr=1e-2)

    # 第一次训练 Adam
    epochs = 50000
    begin_time = time.time()
    train_u1(model_u1_x, model_u1_y, optimizer, 'Adam', epochs)
    end_time = time.time()
    print(f'\nAdam 训练完成，耗时：{end_time - begin_time:.2f}s')

    # 保存模型（带参数标识，不覆盖）
    torch.save(model_u1_x, f"./model_u1/{name}_final_u1_x.pkl")
    torch.save(model_u1_y, f"./model_u1/{name}_final_u1_y.pkl")

    # ---------LBFGS------------
    print('*' * 20, 'LBFGS', '*' * 20)
    lr = 0.1
    epochs = 5000
    optimizer = torch.optim.LBFGS(
        filter(lambda p: p.requires_grad, itertools.chain(model_u1_x.parameters(), model_u1_y.parameters())),
        lr=lr)
    train_u1(model_u1_x, model_u1_y, optimizer, 'LBFGS', epochs)
    
    # 再次保存模型
    torch.save(model_u1_x, f"./model_u1/{name}_final_u1_x.pkl")
    torch.save(model_u1_y, f"./model_u1/{name}_final_u1_y.pkl")
