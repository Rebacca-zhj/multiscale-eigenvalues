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
name = 'd2_ex2_test4'
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
                print('\n Step u1 {} Initial:'.format(optimizer_flag))
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

    model_u1_x = torch.load("./model_u1/d2_ex2_final_u1_x.pkl", map_location=device)
    model_u1_y = torch.load("./model_u1/d2_ex2_final_u1_y.pkl", map_location=device)
    loss = criterion_u1(model_u1_x, model_u1_y)
    print('loss =', loss.item())

    # sizes_periodic = [1, 2, 20, 20, 20, p_u1] #test1 0.005741595518075647
    # sizes_periodic = [1, 2, 50, 50, 50, p_u1] #test2 3.497126452992761
    sizes_periodic = [1, 5, 80, 80, 80, p_u1] #test3 0.0002453737469624951
    # sizes_periodic = [1, 2, 50, 50, 50, 50, p_u1] #test4 3.4971264444943717
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

        
    # # INSERT_YOUR_CODE

    # # --------- 误差计算：当前u1 vs 解析形式 chi1 * grad_{x1} u0 + chi2 * grad_{x2} u0 ---------

    # # INSERT_YOUR_CODE

    # # --------- 误差计算：当前u1 vs 解析形式 chi1 * grad_{x1} u0 + chi2 * grad_{x2} u0 ---------

    # with torch.no_grad():
    #     # 分别在 u1_x 和 u1_y 网格上插值当前的 u1
    #     # 1. 计算 u1 当前值
    #     u1_phi_x, u1_phi_grad_x = model_u1_x(w1, x1, need_grad=1)
    #     u1_phi_y, u1_phi_grad_y = model_u1_y(w3, x3, need_grad=1)
    #     alpha_u1 = model_u1_x.scaling_par()
    #     u1_x_norm_phi = model_u1_x(w1, x1, normed=False)
    #     u1_x_norm = torch.sqrt(torch.sum(w1 * u1_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    #     u1_y_norm_phi = model_u1_y(w3, x3, normed=False)
    #     u1_y_norm = torch.sqrt(torch.sum(w3 * u1_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    #     u1_y_shift = model_u1_y(dummy, x_error_0, normed=False)
    #     u1_y_shift = u1_y_shift / u1_y_norm
    #     u1 = torch.cat((u1_phi_x, u1_phi_y), dim=0)
    #     u1_shift = torch.cat((u1_phi_x, u1_y_shift.expand(2, p_u1, N)), dim=0)
    #     u1_grad_x1 = torch.cat((u1_phi_grad_x[0:1, :, :],u1_phi_x[1:2, :, :], u1_phi_y), dim=0)
    #     u1_grad_x1_shift = torch.cat((u1_phi_grad_x[0:1, :, :],u1_phi_x[1:2, :, :], u1_y_shift.expand(2, p_u1, N)), dim=0)
    #     u1_grad_y1 = torch.cat((u1_phi_x, u1_phi_grad_y[0:1, :, :], u1_phi_y[1:2, :, :]), dim=0)

    #     # 2. 加载/重新计算 chi1, chi2
    #     model_chi1_x = torch.load('./model_chi1/d2_ex2_final_chi1_x.pkl', map_location=device)
    #     model_chi1_y = torch.load('./model_chi1/d2_ex2_final_chi1_y.pkl', map_location=device)
    #     model_chi2_x = torch.load('./model_chi2/d2_ex2_final_chi2_x.pkl', map_location=device)
    #     model_chi2_y = torch.load('./model_chi2/d2_ex2_final_chi2_y.pkl', map_location=device)
    #     chi1_phi_x, chi1_phi_grad_x = model_chi1_x(w1, x1, need_grad=1)
    #     chi1_phi_y, chi1_phi_grad_y = model_chi1_y(w3, x3, need_grad=1)
    #     alpha_chi1 = model_chi1_x.scaling_par()
    #     chi1_x_norm_phi = model_chi1_x(w1, x1, normed=False)
    #     chi1_x_norm = torch.sqrt(torch.sum(w1 * chi1_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    #     chi1_y_norm_phi = model_chi1_y(w3, x3, normed=False)
    #     chi1_y_norm = torch.sqrt(torch.sum(w3 * chi1_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    #     chi1_y_shift = model_chi1_y(dummy, x_error_0, normed=False)
    #     chi1_y_shift = chi1_y_shift / chi1_y_norm

    #     chi2_phi_x, chi2_phi_grad_x = model_chi2_x(w1, x1, need_grad=1)
    #     chi2_phi_y, chi2_phi_grad_y = model_chi2_y(w3, x3, need_grad=1)
    #     alpha_chi2 = model_chi2_x.scaling_par()
    #     chi2_x_norm_phi = model_chi2_x(w1, x1, normed=False)
    #     chi2_x_norm = torch.sqrt(torch.sum(w1 * chi2_x_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    #     chi2_y_norm_phi = model_chi2_y(w3, x3, normed=False)
    #     chi2_y_norm = torch.sqrt(torch.sum(w3 * chi2_y_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]
    #     chi2_y_shift = model_chi2_y(dummy, x_error_0, normed=False)
    #     chi2_y_shift = chi2_y_shift / chi2_y_norm

    #     chi1 = torch.cat((chi1_phi_x, chi1_phi_y), dim=0)
    #     chi1_shift = torch.cat((chi1_phi_x, chi1_y_shift.expand(2, p_chi, N)), dim=0)
    #     chi2 = torch.cat((chi2_phi_x, chi2_phi_y), dim=0)
    #     chi2_shift = torch.cat((chi2_phi_x, chi2_y_shift.expand(2, p_chi, N)), dim=0)
    #     chi1_grad_x1 = torch.cat((chi1_phi_grad_x[0:1, :, :], chi1_phi_x[1:2, :, :], chi1_phi_y), dim=0)
    #     chi1_grad_x1_shift = torch.cat((chi1_phi_grad_x[0:1, :, :], chi1_phi_x[1:2, :, :], chi1_y_shift.expand(2, p_chi, N)), dim=0)
    #     chi2_grad_x1 = torch.cat((chi2_phi_grad_x[0:1, :, :], chi2_phi_x[1:2, :, :], chi2_phi_y), dim=0)
    #     chi2_grad_x1_shift = torch.cat((chi2_phi_grad_x[0:1, :, :], chi2_phi_x[1:2, :, :], chi2_y_shift.expand(2, p_chi, N)), dim=0)
    #     chi1_grad_x2 = torch.cat((chi1_phi_x[0:1, :, :], chi1_phi_grad_x[1:2, :, :], chi1_phi_y), dim=0)
    #     chi1_grad_x2_shift = torch.cat((chi1_phi_x[0:1, :, :], chi1_phi_grad_x[1:2, :, :], chi1_y_shift.expand(2, p_chi, N)), dim=0)
    #     chi2_grad_x2 = torch.cat((chi2_phi_x[0:1, :, :], chi2_phi_grad_x[1:2, :, :], chi2_phi_y), dim=0)
    #     chi2_grad_x2_shift = torch.cat((chi2_phi_x[0:1, :, :], chi2_phi_grad_x[1:2, :, :], chi2_y_shift.expand(2, p_chi, N)), dim=0)
    #     chi1_grad_y1 = torch.cat((chi1_phi_x, chi1_phi_grad_y[0:1, :, :], chi1_phi_y[1:2, :, :]), dim=0)
    #     chi2_grad_y1 = torch.cat((chi2_phi_x, chi2_phi_grad_y[0:1, :, :], chi2_phi_y[1:2, :, :]), dim=0)
    #     chi1_grad_y2 = torch.cat((chi1_phi_x, chi1_phi_y[0:1, :, :], chi1_phi_grad_y[1:2, :, :]), dim=0)
    #     chi2_grad_y2 = torch.cat((chi2_phi_x, chi2_phi_y[0:1, :, :], chi2_phi_grad_y[1:2, :, :]), dim=0)
        
        
    #     # 3. 计算 u0 在 x 上的梯度（∇_{x1} u0, ∇_{x2} u0）
    #     u0_phi, u0_grad_phi, u0_grad_grad_phi = model_u0(w1, x1, need_grad=2)
    #     u0_norm_phi = model_u0(w1, x1, normed=False)
    #     u0_norm = torch.sqrt(torch.sum(w1 * u0_norm_phi**2, dim=-1)).unsqueeze(dim=-1) # [d,p,1]

    #     u0_grad_x1 = torch.cat((u0_grad_phi[0:1, :, :], u0_phi[1:2, :, :], torch.ones((2, p_u0, N), dtype=dtype, device=device)), dim=0)
    #     u0_grad_x2 = torch.cat((u0_phi[0:1, :, :], u0_grad_phi[1:2, :, :], torch.ones((2, p_u0, N), dtype=dtype, device=device)), dim=0)
    #     u0_grad_grad_x1 = torch.cat((u0_grad_grad_phi[0:1, :, :], u0_phi[1:2, :, :], torch.ones((2, p_u0, N), dtype=dtype, device=device)), dim=0)
    #     u0_grad_x1_x2 = torch.cat((u0_grad_phi, torch.ones((2, p_u0, N), dtype=dtype, device=device)), dim=0)
    #     u0_grad_grad_x2 = torch.cat((u0_phi[0:1, :, :], u0_grad_grad_phi[1:2, :, :], torch.ones((2, p_u0, N), dtype=dtype, device=device)), dim=0)
    #     alpha_part_chi1 = torch.outer(alpha_chi1, alpha_u0).view(-1)
    #     alpha_part_chi2 = torch.outer(alpha_chi2, alpha_u0).view(-1)


    #     # (chi1 * ∇_{x1}u0 + chi2 * ∇_{x2}u0)
    #     # grad_x1 u0 × chi1
    #     chi1_u0_grad_x1 = torch.einsum('dpn,dqn->dpqn', chi1, u0_grad_x1).flatten(start_dim=1,end_dim=2)
    #     chi1_shift_u0_grad_x1 = torch.einsum('dpn,dqn->dpqn', chi1_shift, u0_grad_x1).flatten(start_dim=1,end_dim=2)
    #     # grad_x2 u0 × chi2
    #     chi2_u0_grad_x2 = torch.einsum('dpn,dqn->dpqn', chi2, u0_grad_x2).flatten(start_dim=1,end_dim=2)
    #     chi2_shift_u0_grad_x2 = torch.einsum('dpn,dqn->dpqn', chi2_shift, u0_grad_x2).flatten(start_dim=1,end_dim=2)

    #     operand_F = torch.cat((u1,u1_shift, chi1_u0_grad_x1,chi1_shift_u0_grad_x1, chi2_u0_grad_x2,chi2_shift_u0_grad_x2),dim=1)
    #     operand_alpha = torch.cat((-alpha_u1,alpha_u1,alpha_part_chi1,-alpha_part_chi1, alpha_part_chi2,-alpha_part_chi2))
    #     error1 = torch.sqrt(Int2TNN(w1234, operand_alpha, operand_F, operand_alpha, operand_F))


    #     print('\n' + '*' * 40)
    #     print('L2 error between current u1 and chi1 * grad_x1 u0 + chi2 * grad_x2 u0:')
    #     print('Absolute error (using 4D integration) = ', error1.item())
    #     print('*' * 40 + '\n')

    #     #=======================
    #     part1 = torch.einsum('dpn,dqn->dpqn', chi1_grad_x1, u0_grad_x1).flatten(start_dim=1,end_dim=2)
    #     part1_shift = torch.einsum('dpn,dqn->dpqn', chi1_grad_x1_shift, u0_grad_x1).flatten(start_dim=1,end_dim=2)
    #     part2 = torch.einsum('dpn,dqn->dpqn', chi2_grad_x1, u0_grad_x2).flatten(start_dim=1,end_dim=2)
    #     part2_shift = torch.einsum('dpn,dqn->dpqn', chi2_grad_x1_shift, u0_grad_x2).flatten(start_dim=1,end_dim=2)

    #     part3 = torch.einsum('dpn,dqn->dpqn', chi1, u0_grad_grad_x1).flatten(start_dim=1,end_dim=2)
    #     part3_shift = torch.einsum('dpn,dqn->dpqn', chi1_shift, u0_grad_grad_x1).flatten(start_dim=1,end_dim=2)

    #     part4 = torch.einsum('dpn,dqn->dpqn', chi2, u0_grad_x1_x2).flatten(start_dim=1,end_dim=2)
    #     part4_shift = torch.einsum('dpn,dqn->dpqn', chi2_shift, u0_grad_x1_x2).flatten(start_dim=1,end_dim=2)

        
    #     F = torch.cat((u1_grad_x1, u1_grad_x1_shift, part1,part1_shift, part2,part2_shift, part3,part3_shift, part4,part4_shift), dim=1)
    #     alpha_F = torch.cat((-alpha_u1,alpha_u1,alpha_part_chi1,-alpha_part_chi1, alpha_part_chi2,-alpha_part_chi2, alpha_part_chi1,-alpha_part_chi1, alpha_part_chi2,-alpha_part_chi2))
    #     error2 = torch.sqrt(Int2TNN(w1234, alpha_F, F, alpha_F, F))

    #     print(('Absolute error between current u1_grad_x1 and chi1 * grad_x1 u0 + chi2 * grad_x2 u0:'), error2.item())
        
    #     #=======================
    #     part5 = torch.einsum('dpn,dqn->dpqn', chi1_grad_y1, u0_grad_x1).flatten(start_dim=1,end_dim=2)
    #     part6 = torch.einsum('dpn,dqn->dpqn', chi2_grad_y1, u0_grad_x2).flatten(start_dim=1,end_dim=2)

    #     F1 = torch.cat((u1_grad_y1, part5, part6), dim=1)
    #     alpha_F1 = torch.cat((-alpha_u1, alpha_part_chi1, alpha_part_chi2))
    #     error3 = torch.sqrt(Int2TNN(w1234, alpha_F1, F1, alpha_F1, F1))

    #     print(('Absolute error between current u1_grad_y1 and chi1 * grad_y1 u0 + chi2 * grad_y2 u0:'), error3.item())

    #     # INSERT_YOUR_CODE
    #     # ========== 画u1关于x1, x2的三维图像 ==========
    #     import matplotlib.pyplot as plt

    #     # 设尺度参数
    #     eps = 10.0

    #     with torch.no_grad():
    #         # 均匀网格
    #         x1_plot = torch.linspace(a, b, 100, dtype=dtype, device=device)
    #         x2_plot = torch.linspace(a, b, 100, dtype=dtype, device=device)
    #         X1, X2 = torch.meshgrid(x1_plot, x2_plot, indexing='ij')
    #         X1_flat = X1.flatten()
    #         X2_flat = X2.flatten()

    #         # 小尺度坐标
    #         y1_flat = X1_flat * eps - (X1_flat * eps).floor()
    #         y2_flat = X2_flat * eps - (X2_flat * eps).floor()
    #         xy_grid = torch.stack([X1_flat, X2_flat], dim=0)  # [2, n_points]
    #         y1y2_grid = torch.stack([y1_flat, y2_flat], dim=0)  # [2, n_points]
    #         n_points = xy_grid.shape[1]
    #         dummy_plot = torch.ones((1, n_points), dtype=dtype, device=device)

    #         # 模型：假定 model_u1_x, model_u1_y 已经加载/训练完毕，并且有 scaling 参数  
    #         u1_phi_x_plot = model_u1_x(dummy_plot, xy_grid, need_grad=0, normed=False)  # [d, p, n_points]
    #         u1_phi_y_plot = model_u1_y(dummy_plot, y1y2_grid, need_grad=0, normed=False)  # [d, p, n_points]
    #         u1_phi_x_plot = u1_phi_x_plot / u1_x_norm
    #         u1_phi_y_plot = u1_phi_y_plot / u1_y_norm

    #         # 归一化的shift
    #         u1_y_shift = model_u1_y(dummy, x_error_0, normed=False)
    #         u1_y_shift = u1_y_shift / u1_y_norm

    #         # 获取alpha
    #         alpha_u1 = model_u1_x.scaling_par()  # [p]

    #         # u1 = sum(alpha * prod(phi_x) * prod(phi_y))
    #         u1_plot = torch.sum(
    #             alpha_u1.unsqueeze(-1) *
    #             torch.prod(u1_phi_x_plot, dim=0) *
    #             (torch.prod(u1_phi_y_plot, dim=0) - torch.prod(u1_y_shift, dim=0)),
    #             dim=0
    #         )  # [n_points]

    #         # 变为2D形状
    #         u1_plot_2d = u1_plot.reshape(100, 100)
    #         u1_value_2d = u1_plot_2d.detach().cpu().numpy()
    #         X1_np = X1.detach().cpu().numpy()
    #         X2_np = X2.detach().cpu().numpy()

    #         # 绘制 u1(x1, x2) 的二维热力图
    #         fig = plt.figure(figsize=(8, 6))
    #         im = plt.imshow(
    #             u1_value_2d,
    #             extent=(a, b, a, b),
    #             origin='lower',
    #             cmap='viridis',
    #             aspect='auto'
    #         )
    #         plt.xlabel('x1', fontsize=12)
    #         plt.ylabel('x2', fontsize=12)
    #         plt.title(r'$u_1(x_1, x_2)$ 2D Heatmap ($\epsilon={}$)'.format(eps), fontsize=14)
    #         cbar = plt.colorbar(im, shrink=0.8, pad=0.03)
    #         cbar.set_label('u1', fontsize=12)
    #         plt.tight_layout()
    #         plt.savefig('png/u1_x1_x2_heatmap.png', dpi=300, bbox_inches='tight')
    #         plt.close(fig)
    #         print("2D heatmap saved to: png/u1_x1_x2_heatmap.png")

    #     # INSERT_YOUR_CODE

    #     # ========== u1 大/小值区域分析 ==========
    #     # 找到 u1 的高值和低值区域，输出最大/最小值、对应的位置，以及高于/低于一定阈值的区域

    #     # 1. Find max/min value and their locations
    #     u1_max = u1_value_2d.max()
    #     u1_min = u1_value_2d.min()
    #     max_indices = np.argwhere(u1_value_2d == u1_max)
    #     min_indices = np.argwhere(u1_value_2d == u1_min)
    #     print(f"最大 u1 值为: {u1_max:.6f}")
    #     for idx, (i, j) in enumerate(max_indices):
    #         print(f"最大值在二维网格的索引: ({i}, {j})，坐标: x1 = {X1_np[i, j]:.4f}, x2 = {X2_np[i, j]:.4f}")
    #     print(f"最小 u1 值为: {u1_min:.6f}")
    #     for idx, (i, j) in enumerate(min_indices):
    #         print(f"最小值在二维网格的索引: ({i}, {j})，坐标: x1 = {X1_np[i, j]:.4f}, x2 = {X2_np[i, j]:.4f}")

    #     # 2. Find regions with "large" (>=90%max) and "small" (<=110%min) values.
    #     thresh_high = u1_max * 0.6  # 调整比例可自己设定
    #     thresh_low = u1_min * 0.6 if u1_min < 0 else u1_min * 0.9  # 适应正/负场景

    #     high_value_indices = np.argwhere(u1_value_2d >= thresh_high)
    #     low_value_indices = np.argwhere(u1_value_2d <= thresh_low)
    #     print(f"高于 {thresh_high:.6f} 的 (x1, x2) 点数量: {len(high_value_indices)}")
    #     print("部分高值点坐标:")
    #     for ind in high_value_indices[:10]:  # 只显示前10个
    #         i, j = ind
    #         print(f"    (x1={X1_np[i, j]:.4f}, x2={X2_np[i, j]:.4f}, u1={u1_value_2d[i, j]:.4f})")
    #     print(f"低于 {thresh_low:.6f} 的 (x1, x2) 点数量: {len(low_value_indices)}")
    #     print("部分低值点坐标:")
    #     for ind in low_value_indices[:10]:
    #         i, j = ind
    #         print(f"    (x1={X1_np[i, j]:.4f}, x2={X2_np[i, j]:.4f}, u1={u1_value_2d[i, j]:.4f})")

    #     # 3. 可视化高值和低值区域（添加等高线显示高值区和低值区）
    #     fig = plt.figure(figsize=(8, 6))
    #     im = plt.imshow(
    #         u1_value_2d,
    #         extent=(a, b, a, b),
    #         origin='lower',
    #         cmap='viridis',
    #         aspect='auto'
    #     )
    #     # 高值等高线
    #     cs_high = plt.contour(
    #         X1_np, X2_np, u1_value_2d, 
    #         levels=[thresh_high], colors='red', linewidths=2
    #     )
    #     plt.clabel(cs_high, inline=True, fontsize=10, fmt={thresh_high: f">=90%max"})
    #     # 低值等高线（蓝色）
    #     cs_low = plt.contour(
    #         X1_np, X2_np, u1_value_2d, 
    #         levels=[thresh_low], colors='pink', linewidths=2
    #     )
    #     plt.clabel(cs_low, inline=True, fontsize=10, fmt={thresh_low: f"<=1.1×min" if u1_min<0 else f'<=90%min'})
    #     plt.xlabel('x1', fontsize=12)
    #     plt.ylabel('x2', fontsize=12)
    #     plt.title(r'$u_1(x_1, x_2)$ 2D Heatmap with High/Low-Value Contours ($\epsilon={}$)'.format(eps), fontsize=14)
    #     cbar = plt.colorbar(im, shrink=0.8, pad=0.03)
    #     cbar.set_label('u1', fontsize=12)
    #     plt.tight_layout()
    #     plt.savefig('png/u1_x1_x2_heatmap_highlowvalue.png', dpi=300, bbox_inches='tight')
    #     plt.close(fig)
    #     print("2D heatmap with high/low-value contours saved to: png/u1_x1_x2_heatmap_highlowvalue.png")

        
    #     #     # INSERT_YOUR_CODE

    #     #     # 画固定 x1 的 u1(x2) 二维图像
    #     #     # 例如，取 x1=a, x1=(a+b)/2, x1=b 三个典型值
    #     #     fixed_x1_list = [0.25, 0.4, 0.78]
    #     #     color_list = ['red', 'green', 'blue']
    #     #     x2_plot = torch.linspace(a, b, 200, dtype=dtype, device=device)

    #     #     plt.figure(figsize=(8, 6))
    #     #     for i, x1_fixed in enumerate(fixed_x1_list):
    #     #         # 创建 x1 全为固定值的 grid
    #     #         x1_values = torch.full_like(x2_plot, x1_fixed)
    #     #         X1_flat = x1_values
    #     #         X2_flat = x2_plot

    #     #         # 小尺度坐标
    #     #         y1_flat = X1_flat * eps - (X1_flat * eps).floor()
    #     #         y2_flat = X2_flat * eps - (X2_flat * eps).floor()
    #     #         xy_grid = torch.stack([X1_flat, X2_flat], dim=0)  # [2, n_points]
    #     #         y1y2_grid = torch.stack([y1_flat, y2_flat], dim=0)  # [2, n_points]
    #     #         n_points = xy_grid.shape[1]
    #     #         dummy_plot = torch.ones((1, n_points), dtype=dtype, device=device)

    #     #         # 重新计算 u1
    #     #         u1_phi_x_plot = model_u1_x(dummy_plot, xy_grid, need_grad=0, normed=False)  # [d, p, n_points]
    #     #         u1_phi_x_plot = u1_phi_x_plot / u1_x_norm
    #     #         u1_phi_y_plot = model_u1_y(dummy_plot, y1y2_grid, need_grad=0, normed=False)  # [d, p, n_points]
    #     #         u1_phi_y_plot = u1_phi_y_plot / u1_y_norm
    #     #         alpha_u1 = model_u1_x.scaling_par()  # [p]
    #     #         u1_y_integral = (torch.sum(w3 * u1_y_norm_phi, dim=-1) / torch.sqrt(torch.sum(w3 * u1_y_norm_phi**2, dim=-1))).unsqueeze(dim=-1)
    #     #         u1_plot = torch.sum(
    #     #             alpha_u1.unsqueeze(-1) *
    #     #             torch.prod(u1_phi_x_plot, dim=0) *
    #     #             (torch.prod(u1_phi_y_plot, dim=0) - torch.prod(u1_y_integral, dim=0)),
    #     #             dim=0
    #     #         )  # [n_points]
    #     #         u1_plot2 = torch.sum(
    #     #             alpha_u1.unsqueeze(-1) *
    #     #             torch.prod(u1_phi_x_plot, dim=0) *
    #     #             (torch.prod(u1_phi_y_plot, dim=0) - torch.prod(u1_y_integral, dim=0)),
    #     #             dim=0
    #     #         )  # [n_points]

    #     #         u1_value_1d = u1_plot.detach().cpu().numpy()
    #     #         u1_value_1d2 = u1_plot2.detach().cpu().numpy()
    #     #         x2_np = X2_flat.detach().cpu().numpy()
    #     #         plt.plot(x2_np, u1_value_1d, color=color_list[i], label=f"$x_1={x1_fixed:.2f}$", linewidth=2, linestyle='-.')
    #     #         plt.plot(x2_np, u1_value_1d2, color=color_list[i], label=f"$x_1={x1_fixed:.2f}$", linewidth=2, linestyle='--')
    #     #     plt.xlabel("$x_2$", fontsize=12)
    #     #     plt.ylabel(r"$u_1(x_1, x_2)$", fontsize=12)
    #     #     plt.title(r"$u_1(x_1, x_2)$ vs $x_2$ for fixed $x_1$, $\epsilon={}$".format(eps), fontsize=14)
    #     #     plt.legend()
    #     #     plt.grid(True)
    #     #     plt.tight_layout()
    #     #     plt.savefig('png/u1_x2_fixed_x1.png', dpi=300, bbox_inches='tight')
    #     #     plt.close()
    #     #     print("2D u1 (fixed x1) plot saved to: png/u1_x2_fixed_x1.png")

