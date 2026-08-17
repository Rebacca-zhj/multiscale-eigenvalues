"""
A矩阵定义模块
所有A矩阵相关的定义都集中在这个文件中，其他文件只需要导入使用即可
"""
import torch

# 默认常量
pi = 3.14159265358979323846


def init_A_matrices(x1, x2, x3, x4, pi=pi, dtype=torch.double, device='cuda:0'):
    """
    初始化所有A矩阵相关的变量
    
    参数:
        x1, x2, x3, x4: 积分点坐标
        pi: 圆周率，默认使用模块中的pi
        dtype: 数据类型，默认torch.double
        device: 设备，默认'cuda:0'
    
    返回:
        包含所有A相关变量的字典
    """
    N = len(x1)
    
    # 定义A矩阵
    # A = 3 + sin(2 * pi * x1) + cos(2 * pi * x2) + sin(2 * pi * x3) * sin(2 * pi * x4)
    A = torch.ones((4, 3, N), dtype=dtype, device=device)
    A[0, 0, :] = 3 + torch.sin(2 * pi * x1)
    A[1, 1, :] = torch.cos(2 * pi * x2)
    A[2, 2, :] = torch.sin(2 * pi * x3)
    A[3, 2, :] = torch.sin(2 * pi * x4)
    alpha_A = torch.tensor([1,1,1], dtype=dtype, device=device)
    
    # 定义A的梯度
    A_grad = torch.zeros((4, 3, N), dtype=dtype, device=device)
    A_grad[0, 0, :] = 2 * pi * torch.cos(2 * pi * x1)
    A_grad[1, 1, :] = -2 * pi * torch.sin(2 * pi * x2)
    A_grad[2, 2, :] = 2 * pi * torch.cos(2 * pi * x3)
    A_grad[3, 2, :] = 2 * pi * torch.cos(2 * pi * x4)
    
    # 定义A的偏导数组合
    A_grad_x1 = torch.cat((A_grad[0:1], A[1:4]), dim=0)
    alpha_A_grad_x1 = alpha_A
    A_grad_x2 = torch.cat((A[0:1], A_grad[1:2], A[2:4]), dim=0)
    alpha_A_grad_x2 = alpha_A
    A_grad_y1 = torch.cat((A[0:2], A_grad[2:3], A[3:4]), dim=0)
    alpha_A_grad_y1 = alpha_A
    A_grad_y2 = torch.cat((A[0:3], A_grad[3:4]), dim=0)
    alpha_A_grad_y2 = alpha_A
    
    # 返回所有变量
    return {
        'A': A,
        'alpha_A': alpha_A,
        'A_grad': A_grad,
        'A_grad_x1': A_grad_x1,
        'alpha_A_grad_x1': alpha_A_grad_x1,
        'A_grad_x2': A_grad_x2,
        'alpha_A_grad_x2': alpha_A_grad_x2,
        'A_grad_y1': A_grad_y1,
        'alpha_A_grad_y1': alpha_A_grad_y1,
        'A_grad_y2': A_grad_y2,
        'alpha_A_grad_y2': alpha_A_grad_y2,
    }


# 为了向后兼容，如果直接运行此文件，仍然执行原来的代码
if __name__ == '__main__':
    torch.set_printoptions(profile="full", precision=16)
    from integration import *
    from quadrature import *
    
    dtype = torch.double
    device = 'cuda:0'
    
    x1, w1 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
    x2, w2 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
    x3, w3 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
    x4, w4 = composite_quadrature_1d(4, 0, 1, 200, dtype=dtype, device=device)
    
    A_dict = init_A_matrices(x1, x2, x3, x4, pi=pi, dtype=dtype, device=device)
    
    # 将变量设置为模块级别的变量，以便直接访问
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