import numpy as np
import matplotlib.pyplot as plt
import os

def read_txt_data(filepath):
    # 读取以空格分隔的二维数值文本，返回 numpy 数组
    # 支持一维或二维
    with open(filepath, 'r') as f:
        data = []
        for line in f:
            row = [float(x) for x in line.strip().split()]
            data.append(row)
    arr = np.array(data)
    # 如果为(N,1)，自动挤压一维
    if arr.shape[1] == 1:
        arr = arr.squeeze(1)
    return arr

def gather_fem_data(root, e_folder, quad_folder, rank_num=4, R_str="R9"):
    # 合并所有rank的有限元解与积分点坐标
    # 返回 (all_coords: (N,2), all_weights: (N,), all_u: (N,))
    coord_list = []
    w_list = []
    u_list = []
    for rank in range(rank_num):
        # 读取坐标
        coord_path = os.path.join(root, quad_folder, f"coord_{R_str}_rank{rank}.txt")
        weight_path = os.path.join(root, quad_folder, f"w_{R_str}_rank{rank}.txt")
        u_path = os.path.join(root, e_folder, f"u_fem_{R_str}_rank{rank}.txt")
        if not(os.path.exists(coord_path) and os.path.exists(weight_path) and os.path.exists(u_path)):
            print(f"警告: 缺少{rank}号进程的数据，跳过")
            continue
        coords = read_txt_data(coord_path)   # (n_points, 2)
        weights = read_txt_data(weight_path) # (n_points,)
        u = read_txt_data(u_path)            # (n_points,)
        if coords.ndim == 1:
            coords = coords[:,None]   # 某些稀疏节点情况下只有一维
        coord_list.append(coords)
        w_list.append(weights)
        u_list.append(u)
    if len(coord_list) == 0:
        raise RuntimeError("未能读取任何进程的数据.")
    all_coords = np.vstack(coord_list)
    all_weights = np.concatenate(w_list)
    all_u = np.concatenate(u_list)
    return all_coords, all_weights, all_u

def plot_fem_2d_scatter(x, y, u, save_path, marker_size=6, title="", xlabel="x", ylabel="y", cbar_label="u_fem"):
    plt.figure(figsize=(8,6))
    sc = plt.scatter(x, y, c=u, s=marker_size, cmap='viridis', marker='s')
    cbar = plt.colorbar(sc)
    cbar.set_label(cbar_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")

if __name__ == '__main__':
    root = "/home/zhanghuaijia/multiscale_eigen/paper1_example/d2_scale2_ex4_2/periodic_P2_R9_260825"
    quad_folder = "quad"
    rank_num = 4            # 你可以改成自己用的分进程数
    R_str = "R9"            # 文件名中的编号
    # 找出所有 epsilon 文件夹，比如 e_3、e_5 等
    e_folders = [d for d in os.listdir(root) if d.startswith('e_') and os.path.isdir(os.path.join(root, d))]
    if not e_folders:
        print("未找到形如e_xx的子文件夹，请检查目录结构")
        exit(0)
    save_dir = os.path.join(root, "png")
    os.makedirs(save_dir, exist_ok=True)

    # 支持多个epsilon分别画图
    for e_folder in sorted(e_folders):
        # 可从e_folder名称中提取epsilon值做标题
        try:
            epsilon = float(e_folder.split("_")[1])
        except:
            epsilon = e_folder
        try:
            coords, weights, u_fem = gather_fem_data(root, e_folder, quad_folder, rank_num=rank_num, R_str=R_str)
        except Exception as e:
            print(f"读取{e_folder}时出错: {e}")
            continue
        x = coords[:,0]
        y = coords[:,1]
        title = f'FEM Solution, $\\varepsilon={epsilon}$'
        save_path = os.path.join(save_dir, f"u_fem_eps{epsilon}.png")
        plot_fem_2d_scatter(x, y, u_fem, save_path, title=title)