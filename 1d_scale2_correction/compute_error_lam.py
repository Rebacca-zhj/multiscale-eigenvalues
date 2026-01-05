import torch
torch.set_printoptions(profile="full", precision=16)
from matplotlib import pyplot as plt
import scipy.io as sio
import numpy as np

from tnn import *
from integration import *
from quadrature import *

pi = 3.14159265358979323846
dtype = torch.double
device = 'cuda:5'

m = 1
p_chi1 = 20
p_u0 = 18
p_A = 2
p_A0 = p_A + p_A * p_chi1

a, b = 0, 1

def bd_u0(x):
    return (x - a) * (b - x)

def grad_bd_u0(x):
    return a + b - 2 * x

def grad_grad_bd_u0(x):
    return -2 * torch.ones_like(x)

def grad_grad_grad_bd_u0(x):
    return 0 * torch.ones_like(x)

model_u0 = torch.load('./model_u0/x_free_y_per_final_u0.pkl', map_location=device)
x1, w1 = composite_quadrature_1d(8, 0, 1, 200, device=device)
x2, w2 = composite_quadrature_1d(8, 0, 1, 200, device=device)
N = len(x1)
dummy = torch.ones((1, 1), dtype=dtype, device=device)

A = torch.ones((2, 2, N), dtype=dtype, device=device)
A[0, 0, :] = 2 + x1
A[1, 1, :] = torch.sin(2 * pi * x2)
alpha_A = torch.ones(2, dtype=dtype, device=device)

model_chi1_x = torch.load('./model_chi1/x_free_y_per_final_chi1_x.pkl', map_location=device)
model_chi1_y1 = torch.load('./model_chi1/x_free_y_per_final_chi1_y1.pkl', map_location=device)

with torch.no_grad():
    chi1 = torch.ones((2, p_chi1, N), dtype=dtype, device=device)
    chi1_grad_phi = torch.zeros((2, p_chi1, N), dtype=dtype, device=device)
    chi1[0:1, :, :], chi1_grad_phi[0:1, :, :] = model_chi1_x(w1, x1, need_grad=1)
    chi1[1:2, :, :], chi1_grad_phi[1:2, :, :] = model_chi1_y1(w2, x2, need_grad=1)
    alpha_chi1 = model_chi1_x.scaling_par()

    A0 = torch.ones((1, p_A + p_A * p_chi1, N), dtype=dtype, device=device)
    A0[0:1, 0:p_A, :] = A[0:1, :, :]
    A0[0:1, p_A:p_A + p_A * p_chi1, :] = torch.einsum('dpn,dqn->dpqn', A[0:1, :, :], chi1[0:1, :, :]).flatten(start_dim=1, end_dim=2)

    alpha_A0 = torch.ones(p_A + p_A * p_chi1, dtype=dtype, device=device)
    alpha_A0[0:p_A] = torch.sum(w2 * A[1, :, :], dim=-1).squeeze(dim=0) * alpha_A
    alpha_A0[p_A:p_A + p_A * p_chi1] = (torch.einsum('pn,qn->pq', w2 * A[1, :, :], chi1_grad_phi[1, :, :]) * torch.outer(alpha_A, alpha_chi1)).view(-1)

def criterion_0(model_u0):
    u0_phi, u0_grad_phi = model_u0(w1, x1, need_grad=1)
    alpha = torch.ones(p_u0, device=device, dtype=dtype)
    A = Int2TNN_amend_1d_with_coe(w1, w1, alpha, u0_phi, alpha, u0_phi, u0_grad_phi, u0_grad_phi, alpha_A0, A0,
                                  if_sum=False)
    M = Int2TNN(w1, alpha, u0_phi, alpha, u0_phi, if_sum=False)
    L = torch.linalg.cholesky(M)
    C = torch.linalg.solve(L, A.t()).t()
    D = torch.linalg.solve(L, C)
    E, V = torch.linalg.eigh(D)
    ind = torch.argsort(E)
    lam1 = E[ind]
    u1 = torch.linalg.solve(L.t(), V[:, ind])
    return lam1, u1

lam, u = criterion_0(model_u0)


eig_ms_FEM = np.empty((0, 6))
# eig_ms_FEM = np.vstack((eig_ms_FEM, [22.920721577215190, 92.671721037727266, 202.320838867336249, 383.897108711624355, 586.341519031419466, 839.471009526864577]))   # ε=3 的情况
# eig_ms_FEM = np.vstack((eig_ms_FEM, [22.72153431, 91.0404723, 206.905854, 351.6103201, 605.555381, 846.7852382]))   # ε=4 的情况
# eig_ms_FEM = np.vstack((eig_ms_FEM, [22.611156540678955, 90.517251162099413, 203.538837991687984, 365.704775028756728, 539.165556593393262, 879.636282985819207]))   # ε=5 的情况
# eig_ms_FEM = np.vstack((eig_ms_FEM, [22.53992323, 90.23343869, 202.7344157, 359.8294512, 568.7513389, 764.3442633]))   # ε=6 的情况
# eig_ms_FEM = np.vstack((eig_ms_FEM, [22.48983098, 90.04572309, 202.3476137, 358.8702215, 559.4010244, 815.8237371]))   # ε=7 的情况
# eig_ms_FEM = np.vstack((eig_ms_FEM, [22.452571741400472, 89.909280203913269, 202.093692233041111, 358.542422819675835, 558.458595874086654, 801.803104845780695]))   # ε=8 的情况
# eig_ms_FEM = np.vstack((eig_ms_FEM, [22.42372713, 89.80449886, 201.9038427, 358.3438012, 558.4296825, 801.0532745]))   # ε=9 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM, [22.400713985024286, 89.721031714657215, 201.752748724725620, 358.187673460167218, 558.471394027006227, 801.630942808510895]))   # ε=10 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM, [22.366263929218789, 89.595753720315372, 201.523006894734010, 357.934223079881519, 558.468293112219612, 802.523937834335925]))   # ε=12 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM, [22.331855046544749, 89.469623859181070, 201.285439234642070, 357.642600256422440, 558.325122120440710, 803.002851132699107]))   # ε=15 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM, [22.2974415198243, 89.3418726786610, 201.0364580456397, 357.3036373895815, 558.0279646081377, 803.0456727101908]))   # ε=20 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM, [22.2767727317817, 89.2641464189675, 200.8803202198242, 357.0749538050971, 557.7755336299408, 802.8828269084711]))   # ε=25 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM, [22.2629791679403, 89.2117890793396, 200.7730331023456, 356.9111948662018, 557.5762507111131, 802.7011079265938]))   # ε=30 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM, [22.2531178098325, 89.1740981565860, 200.6947143509824, 356.7884344846155, 557.4185614607940, 802.5364934717118]))   # ε=35 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM, [22.2457163028899, 89.1456589946281, 200.6350070879298, 356.6930930117984, 557.2917953334094, 802.3941906993421]))   # ε=40 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM, [22.2353454417953, 89.1055841270903, 200.5499617777880, 356.5547594247445, 557.1018923234258, 802.1678847176478]))   # ε=50 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM, [22.22347906, 89.1055841270903, 200.5499617777880, 356.5547594247445, 557.1018923234258, 802.1678847176478]))   # ε=70 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM, [22.2145687, 89.1055841270903, 200.5499617777880, 356.5547594247445, 557.1018923234258, 802.1678847176478]))   # ε=100 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM, [22.2076316, 89.1055841270903, 200.5499617777880, 356.5547594247445, 557.1018923234258, 802.1678847176478]))   # ε=150 的情况
eig_ms_FEM = np.vstack((eig_ms_FEM, [22.2041607388434, 89.1055841270903, 200.5499617777880, 356.5547594247445, 557.1018923234258, 802.1678847176478]))   # ε=200 的情况

epsilons = [10,12,15,20,25,30,35,40,50,70,100,150,200]
lam_0_lam_1 = np.empty((0, m))
lam_0_lam_1_lam_2 = np.empty((0, m))


for idx, eps in enumerate(epsilons):
    lam1 = 2.0857067746403288
    lam_0_lam_1 = np.vstack((lam_0_lam_1, (lam[:m] + lam1 / eps).detach().cpu().numpy()))
    
    
def plot_eigen_error_vs_epsilon(epsilon_list, eigen_error_list, alpha=None, C=None, show=True, save_path=None, filename=None, ylabel='eigen_error'):
    """
    Plot loglog error convergence curve for eigenvalue error vs epsilon, only first eigenpair and its reference line.
    """
    epsilon_arr = np.array(epsilon_list)
    error_arr = np.array(eigen_error_list)
    m, N = error_arr.shape
    plt.figure(figsize=(7, 5))
    color = plt.cm.viridis(0.0)
    err_col = error_arr[:, 0]
    plt.loglog(epsilon_arr, err_col, 'o-', color=color, label=f'eigen pair 1')
    fit_alpha = alpha
    fit_C = C
    if alpha is None or C is None:
        log_eps = np.log(epsilon_arr)
        log_err = np.log(err_col)
        coeffs = np.polyfit(log_eps, log_err, 1)
        fitted_alpha = coeffs[0]
        fitted_logC = coeffs[1]
        if alpha is None:
            fit_alpha = fitted_alpha
        if C is None:
            fit_C = np.exp(fitted_logC)
    eps_ref = epsilon_arr[-1]
    err_ref = err_col[-1]
    std_line = err_ref * (epsilon_arr / eps_ref) ** fit_alpha
    plt.loglog(epsilon_arr, std_line, '--', color=color, label=f'standard line: $c\\epsilon^{{{fit_alpha:.2f}}}$')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(ylabel)
    if filename is not None:
        plt.title(f'{filename}')
    else:
        plt.title(r'error ~ $\epsilon$ ')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='lower right', fontsize=8)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    else:
        plt.savefig('png/eigen_error_vs_epsilon.png', bbox_inches='tight', dpi=200)
    if show:
        plt.show()
    plt.close()

if __name__ == '__main__':
    eps = [1 / x for x in epsilons]
    eps = np.array(eps)

    lam_0 = lam[:m].detach().cpu().numpy()
    # 计算两个误差
    eigen_error_1 = np.abs(eig_ms_FEM - lam_0)
    eigen_error_2 = np.abs(eig_ms_FEM - lam_0_lam_1)

    # 只画第一个特征对(ground state)
    error1 = eigen_error_1[:, 0]
    error2 = eigen_error_2[:, 0]

    plt.figure(figsize=(7, 5))
    color1 = plt.cm.viridis(0.0)
    color2 = plt.cm.viridis(0.3)

    # 数据线
    plt.loglog(eps, error1, 'o-', color=color1, label=r'$\| \lambda_0^{TNN} - \lambda_{\epsilon}^{FEM} \|$')
    plt.loglog(eps, error2, 's-', color=color2, label=r'$\| \lambda_0^{TNN} + \epsilon\lambda_1^{TNN} - \lambda_{\epsilon}^{FEM} \|$')

    # 拟合第一个误差的阶
    log_eps = np.log(eps)
    log_err1 = np.log(error1)
    log_err2 = np.log(error2)
    coeffs1 = np.polyfit(log_eps, log_err1, 1)
    coeffs2 = np.polyfit(log_eps, log_err2, 1)
    alpha1 = 1 if np.all(np.isfinite(coeffs1)) else coeffs1[0]
    alpha2 = 2 if np.all(np.isfinite(coeffs2)) else coeffs2[0]

    # ref line for error 1 (斜率1)
    eps_ref = eps[-1]
    err1_ref = error1[-1]
    ref_line1 = err1_ref * (eps / eps_ref) ** 1
    plt.loglog(eps, ref_line1, '--', color=color1, alpha=0.6, label=r'ref: slope=1')

    # ref line for error 2 (斜率2)
    err2_ref = error2[-1]
    ref_line2 = err2_ref * (eps / eps_ref) ** 2
    plt.loglog(eps, ref_line2, '--', color=color2, alpha=0.6, label=r'ref: slope=2')

    plt.xlabel(r'$\epsilon$')
    plt.ylabel('eigenvalue error')
    plt.title(r'Eigenvalue error vs $\epsilon$ (first eigenpair)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig('png/eigen_error_lam0_lam1_together.png', bbox_inches='tight', dpi=200)
    plt.show()
    plt.close()

    print('lam_0_lam_1=', lam_0_lam_1)
    print('Done.')
