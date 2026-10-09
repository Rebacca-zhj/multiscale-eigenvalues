"""
独立绘制 L2 误差 log-log 图。

依赖数据（来自 output_error_L2_txt.py 的终端输出）：
    inv_e = [3, 5, 8, 10, 12, 15]

误差数据：
    l2_error0: 2.376e-3, 1.477e-3, 9.343e-4, 7.494e-4, 6.254e-4, 5.009e-4
    l2_error1: 3.811e-4, 1.226e-4, 4.670e-5, 2.974e-5, 2.061e-5, 1.318e-5
    l2_error2: 2.467e-4, 6.154e-5, 2.073e-5, 1.278e-5, 8.693e-6, 5.478e-6
    l2_error3: 3.085e-4, 7.293e-5, 2.363e-5, 1.405e-5, 9.017e-6, 5.732e-6
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ---------- 1. 准备数据 ----------
inv_e = [3, 5, 8, 10, 12, 15]
epsilons = 1.0 / np.array(inv_e)

l2_error0 = np.array([2.3764307336521e-3, 1.4776044642855e-3, 9.3439170503100e-4,
                      7.4948994445620e-4, 6.2546365898360e-4, 5.0094803813210e-4])

l2_error1 = np.array([3.8106456783148715e-4, 1.2261399608616219e-4, 4.6697083964002719e-5,
                      2.9740328383680142e-5, 2.0609249945461032e-5, 1.3179680745570849e-5])

l2_error2 = np.array([2.4674591497182865e-4, 6.1540591026529960e-5, 2.0728356153584847e-5,
                      1.2775951087458961e-5, 8.6930726910561159e-6, 5.4782605174426157e-6])

l2_error3 = np.array([3.0851300305543205e-4, 7.2931998960978605e-5, 2.3629627712804421e-5,
                      1.4053965111964856e-5, 9.0173419501370397e-6, 5.7320378450482507e-6])


# ---------- 2. 创建 png 目录 ----------
if not os.path.exists('./png'):
    os.makedirs('./png')


# ---------- 3. 画 loglog 误差图 ----------
plt.figure(figsize=(8, 6))

plt.loglog(epsilons, l2_error0, marker='o',
           label=r'$||u_\varepsilon - u_0||_{L^2}$')
plt.loglog(epsilons, l2_error1, marker='o',
           label=r'$||u_\varepsilon - (u_0 + \varepsilon (u_1+\tilde{u}_1))||_{L^2}$')
plt.loglog(epsilons, l2_error2, marker='o',
           label=r'$||u_\varepsilon - (u_0 + \varepsilon (u_1+\tilde{u}_1) + \varepsilon^2 (u_2+\delta u_2+\tilde{u}_2))||_{L^2}$')
plt.loglog(epsilons, l2_error3, marker='o',
           label=r'$||u_\varepsilon - (u_0 + \varepsilon (u_1+\tilde{u}_1) + \varepsilon^2 (u_2+\delta u_2+\tilde{u}_2) + \varepsilon^3 u_3)||_{L^2}$')

# 参考线：斜率 1、2、3
ref_x = np.array([epsilons[0], epsilons[-1]])
ref1 = l2_error0[0] * (ref_x / ref_x[0])**1   # slope = 1
ref2 = l2_error1[0] * (ref_x / ref_x[0])**2   # slope = 2
ref3 = l2_error2[0] * (ref_x / ref_x[0])**3   # slope = 3

plt.loglog(ref_x, ref1, 'k--', alpha=0.7, label=r'ref: slope=1')
plt.loglog(ref_x, ref2, 'k-.', alpha=0.7, label=r'ref: slope=2')
plt.loglog(ref_x, ref3, 'k:',  alpha=0.7, label=r'ref: slope=3')

plt.xlabel(r'$\epsilon$', fontsize=12)
plt.ylabel('L2 Error', fontsize=12)
plt.title(r'L2 Error vs $\epsilon$ (log-log)', fontsize=14)
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend(fontsize=9)
plt.tight_layout()

# 保存
out_path = './png/data_error_vs_epsilon_loglog_txt.png'
plt.savefig(out_path, dpi=150)
plt.show()
print(f'Saved to: {out_path}')
