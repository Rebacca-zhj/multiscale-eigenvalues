A = (2+sin(2 pi x1))(2+sin(2 pi x2))(1+0.5*cos(2 * pi * y1))(1+0.5*cos(2 * pi * y2))
F = 30*sin(2 pi x1)*cos(2 pi x2)

1step_chi1.py
50000步Adam
test1 [1, p_net_periodic, p_net, p_net, 20] 2 50 loss = 8.901669837281452e-06 
test2 [1, p_net_periodic, p_net, p_net, 20] 2 20 loss = 1.2988890144230114e-05
test3 [1, p_net_periodic, p_net, p_net, p_net, 20] 2 20 loss = 4.4428456195793685e-06
test4 [1, p_net_periodic, p_net, p_net, p_net, 20] 2 50 loss = 3.3186461703956256e-06 (✅)

2step_chi2.py
选择网络[1,2,50,50,50,20]
50000步Adam
loss = 7.440799014247529e-06

3step_u0.py
20000步Adam（弱形式）+ 50000步Adam（强形式）+ 5000步LBFGS
sizes_periodic = [1, 2, 20, 20, 20, 20] # test1 loss = 0.00039871638656340224
sizes_periodic = [1, 2, 50, 50, 50, 20] # test2 loss = 8.67760111908854e-05（✅）
sizes_periodic = [1, 5, 80, 80, 80, 20] # test3 损失函数不下降
sizes_periodic = [1, 2, 50, 50, 20] # test4 loss = 0.0001719728597632713

4step_u1.py
50000步Adam + 5000步LBFGS
sizes_periodic = [1, 2, 20, 20, 20, 20] #test1 loss = 0.005741595518075647
sizes_periodic = [1, 2, 50, 50, 50, 20] #test2 loss = 3.497126452992761
sizes_periodic = [1, 5, 80, 80, 80, 20] #test3 loss = 0.0002453737469624951 (✅)
sizes_periodic = [1, 2, 50, 50, 50, 50, 20] #test4 loss = 3.4971264444943717

5step_u2.py
50000步Adam + 5000步LBFGS
sizes_periodic = [1, 2, 20, 20, 20, 20] #test1 loss = 2.4794044898677585
sizes_periodic = [1, 2, 50, 50, 50, 20] #test2 loss = 3.1760302257324775
sizes_periodic = [1, 5, 80, 80, 80, 20] #test3 loss = 0.022282080905402856
sizes_periodic = [1, 8, 100, 100, 100, 20] #test4 loss = 0.0036827500314539617 (✅)
sizes_periodic = [1, 2, 50, 50, 50, 50, 20] #test5 loss = 2.4794002996191686
sizes_periodic = [1, 5, 80, 80, 80, 80, 20] #test6 loss = 2.479394206777957
sizes_periodic = [1, 8, 100, 100, 100, 100, 20] #test7 loss = 3.6468436043990744
sizes_periodic = [1, 8, 150, 150, 150, 20] #test8  loss = 0.005324410412208355
sizes_periodic = [1, 8, 150, 150, 150, 150, 20] #test9 loss = 0.004466407163876579
sizes_periodic = [1, 8, 100, 100, 100, 40] #test10 loss = 0.009699490950034978
sizes_periodic = [1, 8, 150, 150, 150, 40] #test11 loss = 0.0030214875356698568
sizes_periodic = [1, 8, 150, 150, 150, 150, 40] #test12 loss = 0.02738878650572112

6step_tilde_u1.py
50000步Adam + 5000步LBFGS
sizes_periodic = [1, 2, 50, 50, 50, p_tilde_u1] #test1 loss = 3.834463070164351e-05
sizes_periodic = [1, 5, 80, 80, 80, p_tilde_u1] #test2 loss = 2.171476908614821e-06
sizes_periodic = [1, 8, 100, 100, 100, p_tilde_u1] # test3 loss = 7.499653456360247e-07 (✅)

7step_delta_u2.py
50000步Adam + 5000步LBFGS
sizes_periodic = [1, 8, 100, 100, 100, p_tilde_u1] # test3 loss = 7.58628272757744e-06 (✅)

8step_u3.py
sizes_periodic = [1, 2, 50, 50, 50, 20] #test1 loss = 7.087705480820808
sizes_periodic = [1, 5, 80, 80, 80, 20] #test2 loss = 0.05238629089784077
sizes_periodic = [1, 8, 100, 100, 100, 20] # test3 loss = 0.04048767926994342
sizes_periodic = [1, 8, 150, 150, 150, 20] # test4 loss = 0.012043522305504184(✅)
sizes_periodic = [1, 2, 50, 50, 50, 50, 20] #test5 loss = 7.0850127764026585
sizes_periodic = [1, 5, 80, 80, 80, 80, 20] #test6 loss = 0.01602652136437395
sizes_periodic = [1, 8, 100, 100, 100, 100, 20] # test7 loss = 7.083858095405842
sizes_periodic = [1, 8, 150, 150, 150, 150, 20] # test8 loss = 2.1954758497231346
sizes_periodic = [1, 8, 200, 200, 200, 20] # test9 loss = 7.106152828420416
sizes_periodic = [1, 8, 200, 200, 200, 200, 20] # test10 loss = 2.2095364970753137
sizes_periodic = [1, 5, 80, 80, 80, 40] #test11 loss = 2.1929037389896604
sizes_periodic = [1, 8, 150, 150, 150, 40] # test12 loss = 7.106152828420416
sizes_periodic = [1, 5, 80, 80, 80, 80, 40] #test13 loss = 2.2078542575763436
sizes_periodic = [1, 8, 200, 200, 200, 40] # test14 loss = 0.21994860543217257

9step_tilde_u2.py
sizes_periodic = [1, 2, 50, 50, 50, 20] #test1 loss = 38.956921642808396
sizes_periodic = [1, 5, 80, 80, 80, 20] #test2 loss = 0.14844737134487795
sizes_periodic = [1, 8, 100, 100, 100, 20] # test3 loss = 0.08467888962555953
sizes_periodic = [1, 8, 150, 150, 150, 20] # test4 loss = 0.0678646839178611(✅)

L2误差
plt.loglog(epsilons, l2_error0_result_eps.cpu().numpy(), marker='o', label=r'$||u_\varepsilon - u_0||_{L^2}$')
plt.loglog(epsilons, l2_error1_result_eps.cpu().numpy(), marker='o', label=r'$||u_\varepsilon - (u_0 + \varepsilon * (u_1+\tilde{u}_1))||_{L^2}$')
plt.loglog(epsilons, l2_error2_result_eps.cpu().numpy(), marker='o', label=r'$||u_\varepsilon - (u_0 + \varepsilon * (u_1+\tilde{u}_1) + \varepsilon^2 * (u_2 + \delta u_2 + \tilde{u}_2))||_{L^2}$')
plt.loglog(epsilons, l2_error3_result_eps.cpu().numpy(), marker='o', label=r'$||u_\varepsilon - (u_0 + \varepsilon * (u_1+\tilde{u}_1) + \varepsilon^2 * (u_2 + \delta u_2 + \tilde{u}_2) + \varepsilon^3 * (u_3))||_{L^2}$')

l2 error0:  tensor([0.0108825342894687, 0.0074684052855219, 0.0048404170575854,
        0.0039049750086850, 0.0032690966220085, 0.0026251334140620],
       device='cuda:1', dtype=torch.float64)
l2 error1:  tensor([0.0063701051454763, 0.0019045768953487, 0.0006852074512419,
        0.0004272327548169, 0.0002926939170352, 0.0001853335241456],
       device='cuda:1', dtype=torch.float64)
l2 error2:  tensor([4.4982049883058976e-03, 9.2974631652210178e-04, 2.1931711383181278e-04,
        1.1083703276007754e-04, 6.3711289815780505e-05, 3.2523705780467571e-05],
       device='cuda:1', dtype=torch.float64)
l2 error3:  tensor([2.3946138485520627e-03, 3.5477001980232825e-04, 6.6854542053796428e-05,
        2.9756996168854553e-05, 1.5755968406193436e-05, 7.4502374576496447e-06],
       device='cuda:1', dtype=torch.float64)


H1误差
||u_e - u_0||_H^1 : [0.14924100250293001, 0.08588406677001892, 0.05251185062750863, 0.041793466565704755, 0.027777997569423517]
||u_e - (u_0+e*\hat u_1)||_H^1 : [0.07506338458726212, 0.02687468648787974, 0.010422752089255237, 0.006733327499551044, 0.0036291968678042476]
||u_e - (u_0+e\hat u_1+e^2\hat u_2)||_H^1 : [0.07506313359412872, 0.026874758325403862, 0.010422751824141019, 0.006733328890046904, 0.0036291969805865476]
