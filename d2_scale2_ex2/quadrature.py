import torch
import numpy as np

from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.special import gamma, roots_jacobi

def quadrature_1d(N, dtype=torch.double, device='cpu'):
    """
    Quadrature points and weights for one-dimensional Gauss-Legendre quadrature rules in computational domain [-1,1].
    
    Parameters:
        N: number of quadrature points in domain [1-,1]
        dtype, device
    Returns:
        X: quadrature points size([N])
        W: quadrature weights size([N])
    """
    if N == 1:
        coord = torch.tensor([[0, 2]],dtype=dtype,device=device)
    elif N == 2:
        coord = torch.tensor([[-np.sqrt(3) / 3, 1],
                            [np.sqrt(3) / 3, 1]],dtype=dtype,device=device)
    elif N == 3:
        coord = torch.tensor([[-np.sqrt(15) / 5, 5 / 9],
                            [0, 8 / 9],
                            [np.sqrt(15) / 5, 5 / 9]],dtype=dtype,device=device)
    elif N == 4:
        coord = torch.tensor([[-np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7), (18 - np.sqrt(30)) / 36],
                            [-np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7), (18 + np.sqrt(30)) / 36],
                            [np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7), (18 + np.sqrt(30)) / 36],
                            [np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7), (18 - np.sqrt(30)) / 36]],dtype=dtype,device=device)
    elif N == 5:
        coord = torch.tensor([[-1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7)), (322 - 13 * np.sqrt(70)) / 900],
                            [- 1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)), (322 + 13 * np.sqrt(70)) / 900],
                            [0, 128 / 225],
                            [1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)), (322 + 13 * np.sqrt(70)) / 900],
                            [1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7)), (322 - 13 * np.sqrt(70)) / 900]],dtype=dtype,device=device)
    elif N == 6:
        coord = torch.tensor([[-0.932469514203152, 0.171324492379170],
                            [-0.661209386466264, 0.360761573048139],
                            [-0.238619186083197, 0.467913934572691],
                            [0.238619186083197, 0.467913934572691],
                            [0.661209386466264, 0.360761573048139],
                            [0.932469514203152, 0.171324492379170]],dtype=dtype,device=device)
    elif N == 7:
        coord = torch.tensor([[-0.949107912342758, 0.129484966168870],
                            [-0.741531185599394, 0.279705391489277],
                            [-0.405845151377397, 0.381830050505119],
                            [0, 0.417959183673469],
                            [0.405845151377397, 0.381830050505119],
                            [0.741531185599394, 0.279705391489277],
                            [0.949107912342758, 0.129484966168870]],dtype=dtype,device=device)
    elif N == 8:
        coord = torch.tensor([[-0.960289856497536, 0.101228536290377],
                            [-0.796666477413627, 0.222381034453374],
                            [-0.525532409916329, 0.313706645877887],
                            [-0.183434642495650, 0.362683783378362],
                            [0.183434642495650, 0.362683783378362],
                            [0.525532409916329, 0.313706645877887],
                            [0.796666477413627, 0.222381034453374],
                            [0.960289856497536, 0.101228536290377]],dtype=dtype,device=device)
    elif N == 9:
        coord = torch.tensor([[-0.968160239507626, 0.0812743883615744],
                            [-0.836031107326636, 0.180648160694858],
                            [-0.613371432700590, 0.260610696402936],
                            [-0.324253423403809, 0.312347077040003],
                            [0.0, 0.330239355001260],
                            [0.324253423403809, 0.312347077040003],
                            [0.613371432700590, 0.260610696402936],
                            [0.836031107326636, 0.180648160694858],
                            [0.968160239507626, 0.0812743883615744]],dtype=dtype,device=device)
    elif N == 10:
        coord = torch.tensor([[-0.973906528517172, 0.0666713443086881],
                            [-0.865063366688985, 0.149451349150581],
                            [-0.679409568299024, 0.219086362515982],
                            [-0.433395394129247, 0.269266719309997],
                            [-0.148874338981631, 0.295524224714753],
                            [0.148874338981631, 0.295524224714753],
                            [0.433395394129247, 0.269266719309997],
                            [0.679409568299024, 0.219086362515982],
                            [0.865063366688985, 0.149451349150581],
                            [0.973906528517172, 0.0666713443086881]],dtype=dtype,device=device)
    elif N == 11:
        coord = torch.tensor([[-0.978228658146057, 0.0556685671161737],
                            [-0.887062599768095, 0.125580369464904],
                            [-0.730152005574049, 0.186290210927734],
                            [-0.519096129206812, 0.233193764591991],
                            [-0.269543155952345, 0.262804544510247],
                            [0.0, 0.272925086777901],
                            [0.269543155952345, 0.262804544510247],
                            [0.519096129206812, 0.233193764591991],
                            [0.730152005574049, 0.186290210927734],
                            [0.887062599768095, 0.125580369464904],
                            [0.978228658146057, 0.0556685671161737]],dtype=dtype,device=device)
    elif N == 12:
        coord = torch.tensor([[-0.981560634246719, 0.0471753363865118],
                            [-0.904117256370475, 0.106939325995318],
                            [-0.769902674194305, 0.160078328543345],
                            [-0.587317954286617, 0.203167426723066],
                            [-0.367831498998180, 0.233492536538356],
                            [-0.125233408511469, 0.249147045813403],
                            [0.125233408511469, 0.249147045813403],
                            [0.367831498998180, 0.233492536538356],
                            [0.587317954286617, 0.203167426723066],
                            [0.769902674194305, 0.160078328543345],
                            [0.904117256370475, 0.106939325995318],
                            [0.981560634246719, 0.0471753363865118]],dtype=dtype,device=device)
    elif N == 13:
        coord = torch.tensor([[-0.984183054718588, 0.0404840047653159],
                            [-0.917598399222978, 0.0921214998377285],
                            [-0.801578090733310, 0.138873510219789],
                            [-0.642349339440340, 0.178145980761946],
                            [-0.448492751036447, 0.207816047536889],
                            [-0.230458315955135, 0.226283180262898],
                            [0.0, 0.232551553230874],
                            [0.230458315955135, 0.226283180262898],
                            [0.448492751036447, 0.207816047536889],
                            [0.642349339440340, 0.178145980761946],
                            [0.801578090733310, 0.138873510219789],
                            [0.917598399222978, 0.0921214998377285],
                            [0.984183054718588, 0.0404840047653159]],dtype=dtype,device=device)
    elif N == 14:
        coord = torch.tensor([[-0.986283808696812, 0.0351194603317519],
                            [-0.928434883663574, 0.0801580871597603],
                            [-0.827201315069765, 0.121518570687902],
                            [-0.687292904811685, 0.157203167158193],
                            [-0.515248636358154, 0.185538397477937],
                            [-0.319112368927890, 0.205198463721295],
                            [-0.108054948707344, 0.215263853463158],
                            [0.108054948707344, 0.215263853463158],
                            [0.319112368927890, 0.205198463721295],
                            [0.515248636358154, 0.185538397477937],
                            [0.687292904811685, 0.157203167158193],
                            [0.827201315069765, 0.121518570687902],
                            [0.928434883663574, 0.0801580871597603],
                            [0.986283808696812, 0.0351194603317519]],dtype=dtype,device=device)
    elif N == 15:
        coord = torch.tensor([[-0.987992518020485, 0.0307532419961174],
                            [-0.937273392400706, 0.0703660474881081],
                            [-0.848206583410427, 0.107159220467172],
                            [-0.724417731360170, 0.139570677926155],
                            [-0.570972172608539, 0.166269205816993],
                            [-0.394151347077563, 0.186161000015562],
                            [-0.201194093997435, 0.198431485327112],
                            [0.0, 0.202578241925561],
                            [0.201194093997435, 0.198431485327112],
                            [0.394151347077563, 0.186161000015562],
                            [0.570972172608539, 0.166269205816993],
                            [0.724417731360170, 0.139570677926155],
                            [0.848206583410427, 0.107159220467172],
                            [0.937273392400706, 0.0703660474881081],
                            [0.987992518020485, 0.0307532419961174]],dtype=dtype,device=device)
    elif N == 16:
        coord = torch.tensor([[-0.989400934991650, 0.0271524594117540],
                            [-0.944575023073233, 0.0622535239386481],
                            [-0.865631202387832, 0.0951585116824914],
                            [-0.755404408355003, 0.124628971255535],
                            [-0.617876244402644, 0.149595988816578],
                            [-0.458016777657227, 0.169156519395002],
                            [-0.281603550779259, 0.182603415044923],
                            [-0.0950125098376374, 0.189450610455069],
                            [0.0950125098376374, 0.189450610455069],
                            [0.281603550779259, 0.182603415044923],
                            [0.458016777657227, 0.169156519395002],
                            [0.617876244402644, 0.149595988816578],
                            [0.755404408355003, 0.124628971255535],
                            [0.865631202387832, 0.0951585116824914],
                            [0.944575023073233, 0.0622535239386481],
                            [0.989400934991650, 0.0271524594117540]],dtype=dtype,device=device)
    else:
        raise ValueError('This quadrature scheme is not implemented now!')
    return coord[:,0], coord[:,1]


def composite_quadrature_1d(N, a, b, M,dtype=torch.double, device='cpu'):
    """
    Quadrature points and quadrature weights for one-dimensional Gauss-Legendre quadrature rules,
    mesh domain [a,b] into M equal subintervels and use N quadrature points in each subinterval.

    Parameters:
        N: number of quadrature points in each subintervals
        a,b: computational domain [a,b]
        M: number of subintervals of [a,b] meshed to
        dtype,device
    Returns:
        X: quadrature points size([N*M])
        W: quadrature weights size([N*M])
    """
    h = (b-a)/M
    x, w = quadrature_1d(N,dtype,device)
    x = ((x+1)/2).repeat(M)*h+torch.linspace(a,b,M+1,dtype=dtype, device=device)[:-1].repeat_interleave(N)
    w = (w/2).repeat(M)*h
    return x, w


def composite_quadrature_2d(N,a1,b1,a2,b2,M1,M2,dtype=torch.double,device='cpu'):
    """
    Quadrature points and quadrature weights for two-dimensional tensor Gauss-Legendre quadrature rules,
    for [a1,b1]*[a2,b2], mesh domain [a1,b1] and [a2,b2] into M1 and M2 equal subintervels respectively,
    and use N quadrature points in one-dimensional subintervals to get tensor quadrature rules.

    Parameters:
        N: number of quadrature points in one dimension
        a1,b1,a2,b2: computational domain [a1,b1]*[a2,b2]
        M1: number of subintervals of [a1,b1] meshed to
        M2: number of subintervals of [a2,b2] meshed to
    Returns:
        X: quadrature points size([N^2*M1*M2,2])
        W: quadrature weights size([N^2*M1*M2])
    """
    p1, w1 = composite_quadrature_1d(N,a1,b1,M1,dtype=dtype,device=device)
    p2, w2 = composite_quadrature_1d(N,a2,b2,M2,dtype=dtype,device=device)
    p1 = p1.repeat(N*M2)
    p2 = p2.repeat_interleave(N*M1)
    w1 = w1.repeat(N*M2)
    w2 = w2.repeat_interleave(N*M1)
    return torch.stack((p2,p1),dim=1), w2*w1

def composite_quadrature_1d_new(N, list_node, M, dtype=torch.double, device='cpu'):
    # x, w = composite_quadrature_1d_new(quad, [-1, -0.1, 0, 0.1, 1], n, dtype=dtype, device=device)
    X = torch.tensor([],dtype=dtype, device=device)
    W = torch.tensor([],dtype=dtype, device=device)
    for i in range(len(list_node)-1):
        a = list_node[i]
        b = list_node[i+1]
        h = (b-a)/M
        x, w = quadrature_1d(N,dtype=dtype, device=device)
        x = ((x+1)/2).repeat(M)*h+torch.linspace(a,b,M+1,dtype=dtype, device=device)[:-1].repeat_interleave(N)
        w = (w/2).repeat(M)*h
        X = torch.cat((X, x))
        W = torch.cat((W, w))
    return X, W


def composite_quadrature_1d_new1(list_node, list_N, list_M, dtype=torch.double, device='cpu'):
    # x, w = composite_quadrature_1d_new1([-1, -0.1, 0, 0.1, 1], [8, 16, 10, 8], [10, 20, 40, 100], dtype=dtype, device=device)
    X = torch.tensor([],dtype=dtype, device=device)
    W = torch.tensor([],dtype=dtype, device=device)
    for i in range(len(list_node)-1):
        a = list_node[i]
        b = list_node[i+1]
        N = list_N[i]
        M = list_M[i]
        h = (b-a)/M
        x, w = quadrature_1d(N,dtype=dtype, device=device)
        x = ((x+1)/2).repeat(M)*h+torch.linspace(a,b,M+1,dtype=dtype, device=device)[:-1].repeat_interleave(N)
        w = (w/2).repeat(M)*h
        X = torch.cat((X, x))
        W = torch.cat((W, w))
    return X, W

def composite_quadrature_1d_new2(list_node, list_N_M, dtype=torch.double, device='cpu'):
    # x, w = composite_quadrature_1d_new2([-1, -0.1, 0, 0.1, 1], [[8, 10], [16, 20], [10, 40], [8, 100]], dtype=dtype, device=device)
    X = torch.tensor([],dtype=dtype, device=device)
    W = torch.tensor([],dtype=dtype, device=device)
    for i in range(len(list_node)-1):
        a = list_node[i]
        b = list_node[i+1]
        N = list_N_M[i][0]
        M = list_N_M[i][1]
        h = (b-a)/M
        x, w = quadrature_1d(N,dtype=dtype, device=device)
        x = ((x+1)/2).repeat(M)*h+torch.linspace(a,b,M+1,dtype=dtype, device=device)[:-1].repeat_interleave(N)
        w = (w/2).repeat(M)*h
        X = torch.cat((X, x))
        W = torch.cat((W, w))
    return X, W

def quadrature_1d_Lobatto(N, dtype=torch.double, device='cpu'):
    """
    Quadrature points and weights for one-dimensional Gauss-Legendre quadrature rules in computational domain [-1,1].
    
    Parameters:
        N: number of quadrature points in domain [1-,1]
        dtype, device
    Returns:
        X: quadrature points size([N])
        W: quadrature weights size([N])
    """
    if N == 3:
        coord = torch.tensor([[-1.000000000000000, 1/3],
                              [0.0, 4/3],
                              [1.000000000000000, 1/3]],dtype=dtype,device=device)
    elif N == 4:
        coord = torch.tensor([[-1.000000000000000, 1/6],
                              [-0.447213595499958, 5/6],
                              [0.447213595499958, 5/6],
                              [1.000000000000000, 1/6]],dtype=dtype,device=device)
    elif N == 5:
        coord = torch.tensor([[-1.000000000000000, 0.1],
                              [-0.654653670707977, 0.544444444444444],
                              [0.0, 0.711111111111111],
                              [0.654653670707977, 0.544444444444444],
                              [1.000000000000000, 0.1]],dtype=dtype,device=device)
    elif N == 6:
        coord = torch.tensor([[-1.000000000000000, 0.066666666666667],
                              [-0.765055323929465, 0.378474956297847],
                              [-0.285231516480645, 0.554858377035486],
                              [0.285231516480645, 0.554858377035486],
                              [0.765055323929465, 0.378474956297847],
                              [1.000000000000000, 0.066666666666667]],dtype=dtype,device=device)
    elif N == 7:
        coord = torch.tensor([[-1.000000000000000, 0.047619047619048],
                              [-0.830223896278567, 0.276826047361566],
                              [-0.468848793470714, 0.431745381209863],
                              [0.0, 0.487619047619048],
                              [0.468848793470714, 0.431745381209863],
                              [0.830223896278567, 0.276826047361566],
                              [1.000000000000000, 0.047619047619048]],dtype=dtype,device=device)
    elif N == 8:
        coord = torch.tensor([[-1.000000000000000, 0.035714285714286],
                              [-0.871740148509607, 0.210704227143506],
                              [-0.591700181433142, 0.341122692483504],
                              [-0.209299217902479, 0.412458794658704],
                              [0.209299217902479, 0.412458794658704],
                              [0.591700181433142, 0.341122692483504],
                              [0.871740148509607, 0.210704227143506],
                              [1.000000000000000, 0.035714285714286]],dtype=dtype,device=device)
    elif N == 9:
        coord = torch.tensor([[-1.000000000000000, 0.027777777777778],
                            [-0.899757995411460, 0.165495361560806],
                            [-0.677186279510738, 0.274538712500162],
                            [-0.363117463826178, 0.346428510973046],
                            [0.0, 0.371519274376417],                   
                            [0.363117463826178, 0.346428510973046],
                            [0.677186279510738, 0.274538712500162],
                            [0.899757995411460, 0.165495361560806],
                            [1.000000000000000, 0.027777777777778]],dtype=dtype,device=device)
    elif N == 10:
        coord = torch.tensor([[-1.000000000000000, 0.022222222222222],
                            [-0.919533908166459, 0.133305990851070],
                            [-0.738773865105505, 0.224889342063127],
                            [-0.477924949810444, 0.292042683679684],
                            [-0.165278957666387, 0.327539761183897],
                            [0.165278957666387, 0.327539761183897],
                            [0.477924949810444, 0.292042683679684],
                            [0.738773865105505, 0.224889342063127],
                            [0.919533908166459, 0.133305990851070],
                            [1.000000000000000, 0.022222222222222]],dtype=dtype,device=device)
    elif N == 11:
        coord = torch.tensor([[-1.000000000000000, 0.018181818181818],
                            [-0.934001430408059, 0.109612273266995],
                            [-0.784483473663144, 0.187169881780305],
                            [-0.565235326996205, 0.248048104264028],
                            [-0.295758135586939, 0.286879124779008],
                            [0.0, 0.300217595455691],                       
                            [0.295758135586939, 0.286879124779008],
                            [0.565235326996205, 0.248048104264028],
                            [0.784483473663144, 0.187169881780305],
                            [0.934001430408059, 0.109612273266995],
                            [1.000000000000000, 0.018181818181818]],dtype=dtype,device=device)
    elif N == 12:
        coord = torch.tensor([[-1.000000000000000, 0.015151515151515],
                            [-0.944899272222882, 0.091684517413196],
                            [-0.819279321644007, 0.157974705564370],
                            [-0.632876153031861, 0.212508417761021],
                            [-0.399530940965349, 0.251275603199201],
                            [-0.136552932854928, 0.271405240910696],
                            [0.136552932854928, 0.271405240910696],
                            [0.399530940965349, 0.251275603199201],
                            [0.632876153031861, 0.212508417761021],
                            [0.819279321644007, 0.157974705564370],
                            [0.944899272222882, 0.091684517413196],
                            [1.000000000000000, 0.015151515151515]],dtype=dtype,device=device)
    elif N == 13:
        coord = torch.tensor([[-1.000000000000000, 0.012820512820513],
                              [-0.953309846642164, 0.077801686746819],
                              [-0.846347564651872, 0.134981926689608],
                              [-0.686188469081757, 0.183646865203550],
                              [-0.482909821091336, 0.220767793566110],
                              [-0.249286930106240, 0.244015790306676],
                              [0.0,    0.251930849333447],
                              [0.249286930106240, 0.244015790306676],
                              [0.482909821091336, 0.220767793566110],
                              [0.686188469081757, 0.183646865203550],
                              [0.846347564651872, 0.134981926689608],
                              [0.953309846642164, 0.077801686746819],
                              [1.000000000000000, 0.012820512820513]],dtype=dtype,device=device)
    elif N == 14:
        coord = torch.tensor([[-1.000000000000000, 0.010989010989011],
                              [-0.959935045267261, 0.066837284497681],
                              [-0.867801053830347, 0.11658665589871],
                              [-0.728868599091326, 0.160021851762952],
                              [-0.550639402928647, 0.194826149373416],
                              [-0.342724013342713, 0.219126253009771],
                              [-0.116331868883704, 0.231612794468457],
                              [0.116331868883704, 0.231612794468457],
                              [0.342724013342713, 0.219126253009771],
                              [0.550639402928647, 0.194826149373416],
                              [0.728868599091326, 0.160021851762952],
                              [0.867801053830347, 0.116586655898711],
                              [0.959935045267261, 0.066837284497681],
                              [1.000000000000000, 0.010989010989011]],dtype=dtype,device=device)
    elif N == 15:
        coord = torch.tensor([[-1.000000000000000, 0.009523809523810],
                              [-0.965245926503839, 0.058029893028601],
                              [-0.885082044222976, 0.101660070325718],
                              [-0.763519689951815, 0.140511699802428],
                              [-0.606253205469846, 0.172789647253601],
                              [-0.420638054713672, 0.186161000015562],
                              [-0.215353955363794, 0.211973585926821],
                              [0.0, 0.217048116348816],
                              [0.215353955363794, 0.211973585926821],
                              [0.420638054713672, 0.186161000015562],
                              [0.606253205469846, 0.172789647253601],
                              [0.763519689951815, 0.140511699802428],
                              [0.885082044222976, 0.101660070325718],
                              [0.965245926503839, 0.058029893028601],
                              [1.000000000000000, 0.009523809523810]],dtype=dtype,device=device)
    elif N == 16:
        coord = torch.tensor([[-1.000000000000000, 0.008333333333333],
                              [-0.969568046270218, 0.050850361005920],
                              [-0.899200533093472, 0.089393697325931],
                              [-0.792008291861815, 0.124255382132514],
                              [-0.652388702882493, 0.154026980807164],
                              [-0.486059421887138, 0.177491913391704],
                              [-0.299830468900763, 0.193690023825204],
                              [-0.101326273521949, 0.201958308178230],
                              [0.101326273521949, 0.201958308178230],
                              [0.299830468900763, 0.193690023825204],
                              [0.486059421887138, 0.177491913391704],
                              [0.652388702882493, 0.154026980807164],
                              [0.792008291861815, 0.124255382132514],
                              [0.899200533093472, 0.089393697325931],
                              [0.969568046270218, 0.050850361005920],
                              [1.000000000000000, 0.008333333333333]],dtype=dtype,device=device)
    else:
        raise ValueError('This quadrature scheme is not implemented now!')
    return coord[:,0], coord[:,1]


def composite_quadrature_1d_Lobatto(N, a, b, M,dtype=torch.double, device='cpu'):
    """
    Quadrature points and quadrature weights for one-dimensional Gauss-Legendre-Lobatto quadrature rules,
    mesh domain [a,b] into M equal subintervels and use N quadrature points in each subinterval.

    Parameters:
        N: number of quadrature points in each subintervals
        a,b: computational domain [a,b]
        M: number of subintervals of [a,b] meshed to
        dtype,device
    Returns:
        X: quadrature points size([N*M])
        W: quadrature weights size([N*M])
    """
    h = (b-a)/M
    x, w = quadrature_1d_Lobatto(N,dtype=dtype, device=device)
    x = ((x+1)/2).repeat(M)*h+torch.linspace(a,b,M+1,dtype=dtype, device=device)[:-1].repeat_interleave(N)
    w = (w/2).repeat(M)*h
    return x, w


def gauss_jacobi_1d(N, alpha, beta, dtype=torch.double, device='cpu'):
    """
    Calculate the nodes and weights of the Gauss-Jacobi integral
    # N: Number of nodes
    # alpha, beta: Parameters of the weight function
    return
    x,quadrature points on [-1,1]
    w, quadrature weights on [-1,1]
    """
    x, w = roots_jacobi(N, alpha, beta)
    x, w = torch.tensor(x,dtype=dtype,device=device),torch.tensor(w,dtype=dtype,device=device)
    return x, w

def gauss_jacobi_quadrature_ab(alpha, beta, N, a, b, dtype=torch.double, device='cpu'):
    """
    Quadrature points and quadrature weights for Gauss-Jacobi quadrature rules on [a, b]. 
    weight function : (b - x)^{\alpha} (x - a)^{\beta}

    Parameters:
        N: number of quadrature points
        a,b: computational domain [a,b]
        dtype,device
    Returns:
        t: quadrature points size([N]) on [a,b]
        v: quadrature weights size([N]) on [a,b]
    """
    # x1, w1 = gauss_jacobi_quadrature_ab(0, -1/2, 160, a, b)
    # Get the Gauss-Jacobi quadrature nodes and weights on the standard interval [-1, 1]
    x, w = gauss_jacobi_1d(N, alpha, beta, dtype, device)

    # Transform nodes from [-1, 1] to [a, b]
    x = ((b - a) * x + (a + b)) / 2

    # Adjust weights for the interval [a, b]
    w =  ((b - a) / 2) **( alpha + beta + 1) * w

    return x, w

def periodic_2pi_quadrature(N, dtype=torch.double, device='cpu'):
    """
    Quadrature points and quadrature weights for 2pi-periodic function on [0, 2 * pi]. Exact for V := span{e^{ikx}: 0 \leq |k| \leq N - 1}

    Parameters:
        N: number of quadrature points
    Returns:
        x: [N] on [0, 2 * pi]
        w: [N]
    """
    x = 2 * torch.pi * torch.linspace(0, N - 1, N, dtype=dtype, device=device) / N
    w = 2 * torch.pi * torch.ones(N, dtype=dtype, device=device) / N

    return x, w

def periodic_0_1_quadrature(N, dtype=torch.double, device='cpu'):
    """
    Quadrature points and quadrature weights for 2pi-periodic function on [0, 2 * pi]. Exact for V := span{e^{ikx}: 0 \leq |k| \leq N - 1}

    Parameters:
        N: number of quadrature points
    Returns:
        x: [N] on [0, 2 * pi]
        w: [N]
    """
    x = torch.linspace(0, N - 1, N, dtype=dtype, device=device) / N
    w = torch.ones(N, dtype=dtype, device=device) / N

    return x, w

# ******************** Bounded Domain ********************







# ******************** Unbounded Domain ********************
# Hermite-Gasuss Rule
def Hermite_Gauss_Quad(k,dtype=torch.double,device='cpu', modified=True):
    x, w = hermgauss(k)
    if modified:
        w = w*np.exp(x**2)
    return torch.from_numpy(x).to(dtype).to(device), torch.from_numpy(w).to(dtype).to(device)


# Laguerre-Gauss Rule
def Laguerre_Gauss_Quad(k,dtype=torch.double,device='cpu', modified=True):
    x, w = laggauss(k)
    if modified:
        w = w*np.exp(x)
    return torch.from_numpy(x).to(dtype).to(device), torch.from_numpy(w).to(dtype).to(device)


# 2D Tensor Laguerre-Gauss Rule
def Laguerre_Gauss_Quad2D(N,dtype=torch.double,device='cpu', modified=True):
    x, w = laggauss(N)
    if modified:
        w = w*np.exp(x)
    x = torch.from_numpy(x).to(dtype).to(device)
    w = torch.from_numpy(w).to(dtype).to(device)
    p1 = x.repeat_interleave(N)
    p2 = x.repeat(N)
    return torch.stack((p1,p2),dim=1), torch.outer(w,w).view(-1)

# Laguerre-Gauss-Radau rule
def Laguerre_Gauss_Radau(dtype=torch.double,device='cpu'):
    coord = torch.tensor([[0.000000000000000e+00, 5.882352941176471e-02],
            [2.161403052394536e-01, 2.927604493268249e-01],
            [7.263882432518047e-01, 3.181362909815314e-01],
            [1.533593160373541e+00, 2.066607692008763e-01],
            [2.644970998611911e+00, 8.994208934619415e-02],
            [4.070978160880192e+00, 2.708753007297033e-02],
            [5.825855515105604e+00, 5.679781839868921e-03],
            [7.928504185306668e+00, 8.230703112809221e-04],
            [1.040380828995104e+01, 8.100028502120838e-05],
            [1.328466107070703e+01, 5.266367668946434e-06],
            [1.661517321686662e+01, 2.173968333033556e-07],
            [2.045600602002722e+01, 5.384928271901132e-09],
            [2.489384702535191e+01, 7.374041106876865e-11],
            [3.005986292020259e+01, 4.928461815736925e-13],
            [3.617069454367918e+01, 1.309028045707531e-15],
            [4.364036518417683e+01, 9.358643168465394e-19],
            [5.352915116026845e+01, 6.770058713668848e-23]],dtype=dtype,device=device)
    return coord[:,0], coord[:,1]


def Laguerre_Gauss_Radau2D(N=1,dtype=torch.double,device='cpu'):
    x, w = Laguerre_Gauss_Radau(dtype=dtype,device=device)

    x = torch.from_numpy(x).to(dtype).to(device)
    w = torch.from_numpy(w).to(dtype).to(device)
    p1 = x.repeat_interleave(N)
    p2 = x.repeat(N)
    return torch.stack((p1,p2),dim=1), torch.outer(w,w).view(-1)




def main():
    # precision of prints
    torch.set_printoptions(precision=20)

    k = 3
    a = -5
    b = 5
    n = 3

    # test 1-dimensional composite quadrature rules
    # x, w = composite_quadrature_1d(k,a,b,n)
    # print(x)
    # print(w)

    # test 2-dimensional composite quadrature rules
    # x2d, w2d = composite_quadrature_2d(k,a,b,a,b,n,n)
    # print(x2d)
    # print(w2d)

    x, w = Laguerre_Gauss_Quad(3)


if __name__ == '__main__':
    main()
