import torch.nn as nn
import torch
import numpy as np

from quadrature import *

seed = 1234
torch.manual_seed(seed=seed)
torch.cuda.manual_seed_all(seed=seed)

# ********** Activation functions with Derivative **********
# Redefine activation functions and the corresponding local gradient

# tanh(x)
class TNN_Tanh(nn.Module):
    """Tnn_Tanh"""
    def forward(self,x):
        return torch.tanh(x)

    def grad(self,x):
        return 1-torch.tanh(x)**2
    
    def grad_grad(self, x):
        return 2 * torch.tanh(x)**3 - 2 * torch.tanh(x)

# sigmoid(x)
class TNN_Sigmoid(nn.Module):
    """TNN_Sigmoid"""
    def forward(self,x):
        return torch.sigmoid(x)

    def grad(self,x):
        return torch.sigmoid(x)*(1-torch.sigmoid(x))

# sin(x)
class TNN_Sin(nn.Module):
    """TNN_Sin"""
    def forward(self,x):
        return torch.sin(x)

    def grad(self,x):
        return torch.cos(x)

    def grad_grad(self,x):
        return -torch.sin(x)
    
    def grad_grad_grad(self,x):
        return -torch.cos(x)

# cos(x)
class TNN_Cos(nn.Module):
    """for TNN_Sin"""
    def forward(self,x):
        return torch.cos(x)

    def grad(self,x):
        return -torch.sin(x)

    def grad_grad(self,x):
        return -torch.cos(x)
    
    def grad_grad_grad(self,x):
        return torch.sin(x)


# ReQU(x)=
#         x^2, x\geq0,
#         0,   x<0.
class TNN_ReQU(nn.Module):
    """docstring for TNN_ReQU"""
    def forward(self,x):
        return x*torch.relu(x)

    def grad(self,x):
        return 2*torch.relu(x)
        


# ********** Network layers **********
# Linear layer for TNN
class TNN_Linear(nn.Module):
    """
    Applies a batch linear transformation to the incoming data:
        input data: x:[dim, n1, N]
        learnable parameters: W:[dim,n2,n1], b:[dim,n2,1]
        output data: y=Wx+b:[dim,n2,N]

    Parameters:
        dim: dimension of TNN
        out_features: n2
        in_features: n1
        bias: if bias needed or not (boolean)
    """
    def __init__(self, dim, out_features, in_features, bias):
        super(TNN_Linear, self).__init__()
        self.dim = dim
        self.out_features = out_features
        self.in_features = in_features

        self.weight = nn.Parameter(torch.empty((self.dim, self.out_features, self.in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty((self.dim, self.out_features, 1)))
        else:
            self.bias = None

    def forward(self,x):
        if self.bias==None:
            if self.in_features==1:
                return self.weight*x
            else:
                return self.weight@x
        else:
            if self.in_features==1:
                return self.weight*x+self.bias
            else:
                return self.weight@x+self.bias

    def extra_repr(self):
        return 'weight.size={}, bias.size={}'.format(
            [self.dim, self.out_features, self.in_features], [self.dim, self.out_features, 1] if self.bias!=None else []
        )


# Scaling layer for TNN.
class TNN_Scaling(nn.Module):
    """
    Define the scaling parameters.

    size:
        [k,p] for Multi-TNN
        [p] for TNN
    """
    def __init__(self, size):
        super(TNN_Scaling, self).__init__()
        self.size = size
        self.alpha = nn.Parameter(torch.empty(self.size))

    def extra_repr(self):
        return 'size={}'.format(self.size)


# Define extra parameters
class TNN_Extra(nn.Module):
    """
    Define extra parameters.
    """
    def __init__(self, size):
        super(TNN_Extra, self).__init__()
        self.size = size
        self.beta = nn.Parameter(torch.empty(self.size))
        
    def extra_repr(self):
        return 'size={}'.format(self.size)


# ********** TNN architectures **********
# One simple TNN
class TNN(nn.Module):
    """
    Architectures of the simple tensor neural network.
    FNN on each demension has the same size,
    and the input integration points are same in different dinension. 
    TNN values and gradient values at data points are provided.

    Parameters:
        dim: dimension of TNN, number of FNNs
        size: [1, n0, n1, ..., nl, p], size of each FNN
        activation: activation function used in hidden layers
        bd: extra function for boundary condition
        grad_bd: gradient of bd
        bd_rank: array of indices specifying which ranks (from total p) to operate bd on 
        initializer: initial method for learnable parameters
    """
    def __init__(self, dim, size, activation, bd=None, grad_bd=None, grad_grad_bd=None, grad_grad_grad_bd=None, bd_rank=None, scaling=True, extra_size=False, initializer=['default', None]):
        super(TNN, self).__init__()
        self.dim = dim
        self.size = size
        self.activation = activation()
        self.bd = bd
        self.grad_bd = grad_bd
        self.grad_grad_bd = grad_grad_bd
        self.grad_grad_grad_bd = grad_grad_grad_bd
        self.bd_rank = bd_rank
        self.scaling = scaling
        self.extra_size = extra_size

        self.p = abs(self.size[-1])

        self.init_type = initializer[0]
        self.init_method = initializer[1]

        self.ms = self.__init_modules()
        self.__initialize()

    # Register learnable parameters of TNN module.
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(1, len(self.size)):
            bias = True if self.size[i] > 0 else False
            modules['TNN_Linear{}'.format(i-1)] = TNN_Linear(self.dim,abs(self.size[i]),abs(self.size[i-1]),bias)
        if self.scaling:
            modules['TNN_Scaling'] = TNN_Scaling([self.p])
        if self.extra_size:
            modules['TNN_Extra'] = TNN_Extra(self.extra_size)
        return modules

    # Initialize learnable parameters of TNN module.
    def __initialize(self):
        # default initialization.
        if self.init_type == 'default':
            for i in range(1, len(self.size)):
                for j in range(self.dim):
                    # nn.init.orthogonal_(self.ms['TNN_Linear{}'.format(i-1)].weight[j,:,:]) # 初始化为正交矩阵
                    nn.init.kaiming_normal_(self.ms['TNN_Linear{}'.format(i-1)].weight[j,:,:]) # 缩放的正态分布
                    # nn.init.normal_(self.ms['TNN_Linear{}'.format(i-1)].weight[j,:,:]) # 从标准正态分布中取值
                if self.size[i] > 0:
                    # nn.init.constant_(self.ms['TNN_Linear{}'.format(i-1)].bias, 0)
                    # nn.init.normal_(self.ms['TNN_Linear{}'.format(i-1)].bias)
                    nn.init.kaiming_normal_(self.ms['TNN_Linear{}'.format(i-1)].bias)
                    # nn.init.orthogonal_(self.ms['TNN_Linear{}'.format(i-1)].bias)
            if self.scaling:
                nn.init.constant_(self.ms['TNN_Scaling'].alpha, 1)
            if self.extra_size:
                nn.init.constant_(self.ms['TNN_Extra'].beta, 1)
        
        if self.init_type == 'customized':
            # model = TNN(dim, sizes, activation, initializer=['customized', ['orthogonal', 'orthogonal']]).to(dtype).to(device)
            weight_type = self.init_method[0]
            bias_type = self.init_method[1]

            if weight_type == 'orthogonal':
                weight_initializer = torch.nn.init.orthogonal_
            elif weight_type == 'normal':
                weight_initializer = torch.nn.init.normal_
            elif weight_type == 'kaiming_normal':
                weight_initializer = torch.nn.init.kaiming_normal_
            elif weight_type == 'xavier_normal':
                weight_initializer = torch.nn.init.xavier_normal_
            else:
                raise NotImplementedError
            
            if bias_type == 'orthogonal':
                bias_initializer = torch.nn.init.orthogonal_
            elif bias_type == 'normal':
                bias_initializer = torch.nn.init.normal_
            elif bias_type == 'kaiming_normal':
                bias_initializer = torch.nn.init.kaiming_normal_
            elif bias_type == 'xavier_normal':
                bias_initializer = torch.nn.init.xavier_normal_
            elif bias_type == 'zero':
                bias_initializer = torch.nn.init.zeros_
            else:
                raise NotImplementedError
            
            for i in range(1, len(self.size)):
                for j in range(self.dim):
                    weight_initializer(self.ms['TNN_Linear{}'.format(i-1)].weight[j,:,:])
                if self.size[i] > 0:
                    bias_initializer(self.ms['TNN_Linear{}'.format(i-1)].bias)

            # with torch.no_grad():
            #     for j in range(self.dim):
            #         self.ms['TNN_Linear{}'.format(0)].weight[j,:,:] = self.ms['TNN_Linear{}'.format(0)].weight[j,:,:] * 1

            if self.scaling:
                nn.init.constant_(self.ms['TNN_Scaling'].alpha, 1)
            if self.extra_size:
                nn.init.constant_(self.ms['TNN_Extra'].beta, 1)


    # function to return scaling parameters
    def scaling_par(self):
        if self.scaling:
            return self.ms['TNN_Scaling'].alpha
        else:
            raise NameError('The TNN Module does not have Scaling Parameters')

    # function to return extra parameters
    def extra_par(self):
        if self.extra_size:
            return self.ms['TNN_Extra'].beta
        else:
            raise NameError('The TNN Module does not have Extra Parameters')


    def forward(self,w,x,need_grad=0,normed=True):
        """
        Parameters:
            w: quadrature weights [N]
            x: quadrature points [N]
            need_grad: if return gradient or not
        
        Returns:
            phi: values of each dimensional FNN [dim, p, N]
            grad_phi: gradient values of each dimensional FNN [dim, p, N]
        """
        x = x.unsqueeze(dim=-2)
        w = w.unsqueeze(dim=-2)
        # Compute values of each one-dimensional input FNN at each quadrature point.
        if need_grad==0:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            if self.bd_rank==None:
                bd_rank = None
            else:
                bd_rank = self.bd_rank
            # Forward process.
            for i in range(1, len(self.size) - 1):
                x = self.ms['TNN_Linear{}'.format(i-1)](x)
                x = self.activation(x)
            if bd_value==None:
                phi = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
            else:
                if bd_rank==None:
                    phi = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)*bd_value
                else:
                    raw_phi = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
                    mask = torch.zeros(raw_phi.shape, dtype=torch.bool, device=raw_phi.device)
                    mask[:, bd_rank, :] = True
                    phi = torch.where(mask, raw_phi * bd_value, raw_phi)
            # normalization
            if normed:
                norm_phi = torch.sqrt(torch.sum(w * phi ** 2, dim=-1)).unsqueeze(dim=-1)
                return phi / norm_phi
            else:
                return phi


        # Compute values and gradient values of each one-dimensional input FNN at each quadrature point simutaneously.
        if need_grad==1:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            if self.bd_rank==None:
                bd_rank = None
            else:
                bd_rank = self.bd_rank
            # Get gradient values of forced boundary condition function.
            if self.grad_bd==None:
                grad_bd_value = None
            else:
                grad_bd_value = self.grad_bd(x)
            # Compute forward and backward process simutaneously.
            grad_x = self.ms['TNN_Linear{}'.format(0)].weight
            for i in range(1, len(self.size) - 1):
                x = self.ms['TNN_Linear{}'.format(i-1)](x)
                grad_x = self.activation.grad(x)*grad_x
                grad_x = self.ms['TNN_Linear{}'.format(i)].weight@grad_x
                x = self.activation(x)
            x = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
            if self.bd==None:
                phi = x
                grad_phi = grad_x
            else:
                if bd_rank==None:
                    phi = x*bd_value
                    grad_phi = x*grad_bd_value+grad_x*bd_value
                else:
                    mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
                    mask[:, bd_rank, :] = True
                    phi = torch.where(mask, x*bd_value, x)
                    grad_phi = torch.where(mask, x*grad_bd_value+grad_x*bd_value, grad_x)
            # normalization
            if normed:
                norm_phi = torch.sqrt(torch.sum(w * phi ** 2, dim=-1)).unsqueeze(dim=-1)
                return phi / norm_phi, grad_phi / norm_phi
            else:
                return phi, grad_phi


        # Compute values and gradient values of each one-dimensional input FNN at each quadrature point simutaneously.
        if need_grad==2:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            if self.bd_rank==None:
                bd_rank = None
            else:
                bd_rank = self.bd_rank
            # Get gradient values of forced boundary condition function.
            if self.grad_bd==None:
                grad_bd_value = None
            else:
                grad_bd_value = self.grad_bd(x)
            # get grad_grad_bd values
            if self.grad_grad_bd==None:
                grad_grad_bd_value = None
            else:
                grad_grad_bd_value = self.grad_grad_bd(x)

            # Compute forward and backward process simutaneously.
            grad_x = self.ms['TNN_Linear{}'.format(0)].weight
            grad_grad_x = torch.zeros_like(grad_x)
            
            for i in range(1, len(self.size) - 1):
                x = self.ms['TNN_Linear{}'.format(i-1)](x)
                grad_grad_x = self.activation.grad_grad(x)*(grad_x**2)+self.activation.grad(x)*grad_grad_x
                grad_grad_x = self.ms['TNN_Linear{}'.format(i)].weight@grad_grad_x
                grad_x = self.activation.grad(x)*grad_x
                grad_x = self.ms['TNN_Linear{}'.format(i)].weight@grad_x
                x = self.activation(x)
            x = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)

            if self.bd==None:
                phi = x
                grad_phi = grad_x
                grad_grad_phi = grad_grad_x
            else:
                if bd_rank==None:
                    phi = x*bd_value
                    grad_phi = x*grad_bd_value+grad_x*bd_value
                    grad_grad_phi = x*grad_grad_bd_value + 2*grad_x*grad_bd_value + grad_grad_x*bd_value
                else:
                    mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
                    mask[:, bd_rank, :] = True
                    phi = torch.where(mask, x*bd_value, x)
                    grad_phi = torch.where(mask, x*grad_bd_value+grad_x*bd_value, grad_x)
                    grad_grad_phi = torch.where(mask, x*grad_grad_bd_value + 2*grad_x*grad_bd_value + grad_grad_x*bd_value, grad_grad_x)
            # normalization
            if normed:
                norm_phi = torch.sqrt(torch.sum(w * phi ** 2, dim=-1)).unsqueeze(dim=-1)
                return phi / norm_phi, grad_phi / norm_phi, grad_grad_phi / norm_phi
                
            else:
                return phi, grad_phi, grad_grad_phi

        if need_grad==3:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            if self.bd_rank==None:
                bd_rank = None
            else:
                bd_rank = self.bd_rank
            if self.grad_bd==None:
                grad_bd_value = None
            else:
                grad_bd_value = self.grad_bd(x)
            if self.grad_grad_bd==None:
                grad_grad_bd_value = None
            else:
                grad_grad_bd_value = self.grad_grad_bd(x)
            if self.grad_grad_grad_bd==None:
                grad_grad_grad_bd_value = None
            else:
                grad_grad_grad_bd_value = self.grad_grad_grad_bd(x)
            # Compute forward and backward process simutaneously.
            grad_x = self.ms['TNN_Linear{}'.format(0)].weight
            grad_grad_x = torch.zeros_like(grad_x)
            grad_grad_grad_x = torch.zeros_like(grad_x)
            
            for i in range(1, len(self.size) - 1):
                x = self.ms['TNN_Linear{}'.format(i-1)](x)
                grad_grad_grad_x = self.activation.grad_grad_grad(x)*(grad_x**3)+3*self.activation.grad_grad(x)*grad_grad_x*grad_x+self.activation.grad(x)*grad_grad_grad_x
                grad_grad_grad_x = self.ms['TNN_Linear{}'.format(i)].weight@grad_grad_grad_x
                grad_grad_x = self.activation.grad_grad(x)*(grad_x**2)+self.activation.grad(x)*grad_grad_x
                grad_grad_x = self.ms['TNN_Linear{}'.format(i)].weight@grad_grad_x
                grad_x = self.activation.grad(x)*grad_x
                grad_x = self.ms['TNN_Linear{}'.format(i)].weight@grad_x
                x = self.activation(x)
            x = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)

            if self.bd==None:
                phi = x
                grad_phi = grad_x
                grad_grad_phi = grad_grad_x
                grad_grad_grad_phi = grad_grad_grad_x
            else:   
                if bd_rank==None:
                    phi = x*bd_value
                    grad_phi = x*grad_bd_value+grad_x*bd_value
                    grad_grad_phi = x*grad_grad_bd_value + 2*grad_x*grad_bd_value + grad_grad_x*bd_value
                    grad_grad_grad_phi = x*grad_grad_grad_bd_value + 3*grad_grad_x*grad_bd_value + 3*grad_x*grad_grad_bd_value + grad_grad_grad_x*bd_value
                else:
                    mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
                    mask[:, bd_rank, :] = True
                    phi = torch.where(mask, x*bd_value, x)
                    grad_phi = torch.where(mask, x*grad_bd_value+grad_x*bd_value, grad_x)
                    grad_grad_phi = torch.where(mask, x*grad_grad_bd_value + 2*grad_x*grad_bd_value + grad_grad_x*bd_value, grad_grad_x)
                    grad_grad_grad_phi = torch.where(mask, x*grad_grad_grad_bd_value + 3*grad_grad_x*grad_bd_value + 3*grad_x*grad_grad_bd_value + grad_grad_grad_x*bd_value, grad_grad_grad_x)
            # normalization
            if normed:
                norm_phi = torch.sqrt(torch.sum(w * phi ** 2, dim=-1)).unsqueeze(dim=-1)
                return phi / norm_phi, grad_phi / norm_phi, grad_grad_phi / norm_phi, grad_grad_grad_phi / norm_phi
            else:
                return phi, grad_phi, grad_grad_phi, grad_grad_grad_phi
                
    def extra_repr(self):
        return '{}\n{}'.format('Architectures of one TNN(dim={},rank={}) which has {} FNNs:'.format(self.dim,self.p,self.dim),\
                'Each FNN has size: {}'.format(self.size))

class TNN_ResNet(nn.Module):
    """
    Architectures of the simple tensor neural network.
    FNN on each demension has the same size,
    and the input integration points are same in different dinension. 
    TNN values and gradient values at data points are provided.

    Parameters:
        dim: dimension of TNN, number of FNNs
        size: [1, n0, n1, ..., nl, p], size of each FNN
        activation: activation function used in hidden layers
        bd: extra function for boundary condition
        grad_bd: gradient of bd
        initializer: initial method for learnable parameters
    """
    def __init__(self, dim, size, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=True, frequency=None, extra_size=False, initializer=['default',None]):
        super(TNN_ResNet, self).__init__()
        self.dim = dim
        self.size = size
        self.activation = activation()
        self.bd = bd
        self.grad_bd = grad_bd
        self.grad_grad_bd = grad_grad_bd
        self.scaling = scaling
        self.frequency = frequency
        self.extra_size = extra_size

        self.p = abs(self.size[-1])

        self.init_type = initializer[0]
        self.init_data = initializer[1]

        self.ms = self.__init_modules()
        self.__initialize()

    # Register learnable parameters of TNN module.
    def __init_modules(self):
        modules = nn.ModuleDict()
        # for i in range(1, len(self.size)):
        bias = True if self.size[-2] > 0 else False
        modules['TNN_Linear{}'.format(0)] = TNN_Linear(self.dim,abs(self.size[1]),abs(self.size[0]),bias)
        
        for i in range(1, len(self.size) - 2):
            bias = True if self.size[i] > 0 else False
            modules['TNN_Linear{}_{}'.format(i,0)] = TNN_Linear(self.dim,abs(self.size[i+1]),abs(self.size[i]),bias)
            modules['TNN_Linear{}_{}'.format(i,1)] = TNN_Linear(self.dim,abs(self.size[i+1]),abs(self.size[i]),bias)
        
        bias = True if self.size[-2] > 0 else False
        modules['TNN_Linear{}'.format(len(self.size) - 2)] = TNN_Linear(self.dim,abs(self.size[-1]),abs(self.size[-2]),bias)
        if self.scaling:
            modules['TNN_Scaling'] = TNN_Scaling([self.p])
        if self.extra_size:
            modules['TNN_Extra'] = TNN_Extra(self.extra_size)
        return modules

    # Initialize learnable parameters of TNN module.
    def __initialize(self):
        # default initialization.
        if self.init_type == 'default':
            # for i in range(1, len(self.size)):
            # nn.init.orthogonal_(self.ms['TNN_Linear{}'.format(0)].weight)
            nn.init.kaiming_normal_(self.ms['TNN_Linear{}'.format(0)].weight)
            if self.size[1] > 0:
                # nn.init.orthogonal_(self.ms['TNN_Linear{}'.format(0)].bias)
                nn.init.kaiming_normal_(self.ms['TNN_Linear{}'.format(0)].bias)

            if self.frequency:
                with torch.no_grad():
                    for j in range(self.dim):
                        # self.ms['TNN_Linear{}'.format(0)].weight[j,:,:] = self.ms['TNN_Linear{}'.format(0)].weight[j,:,:] * torch.arange(1, self.size[1] + 1, dtype=torch.float64).unsqueeze(dim=-1) / self.size[1] * self.frequency
                        self.ms['TNN_Linear{}'.format(0)].weight[j,:,:] = self.ms['TNN_Linear{}'.format(0)].weight[j,:,:] * self.frequency[j]
            
            for i in range(1, len(self.size) - 2):
                for j in range(self.dim):
                    # nn.init.orthogonal_(self.ms['TNN_Linear{}_{}'.format(i,0)].weight[j,:,:])
                    # nn.init.orthogonal_(self.ms['TNN_Linear{}_{}'.format(i,1)].weight[j,:,:])
                    nn.init.kaiming_normal_(self.ms['TNN_Linear{}_{}'.format(i,0)].weight[j,:,:])
                    nn.init.kaiming_normal_(self.ms['TNN_Linear{}_{}'.format(i,1)].weight[j,:,:])
                    # nn.init.normal_(self.ms['TNN_Linear{}'.format(i-1)].weight[j,:,:])
                if self.size[i+1] > 0:
                    # nn.init.orthogonal_(self.ms['TNN_Linear{}_{}'.format(i,0)].bias)
                    # nn.init.orthogonal_(self.ms['TNN_Linear{}_{}'.format(i,1)].bias)
                    nn.init.kaiming_normal_(self.ms['TNN_Linear{}_{}'.format(i,0)].bias)
                    nn.init.kaiming_normal_(self.ms['TNN_Linear{}_{}'.format(i,1)].bias)
                    # nn.init.constant_(self.ms['TNN_Linear{}'.format(i-1)].bias, 0.5)
            
            # nn.init.orthogonal_(self.ms['TNN_Linear{}'.format(len(self.size) - 2)].weight)
            nn.init.kaiming_normal_(self.ms['TNN_Linear{}'.format(len(self.size) - 2)].weight)
            if self.size[-1] > 0:
                # nn.init.orthogonal_(self.ms['TNN_Linear{}'.format(len(self.size) - 2)].bias)
                nn.init.kaiming_normal_(self.ms['TNN_Linear{}'.format(len(self.size) - 2)].bias)
            
            if self.scaling:
                nn.init.constant_(self.ms['TNN_Scaling'].alpha, 1)
            if self.extra_size:
                nn.init.constant_(self.ms['TNN_Extra'].beta, 1)
        

    # function to return scaling parameters
    def scaling_par(self):
        if self.scaling:
            return self.ms['TNN_Scaling'].alpha
        else:
            raise NameError('The TNN Module does not have Scaling Parameters')

    # function to return extra parameters
    def extra_par(self):
        if self.extra_size:
            return self.ms['TNN_Extra'].beta
        else:
            raise NameError('The TNN Module does not have Extra Parameters')


    def forward(self,w,x,need_grad=0,normed=True):
        """
        Parameters:
            w: quadrature weights [N]
            x: quadrature points [N]
            need_grad: if return gradient or not
        
        Returns:
            phi: values of each dimensional FNN [dim, p, N]
            grad_phi: gradient values of each dimensional FNN [dim, p, N]
        """
        x = x.unsqueeze(dim=-2)
        w = w.unsqueeze(dim=-2)
        # Compute values of each one-dimensional input FNN at each quadrature point.
        if need_grad==0:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            
            x = self.ms['TNN_Linear{}'.format(0)](x)
            # Forward process.
            for i in range(1, len(self.size) - 2):
                x0 = x
                x = self.ms['TNN_Linear{}_{}'.format(i,0)](x)
                x = self.activation(x)
                x = self.ms['TNN_Linear{}_{}'.format(i,1)](x)
                x = self.activation(x) + x0
                # x = x + x0
            # x = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)

            if bd_value==None:
                phi = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
            else:
                phi = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)*bd_value
            # normalization
            if normed:
                return phi / torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1)
            else:
                return phi


        # Compute values and gradient values of each one-dimensional input FNN at each quadrature point simutaneously.
        if need_grad==1:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            # Get gradient values of forced boundary condition function.
            if self.grad_bd==None:
                grad_bd_value = None
            else:
                grad_bd_value = self.grad_bd(x)
            # Compute forward and backward process simutaneously.
            # grad_x = self.ms['TNN_Linear{}_{}'.format(0,0)].weight
            # print(self.ms['TNN_Linear{}_{}'.format(0,0)].weight.size())
            grad_x = self.ms['TNN_Linear{}'.format(0)].weight
            x = self.ms['TNN_Linear{}'.format(0)](x)
            for i in range(1, len(self.size) - 2):
                x0 = x
                grad_x0 = grad_x
                
                grad_x = self.ms['TNN_Linear{}_{}'.format(i,0)].weight@grad_x
                # print(grad_x.size())
                x = self.ms['TNN_Linear{}_{}'.format(i,0)](x)
                grad_x = self.activation.grad(x)*grad_x
                # print(grad_x.size())
                x = self.activation(x)
                # print(self.ms['TNN_Linear{}_{}'.format(i-1,1)].weight.size())
                grad_x = self.ms['TNN_Linear{}_{}'.format(i,1)].weight@grad_x
                # print(grad_x.size())
                x = self.ms['TNN_Linear{}_{}'.format(i,1)](x)
                grad_x = self.activation.grad(x)*grad_x + grad_x0
                x = self.activation(x) + x0
            grad_x = self.ms['TNN_Linear{}'.format(len(self.size) - 2)].weight@grad_x
            x = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
            if self.bd==None:
                phi = x
                grad_phi = grad_x
            else:
                phi = x*bd_value
                grad_phi = x*grad_bd_value+grad_x*bd_value
            # normalization
            if normed:
                return phi / (torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1)), grad_phi / (torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1))
            else:
                return phi, grad_phi


        # Compute values and gradient values of each one-dimensional input FNN at each quadrature point simutaneously.
        if need_grad==2:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            # Get gradient values of forced boundary condition function.
            if self.grad_bd==None:
                grad_bd_value = None
            else:
                grad_bd_value = self.grad_bd(x)
            # get grad_grad_bd values
            if self.grad_grad_bd==None:
                grad_grad_bd_value = None
            else:
                grad_grad_bd_value = self.grad_grad_bd(x)

            # Compute forward and backward process simutaneously.
            x = self.ms['TNN_Linear{}'.format(0)](x)
            grad_x = self.ms['TNN_Linear{}'.format(0)].weight
            grad_grad_x = torch.zeros_like(grad_x)
            
            for i in range(1, len(self.size) - 2):
                x0 = x
                grad_x0 = grad_x
                grad_grad_x0 = grad_grad_x

                grad_x = self.ms['TNN_Linear{}_{}'.format(i,0)].weight@grad_x
                x = self.ms['TNN_Linear{}_{}'.format(i,0)](x)
                grad_grad_x = self.activation.grad_grad(x)*(grad_x**2)+self.activation.grad(x)*grad_grad_x
                # grad_grad_x = self.ms['TNN_Linear{}'.format(i)].weight@grad_grad_x
                grad_x = self.activation.grad(x)*grad_x
                x = self.activation(x)

                grad_x = self.ms['TNN_Linear{}_{}'.format(i,1)].weight@grad_x
                x = self.ms['TNN_Linear{}_{}'.format(i,1)](x)
                grad_grad_x = self.activation.grad_grad(x)*(grad_x**2)+self.activation.grad(x)*grad_grad_x + grad_grad_x0
                # grad_grad_x = self.ms['TNN_Linear{}'.format(i)].weight@grad_grad_x
                grad_x = self.activation.grad(x)*grad_x + grad_x0
                x = self.activation(x) + x0
            
            grad_x = self.ms['TNN_Linear{}'.format(len(self.size) - 2)].weight@grad_x
            x = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
            grad_grad_x = self.ms['TNN_Linear{}'.format(len(self.size) - 2)].weight@grad_grad_x
            
            
            if self.bd==None:
                phi = x
                grad_phi = grad_x
                grad_grad_phi = grad_grad_x
            else:
                phi = x*bd_value
                grad_phi = x*grad_bd_value+grad_x*bd_value
                grad_grad_phi = x*grad_grad_bd_value + 2*grad_x*grad_bd_value + grad_grad_x*bd_value
            # normalization
            if normed:
                return phi / (torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1)), grad_phi / (torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1)), grad_grad_phi / (torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1))
            else:
                return phi, grad_phi, grad_grad_phi

    def extra_repr(self):
        return '{}\n{}'.format('Architectures of one TNN(dim={},rank={}) which has {} FNNs:'.format(self.dim,self.p,self.dim),\
                'Each FNN has size: {}'.format(self.size))

class TNN_periodic(nn.Module):
    """
    Architectures of the simple tensor neural network.
    FNN on each demension has the same size,
    and the input integration points are same in different dinension.
    TNN values and gradient values at data points are provided.

    Parameters:
        dim: dimension of TNN, number of FNNs
        size: [1, n0, n1, ..., nl, p], size of each FNN
        activation: activation function used in hidden layers
        period: period of function
        initializer: initial method for learnable parameters
    """

    def __init__(self, dim, size, activation, period, bd=None, grad_bd=None, grad_grad_bd=None, grad_grad_grad_bd=None, bd_rank=None, scaling=True, extra_size=False,
                 initializer='default'):
        super(TNN_periodic, self).__init__()
        self.dim = dim
        self.size = size
        self.activation = activation()
        self.period = period
        self.bd = bd
        self.grad_bd = grad_bd
        self.grad_grad_bd = grad_grad_bd
        self.grad_grad_grad_bd = grad_grad_grad_bd
        self.bd_rank = bd_rank
        self.scaling = scaling
        self.extra_size = extra_size

        self.p = abs(self.size[-1])

        self.ms = self.__init_modules()
        self.__initialize()

    # Register learnable parameters of TNN module.
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(1, len(self.size)):
            bias = True if self.size[i] > 0 else False
            modules['TNN_Linear{}'.format(i - 1)] = TNN_Linear(self.dim, abs(self.size[i]), abs(self.size[i - 1]), bias)
        if self.scaling:
            modules['TNN_Scaling'] = TNN_Scaling([self.p])
        if self.extra_size:
            modules['TNN_Extra'] = TNN_Extra(self.extra_size)
        return modules

    # Initialize learnable parameters of TNN module.
    def __initialize(self):
        # for i in range(1, len(self.size)):
        #     nn.init.orthogonal_(self.ms['TNN_Linear{}'.format(i-1)].weight)
        #     if self.size[i] > 0:
        #         nn.init.constant_(self.ms['TNN_Linear{}'.format(i-1)].bias, 0)
        for i in range(2, len(self.size)):
            for j in range(self.dim):
                    # nn.init.orthogonal_(self.ms['TNN_Linear{}'.format(i-1)].weight[j,:,:])
                    nn.init.kaiming_normal_(self.ms['TNN_Linear{}'.format(i-1)].weight[j,:,:])
                    # nn.init.normal_(self.ms['TNN_Linear{}'.format(i-1)].weight[j,:,:])
            if self.size[i] > 0:
                # nn.init.constant_(self.ms['TNN_Linear{}'.format(i-1)].bias, 0)
                # nn.init.normal_(self.ms['TNN_Linear{}'.format(i-1)].bias)
                nn.init.kaiming_normal_(self.ms['TNN_Linear{}'.format(i-1)].bias)
                # nn.init.orthogonal_(self.ms['TNN_Linear{}'.format(i-1)].bias)

        if self.size[1] > 0:
            # nn.init.constant_(self.ms['TNN_Linear{}'.format(0)].bias, 0)
            # nn.init.normal_(self.ms['TNN_Linear{}'.format(0)].bias)
            # nn.init.orthogonal_(self.ms['TNN_Linear{}'.format(0)].bias)
            nn.init.kaiming_normal_(self.ms['TNN_Linear{}'.format(0)].bias)

        self.ms['TNN_Linear{}'.format(0)].weight.requires_grad = False

        for j in range(self.dim):
            # with torch.no_grad():
            self.ms['TNN_Linear{}'.format(0)].weight[j, :, :] = (torch.arange(1, self.size[1] + 1, dtype=torch.double) * 2 * torch.pi / self.period).unsqueeze(-1)
        
        if self.scaling:
            nn.init.constant_(self.ms['TNN_Scaling'].alpha, 1)
        if self.extra_size:
            nn.init.constant_(self.ms['TNN_Extra'].beta, 1)

    # function to return scaling parameters
    def scaling_par(self):
        if self.scaling:
            return self.ms['TNN_Scaling'].alpha
        else:
            raise NameError('The TNN Module does not have Scaling Parameters')

    # function to return extra parameters
    def extra_par(self):
        if self.extra_size:
            return self.ms['TNN_Extra'].beta
        else:
            raise NameError('The TNN Module does not have Extra Parameters')

    def forward(self, w, x, need_grad=0, normed=True):
        """
        Parameters:
            w: quadrature weights [N]
            x: quadrature points [N]
            need_grad: if return gradient or not

        Returns:
            phi: values of each dimensional FNN [dim, p, N]
            grad_phi: gradient values of each dimensional FNN [dim, p, N]
        """
        x = x.unsqueeze(dim=-2)
        w = w.unsqueeze(dim=-2)
        # Compute values of each one-dimensional input FNN at each quadrature point.
        if need_grad == 0:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            if self.bd_rank==None:
                bd_rank = None
            else:
                bd_rank = self.bd_rank
            # Forward process.
            for i in range(1, len(self.size) - 1):
                x = self.ms['TNN_Linear{}'.format(i - 1)](x)
                x = self.activation(x)
            if bd_value==None:
                phi = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
            else:
                if bd_rank==None:
                    phi = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)*bd_value
                else:
                    raw_phi = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
                    mask = torch.zeros(raw_phi.shape, dtype=torch.bool, device=raw_phi.device)
                    mask[:, bd_rank, :] = True
                    phi = torch.where(mask, raw_phi * bd_value, raw_phi)
            
            # normalization
            if normed:
                norm_phi = torch.sqrt(torch.sum(w * phi ** 2, dim=-1)).unsqueeze(dim=-1)
                return phi / norm_phi
            else:
                return phi

        # Compute values and gradient values of each one-dimensional input FNN at each quadrature point simutaneously.
        if need_grad == 1:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            if self.bd_rank==None:
                bd_rank = None
            else:
                bd_rank = self.bd_rank
            # Get gradient values of forced boundary condition function.
            if self.grad_bd==None:
                grad_bd_value = None
            else:
                grad_bd_value = self.grad_bd(x)
            # Compute forward and backward process simutaneously.
            grad_x = self.ms['TNN_Linear{}'.format(0)].weight
            for i in range(1, len(self.size) - 1):
                x = self.ms['TNN_Linear{}'.format(i - 1)](x)
                grad_x = self.activation.grad(x) * grad_x
                grad_x = self.ms['TNN_Linear{}'.format(i)].weight @ grad_x
                x = self.activation(x)

            x = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
            if self.bd==None:
                phi = x
                grad_phi = grad_x
            else:
                if self.bd_rank==None:
                    phi = x*bd_value
                    grad_phi = x*grad_bd_value+grad_x*bd_value
                else:
                    mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
                    mask[:, bd_rank, :] = True
                    phi = torch.where(mask, x*bd_value, x)
                    grad_phi = torch.where(mask, x*grad_bd_value+grad_x*bd_value, grad_x)
            # normalization
            if normed:
                norm_phi = torch.sqrt(torch.sum(w * phi ** 2, dim=-1)).unsqueeze(dim=-1)
                return phi / norm_phi, grad_phi / norm_phi
            else:
                return phi, grad_phi
            
        if need_grad == 2:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            # Get gradient values of forced boundary condition function.
            if self.grad_bd==None:
                grad_bd_value = None
            else:
                grad_bd_value = self.grad_bd(x)
            # get grad_grad_bd values
            if self.grad_grad_bd==None:
                grad_grad_bd_value = None
            else:
                grad_grad_bd_value = self.grad_grad_bd(x)

            # Compute forward and backward process simutaneously.
            grad_x = self.ms['TNN_Linear{}'.format(0)].weight
            grad_grad_x = torch.zeros_like(grad_x)
            for i in range(1, len(self.size) - 1):
                x = self.ms['TNN_Linear{}'.format(i-1)](x)
                grad_grad_x = self.activation.grad_grad(x)*(grad_x**2)+self.activation.grad(x)*grad_grad_x
                grad_grad_x = self.ms['TNN_Linear{}'.format(i)].weight@grad_grad_x
                grad_x = self.activation.grad(x)*grad_x
                grad_x = self.ms['TNN_Linear{}'.format(i)].weight@grad_x
                x = self.activation(x)
            x = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
            if self.bd==None:
                phi = x
                grad_phi = grad_x
                grad_grad_phi = grad_grad_x
            else:
                if self.bd_rank==None:
                    phi = x*bd_value
                    grad_phi = x*grad_bd_value+grad_x*bd_value
                    grad_grad_phi = x*grad_grad_bd_value + 2*grad_x*grad_bd_value + grad_grad_x*bd_value
                else:
                    mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
                    mask[:, self.bd_rank, :] = True
                    phi = torch.where(mask, x*bd_value, x)
                    grad_phi = torch.where(mask, x*grad_bd_value+grad_x*bd_value, grad_x)
                    grad_grad_phi = torch.where(mask, x*grad_grad_bd_value + 2*grad_x*grad_bd_value + grad_grad_x*bd_value, grad_grad_x)

            if normed:
                norm_phi = torch.sqrt(torch.sum(w * phi ** 2, dim=-1)).unsqueeze(dim=-1)
                return phi / norm_phi, grad_phi / norm_phi, grad_grad_phi / norm_phi
            else:
                return phi, grad_phi, grad_grad_phi

        if need_grad==3:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            if self.grad_bd==None:
                grad_bd_value = None
            else:
                grad_bd_value = self.grad_bd(x)
            if self.bd_rank==None:
                bd_rank = None
            else:
                bd_rank = self.bd_rank
            if self.grad_grad_bd==None:
                grad_grad_bd_value = None
            else:
                grad_grad_bd_value = self.grad_grad_bd(x)
            if self.grad_grad_grad_bd==None:
                grad_grad_grad_bd_value = None
            else:
                grad_grad_grad_bd_value = self.grad_grad_grad_bd(x)
            # Compute forward and backward process simutaneously.
            grad_x = self.ms['TNN_Linear{}'.format(0)].weight
            grad_grad_x = torch.zeros_like(grad_x)
            grad_grad_grad_x = torch.zeros_like(grad_x)
            
            for i in range(1, len(self.size) - 1):
                x = self.ms['TNN_Linear{}'.format(i-1)](x)
                grad_grad_grad_x = self.activation.grad_grad_grad(x)*(grad_x**3)+3*self.activation.grad_grad(x)*grad_grad_x*grad_x+self.activation.grad(x)*grad_grad_grad_x
                grad_grad_grad_x = self.ms['TNN_Linear{}'.format(i)].weight@grad_grad_grad_x
                grad_grad_x = self.activation.grad_grad(x)*(grad_x**2)+self.activation.grad(x)*grad_grad_x
                grad_grad_x = self.ms['TNN_Linear{}'.format(i)].weight@grad_grad_x
                grad_x = self.activation.grad(x)*grad_x
                grad_x = self.ms['TNN_Linear{}'.format(i)].weight@grad_x
                x = self.activation(x)
            x = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)

            if self.bd==None:
                phi = x
                grad_phi = grad_x
                grad_grad_phi = grad_grad_x
                grad_grad_grad_phi = grad_grad_grad_x
            else:   
                if bd_rank==None:
                    phi = x*bd_value
                    grad_phi = x*grad_bd_value+grad_x*bd_value
                    grad_grad_phi = x*grad_grad_bd_value + 2*grad_x*grad_bd_value + grad_grad_x*bd_value
                    grad_grad_grad_phi = x*grad_grad_grad_bd_value + 3*grad_grad_x*grad_bd_value + 3*grad_x*grad_grad_bd_value + grad_grad_grad_x*bd_value
                else:
                    mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
                    mask[:, bd_rank, :] = True
                    phi = torch.where(mask, x*bd_value, x)
                    grad_phi = torch.where(mask, x*grad_bd_value+grad_x*bd_value, grad_x)
                    grad_grad_phi = torch.where(mask, x*grad_grad_bd_value + 2*grad_x*grad_bd_value + grad_grad_x*bd_value, grad_grad_x)
                    grad_grad_grad_phi = torch.where(mask, x*grad_grad_grad_bd_value + 3*grad_grad_x*grad_bd_value + 3*grad_x*grad_grad_bd_value + grad_grad_grad_x*bd_value, grad_grad_grad_x)
            # normalization
            if normed:
                norm_phi = torch.sqrt(torch.sum(w * phi ** 2, dim=-1)).unsqueeze(dim=-1)
                return phi / norm_phi, grad_phi / norm_phi, grad_grad_phi / norm_phi, grad_grad_grad_phi / norm_phi
            else:
                return phi, grad_phi, grad_grad_phi, grad_grad_grad_phi

    def extra_repr(self):
        return '{}\n{}'.format(
            'Architectures of one TNN(dim={},rank={}) which has {} FNNs:'.format(self.dim, self.p, self.dim), \
            'Each FNN has size: {}'.format(self.size))


# Multi-TNN constructed by k TNNs.
class Multi_TNN(nn.Module):
    """
    Define k simple TNNs simultaneously.
    Each TNN has the same structure.

    Parameters:
        k: number of TNNs
        dim: dimension of each TNN
        size: size of each FNN
        activation: activation function used in hidden layers
        bd: extra function for boundary condition
        grad_bd: gradient of bd
        initializer: initial method for learnable parameters
    """
    def __init__(self, k, dim, size, activation, bd=None, grad_bd=None, scaling=True, extra_size=False, initializer='default'):
        super(Multi_TNN, self).__init__()
        self.k = k
        self.dim = dim
        self.size = size
        self.activation = activation()
        self.bd = bd
        self.grad_bd = grad_bd
        self.scaling = scaling
        self.extra_size = extra_size

        self.p = abs(self.size[-1])

        self.ms = self.__init_modules()
        self.__initialize()

    # Register learnable parameters of TNN module.
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(1, len(self.size)):
            bias = True if self.size[i] > 0 else False
            modules['TNN_Linear{}'.format(i-1)] = TNN_Linear(self.k*self.dim,abs(self.size[i]),abs(self.size[i-1]),bias)
        if self.scaling:
            modules['TNN_Scaling'] = TNN_Scaling([self.k,self.p])
        if self.extra_size:
            modules['TNN_Extra'] = TNN_Extra(self.extra_size)
        return modules

    # Initialize learnable parameters of TNN module.
    def __initialize(self):
        # for i in range(1, len(self.size)):
        #     nn.init.orthogonal_(self.ms['TNN_Linear{}'.format(i-1)].weight)
        #     if self.size[i] > 0:
        #         nn.init.constant_(self.ms['TNN_Linear{}'.format(i-1)].bias, 0)
        for i in range(1, len(self.size)):
            for j in range(self.k*self.dim):
                nn.init.orthogonal_(self.ms['TNN_Linear{}'.format(i-1)].weight[j,:,:])
            if self.size[i] > 0:
                # nn.init.constant_(self.ms['TNN_Linear{}'.format(i-1)].bias, 0)
                nn.init.constant_(self.ms['TNN_Linear{}'.format(i-1)].bias, 0)
        if self.scaling:
            nn.init.constant_(self.ms['TNN_Scaling'].alpha, 1)
        if self.extra_size:
            nn.init.constant_(self.ms['TNN_Extra'].beta, 1)

    # function to return scaling parameters
    def scaling_par(self):
        if self.scaling:
            return self.ms['TNN_Scaling'].alpha
        else:
            raise NameError('The TNN Module does not have Scaling Parameters')

    # function to return extra parameters
    def extra_par(self):
        if self.extra_size:
            return self.ms['TNN_Extra'].beta
        else:
            raise NameError('The TNN Module does not have Extra Parameters')


    def forward(self,w,x,need_grad=0,normed=True):
        """
        Parameters:
            w: quadrature weights [N]
            x: quadrature points [N]
            need_grad: if return gradient or not
        
        Returns:
            phi: values of each dimensional FNN [k, dim, p, N]
            grad_phi: gradient values of each dimensional FNN [k, dim, p, N]
        """
        # Compute values of each one-dimensional input FNN at each quadrature point.
        if need_grad==0:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            # Forward process.
            for i in range(1, len(self.size) - 1):
                x = self.ms['TNN_Linear{}'.format(i-1)](x)
                x = self.activation(x)
            if bd_value==None:
                phi = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
            else:
                phi = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)*bd_value
            # normalization
            if normed:
                return (phi / torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1)).view(self.k,self.dim,self.p,-1)
            else:
                phi.view(self.k,self.dim,self.p,-1)

        # Compute values and gradient values of each one-dimensional input FNN at each quadrature point simutaneously.
        if need_grad==1:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            # Get gradient values of forced boundary condition function.
            if self.grad_bd==None:
                grad_bd_value = None
            else:
                grad_bd_value = self.grad_bd(x)
            # Compute forward and backward process simutaneously.
            grad_x = self.ms['TNN_Linear{}'.format(0)].weight
            for i in range(1, len(self.size) - 1):
                x = self.ms['TNN_Linear{}'.format(i-1)](x)
                grad_x = self.activation.grad(x)*grad_x
                grad_x = self.ms['TNN_Linear{}'.format(i)].weight@grad_x
                x = self.activation(x)
            x = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
            if self.bd==None:
                phi = x
                grad_phi = grad_x
            else:
                phi = x*bd_value
                grad_phi = x*grad_bd_value+grad_x*bd_value
            # noemalization
            if normed:
                return (phi / torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1)).view(self.k,self.dim,self.p,-1), (grad_phi / torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1)).view(self.k,self.dim,self.p,-1)
            else:
                return phi.view(self.k,self.dim,self.p,-1), grad_phi.view(self.k,self.dim,self.p,-1)

    def extra_repr(self):
        return '{}\n{}'.format('Architectures of {} TNNs(each dim={},rank={}), and each TNN has {} FNNs:'.format(self.k,self.dim,self.p,self.dim), 'The total number of FNNs is {}, and each FNN has size: {}'.format(self.k*self.dim,self.size))



class TNN_ParFree(nn.Module):
    """

    """
    def __init__(self, dim, p, size, activation, bd=None, grad_bd=None, initializer='default'):
        super(TNN_ParFree, self).__init__()
        self.dim = dim
        self.p = p
        self.size = size
        self.activation = activation()
        self.bd = bd
        self.grad_bd = grad_bd

        self.ms = self.__init_modules()
        self.__initialize()

    # Register learnable parameters of TNN module.
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(1, len(self.size)):
            bias = True if self.size[i] > 0 else False
            modules['TNN_Linear{}'.format(i-1)] = TNN_Linear(self.dim*self.p,abs(self.size[i]),abs(self.size[i-1]),bias)
        modules['TNN_Scaling'] = TNN_Scaling([self.p])
        return modules

    # Initialize learnable parameters of TNN module.
    def __initialize(self):
        # for i in range(1, len(self.size)):
        #     nn.init.orthogonal_(self.ms['TNN_Linear{}'.format(i-1)].weight)
        #     if self.size[i] > 0:
        #         nn.init.constant_(self.ms['TNN_Linear{}'.format(i-1)].bias, 0)
        for i in range(1, len(self.size)):
            for j in range(self.dim*self.p):
                nn.init.orthogonal_(self.ms['TNN_Linear{}'.format(i-1)].weight[j,:,:])
            if self.size[i] > 0:
                nn.init.constant_(self.ms['TNN_Linear{}'.format(i-1)].bias, 0)
        nn.init.constant_(self.ms['TNN_Scaling'].alpha, 1)


    def forward(self,w,x,need_grad=0):
        # Compute values of each one-dimensional input FNN at each quadrature point.
        if need_grad==0:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            # Forward process.
            for i in range(1, len(self.size) - 1):
                x = self.ms['TNN_Linear{}'.format(i-1)](x)
                x = self.activation(x)
            if bd_value==None:
                phi = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
            else:
                phi = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)*bd_value
            # normalization
            return self.ms['TNN_Scaling'].alpha, (phi / torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1)).squeeze().view(self.dim,self.p,-1)

        # Compute values and gradient values of each one-dimensional input FNN at each quadrature point simutaneously.
        if need_grad==1:
            # Get values of forced boundary condition function.
            if self.bd==None:
                bd_value = None
            else:
                bd_value = self.bd(x)
            # Get gradient values of forced boundary condition function.
            if self.grad_bd==None:
                grad_bd_value = None
            else:
                grad_bd_value = self.grad_bd(x)
            # Compute forward and backward process simutaneously.
            grad_x = self.ms['TNN_Linear{}'.format(0)].weight
            for i in range(1, len(self.size) - 1):
                x = self.ms['TNN_Linear{}'.format(i-1)](x)
                grad_x = self.activation.grad(x)*grad_x
                grad_x = self.ms['TNN_Linear{}'.format(i)].weight@grad_x
                x = self.activation(x)
            x = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
            if self.bd==None:
                phi = x
                grad_phi = grad_x
            else:
                phi = x*bd_value
                grad_phi = x*grad_bd_value+grad_x*bd_value
            return self.ms['TNN_Scaling'].alpha, (phi / torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1)).squeeze().view(self.dim,self.p,-1), (grad_phi / torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1)).squeeze().view(self.dim,self.p,-1)

    def extra_repr(self):
        return '{}\n{}'.format('Architectures of one TNN(dim={},rank={}) which has {} FNNs'.format(self.dim,self.p,self.dim*self.p),\
                'Each FNN has size: {}'.format(self.size))




def main():
    dtype = torch.double
    device = 'cpu'

    dim = 3
    p = 5
    k = 3
    size = [1, 20, -10, 15, -5]
    activation = TNN_Tanh

    a = -10
    b = 10
    # quadrature rule:
    # number of quad points
    quad = 16
    # number of partitions for [a,b]
    n = 10
    # quad ponits and quad weights.
    x, w = composite_quadrature_1d(quad, a, b, n, device=device, dtype=dtype)

    # define forced boundary condition function.
    def bd(x):
        return (x-a)*(b-x)
    # define derivative of forced boundary condition function.
    def grad_bd(x):
        return 2*x-2*a*b

    # print(size)
    # model = TNN(dim,size,activation,bd=bd,grad_bd=grad_bd,scaling=False).to(dtype).to(device)
    # print(model)

    # model = Multi_TNN(k,dim,size,activation,bd=bd,grad_bd=grad_bd,scaling=False).to(dtype).to(device)
    model = Multi_TNN(k,dim,size,activation,bd=bd,grad_bd=grad_bd).to(dtype).to(device)
    print(model)

    # size = [1,5,10,8,-1]
    # model = TNN_ParFree(dim,p,size,activation,bd=bd,grad_bd=grad_bd).to(dtype).to(device)
    # print(model)

    # phi, grad_phi = model(w,x,need_grad=1)
    alpha, phi, grad_phi = model(w,x,need_grad=1)
    # alpha, phi, grad_phi = model(w,x,need_grad=1)
    # print(alpha.size())
    print(phi.size())
    print(torch.sum(w*phi[2]**2,dim=2))



if __name__ == '__main__':
    main()
