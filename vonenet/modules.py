
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .utils import gabor_kernel
from numba import jit

torch.pi = torch.acos(torch.zeros(1)).item() * 2

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_built():
    device = "mps"
    mps = True
else:
    device = "cpu"


class Identity(nn.Module):
    def forward(self, x):
        return x


class GFB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (kernel_size // 2, kernel_size // 2)

        # Param instatiations
        self.weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding)

    def initialize(self, sf, theta, sigx, sigy, phase, rgb_seed = 0):
        torch.manual_seed(rgb_seed)
        random_channel = torch.randint(0, self.in_channels, (self.out_channels,))
        self.random_channel = random_channel
        for i in range(self.out_channels):
            self.weight[i, random_channel[i]] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                             theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
        self.weight = nn.Parameter(self.weight, requires_grad=False)


class VOneBlock(nn.Module):
    def __init__(self, sf, theta, sigx, sigy, phase,
                 k_exc=25, noise_mode=None, noise_scale=1, noise_level=1,
                 simple_channels=128, complex_channels=128, ksize=25, stride=4, input_size=224,
                 rgb_seed = 0):
        super().__init__()

        self.in_channels = 3

        self.simple_channels = simple_channels
        self.complex_channels = complex_channels
        self.out_channels = simple_channels + complex_channels
        self.stride = stride
        self.ksize = ksize
        self.input_size = input_size

        self.sf = sf
        self.theta = theta
        self.sigx = sigx
        self.sigy = sigy
        self.phase = phase
        self.k_exc = k_exc

        self.set_noise_mode(noise_mode, noise_scale, noise_level)
        self.fixed_noise = None

        self.simple_conv_q0 = GFB(self.in_channels, self.out_channels, self.ksize, self.stride)
        self.simple_conv_q1 = GFB(self.in_channels, self.out_channels, self.ksize, self.stride)
        self.simple_conv_q0.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase, rgb_seed = rgb_seed)
        self.simple_conv_q1.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase + np.pi / 2, rgb_seed = rgb_seed+1)

        self.simple = nn.ReLU(inplace=True)
        self.complex = Identity()
        self.gabors = Identity()
        self.noise = nn.ReLU(inplace=True)
        self.output = Identity()

    def forward(self, x):
        # Gabor activations [Batch, out_channels, H/stride, W/stride]
        x = self.gabors_f(x)
        # Noise [Batch, out_channels, H/stride, W/stride]
        x = self.noise_f(x)
        # V1 Block output: (Batch, out_channels, H/stride, W/stride)
        x = self.output(x)
        return x

    def gabors_f(self, x):
        s_q0 = self.simple_conv_q0(x)
        s_q1 = self.simple_conv_q1(x)
        c = self.complex(torch.sqrt(s_q0[:, self.simple_channels:, :, :] ** 2 +
                                    s_q1[:, self.simple_channels:, :, :] ** 2) / np.sqrt(2))
        s = self.simple(s_q0[:, 0:self.simple_channels, :, :])
        return self.gabors(self.k_exc * torch.cat((s, c), 1))

    def noise_f(self, x):
        if self.noise_mode == 'neuronal':
            eps = 10e-5
            x *= self.noise_scale
            x += self.noise_level
            if self.fixed_noise is not None:
                x += self.fixed_noise * torch.sqrt(F.relu(x.clone()) + eps)
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * \
                     torch.sqrt(F.relu(x.clone()) + eps)
            x -= self.noise_level
            x /= self.noise_scale
        if self.noise_mode == 'gaussian':
            if self.fixed_noise is not None:
                x += self.fixed_noise * self.noise_scale
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * self.noise_scale
        return self.noise(x)

    def set_noise_mode(self, noise_mode=None, noise_scale=1, noise_level=1):
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.noise_level = noise_level

    def fix_noise(self, batch_size=256, seed=None):
        noise_mean = torch.zeros(batch_size, self.out_channels, int(self.input_size/self.stride),
                                 int(self.input_size/self.stride))
        if seed:
            torch.manual_seed(seed)
        if self.noise_mode:
            self.fixed_noise = torch.distributions.normal.Normal(noise_mean, scale=1).rsample().to(device)

    def unfix_noise(self):
        self.fixed_noise = None

# def gaussian_fn(M, mu, std):
#     n = torch.arange(0-mu, M-mu) - (M - 1.0) / 2.0
#     sig2 = 2 * std * std
#     w = torch.exp(-n ** 2 / sig2)
#     return w

# def gkern(kernlen=256, std=128):
#     """Returns a 2D Gaussian kernel array."""
#     gkern1d = gaussian_fn(kernlen, std=std) 
#     gkern2d = torch.outer(gkern1d, gkern1d)
#     return gkern2d

# def gaussianKernel(theta, v, w, rho, sigma, A, in_size = 50):

#     Sigma = torch.diag(torch.Tensor([rho, sigma]))
#     mu = torch.Tensor([v,w])

#     x = torch.linspace(-1,1,in_size)
#     # print(x.device)

#     x, y = torch.meshgrid(x, x)
#     # print(x.device, theta.device)

#     x_rot = x * torch.cos(theta) + y * torch.sin(theta)
#     y_rot = -x * torch.sin(theta) + y * torch.cos(theta)

#     pos = torch.zeros(x_rot.shape + (2,))
#     pos[:, :, 0] = x_rot
#     pos[:, :, 1] = y_rot

#     const = A / (2 * torch.pi * rho * sigma)
    
#     Sigma_inv = torch.inverse(Sigma)

#     delta = torch.subtract(pos,mu)

#     fac = torch.einsum('...k,kl,...l->...', delta, Sigma_inv, delta)

#     return const * torch.exp(-fac / 2)
        

@torch.jit.script
def gaussianKernel(theta, v, w, rho, sigma, A, in_size:int=50, device: torch.device = torch.device(device)):

    Sigma = torch.diag(torch.hstack([rho, sigma])).to(device)
    mu = torch.hstack([v,w]).to(device)

    # x = np.arange(0, in_size)
    x = torch.linspace(-1,1,in_size).to(device)

    # print(x.shape)

    x, y = torch.meshgrid(x, x)

    # print(x.device, theta.device)

    x_rot = x * torch.cos(theta) + y * torch.sin(theta)
    y_rot = -x * torch.sin(theta) + y * torch.cos(theta)

    pos = torch.zeros(x_rot.shape + (2,)).to(device)
    pos[:, :, 0] = x_rot
    pos[:, :, 1] = y_rot

    const = A / (2 * torch.pi * rho * sigma)
    
    Sigma_inv = torch.inverse(Sigma)

    delta = torch.subtract(pos,mu)

    fac = torch.einsum('...k,kl,...l->...', delta, Sigma_inv, delta)

    return const * torch.exp(-fac / 2)

# DIMENSIONS OF OUTPUT FROM VONEBLOCK MUST BE STORED SOMEWHERE!
# MAP FROM PARAMETERS TO FILTER!

class DNBlock(nn.Module):

    # initialise all parameters
    # compute denominator (bias plus params of kernels all trainable?)
    # bank size, image size
    # compute full expression
    # return

    def __init__(self, beta=1e-6):
        super().__init__()

        self.kernel = None

        self.beta = beta

    def initialise(self, cov_matrix):

        self.kernel = cov_matrix

        print("Cov matrix passed to DNBlock")
    
    def denominator(self,x):

        trial = x.reshape(-1, np.prod(list(x.shape[1:])))

        div = self.kernel@trial.T

        return div.T.reshape(x.shape)

    def forward(self,x):
            
        if self.kernel != None:

            den = self.denominator(x)

            den += self.beta

            return x / den
        
        else:

            return x
        
class DNBlockv2(nn.Module):

    # initialise all parameters
    # compute denominator (bias plus params of kernels all trainable?)
    # bank size, image size
    # compute full expression
    # return

    def __init__(self, beta=1.0, channels = 64):
        super().__init__()

        self.kernel = None

        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)

        self.channels = channels

        norm_mults = torch.zeros(channels, channels) * 1 / channels**2

        self.norm_mults = nn.Parameter(torch.tensor(norm_mults), requires_grad=False)

        
    def initialise(self, cov_matrix):

        self.kernel = cov_matrix

        print("Cov matrix passed to DNBlock")
    
    def denominator(self,x):

        inter = x.permute(0,3,2,1)
        result = torch.einsum('bxyc,cd->bxyc', inter, self.norm_mults)

        result = result.permute(0, 3, 1, 2)

        trial = result.reshape(-1, np.prod(list(result.shape[1:])))

        div = self.kernel@trial.T

        return div.T.reshape(x.shape)

    def forward(self,x):
            
        if self.kernel != None:

            den = self.denominator(x)

            den += self.beta

            return x / den
        
        else:

            return x
    

class GaussianDNBlock(nn.Module):

    # initialise all parameters
    # compute denominator (bias plus params of kernels all trainable?)
    # bank size, image size
    # compute full expression
    # return

    def __init__(self, beta=0.0001, channels = 64, in_size = 32, ksize = 32):
        super().__init__()

        self.in_size = in_size
        self.bank_size = channels
        self.kernel = gaussianKernel

        self.bias = nn.Parameter(torch.tensor(beta), requires_grad=True)

        self.ksize = ksize

        self.initialise()


    def initialise(self):

        # 512^2 kernels. scale, two means, two variances, rotation
        # = 6 parameters per kernel
        self.params = torch.rand(self.bank_size, self.bank_size, 6)

        self.computeCoefficients()
        # enable autograd to accumulate across params


    def computeCoefficients(self, device: torch.device = torch.device(device)):

        # NEEDS TO HAPPEN ON CPU!!!

        weights = torch.zeros(self.bank_size, self.bank_size, self.in_size, self.in_size).to(device)

        # x = torch.linspace(-1, 1, self.in_size, device="cpu")
        # params = self.params.to("cpu")

        # I STILL EXPECT THIS TO BE VERY COSTLY.

        for i in range(self.bank_size):
            for j in range(self.bank_size):
                weights[i][j] = self.kernel(*self.params[i][j], in_size=self.in_size)

        print(self.bank_size, self.in_size)

        raise(ValueError)
        
        self.kernel = weights.reshape(self.bank_size * self.in_size, -1)

    
    def denominator(self,x):

        trial = x.reshape(-1, np.prod(list(x.shape[1:])))

        div = self.kernel@trial.T

        return div.T.reshape(x.shape)


    def forward(self,x):

        den = self.denominator(x) + self.bias

        return x / den

# class DNBlock(nn.Module):

#     # initialise all parameters
#     # compute denominator (bias plus params of kernels all trainable?)
#     # bank size, image size
#     # compute full expression
#     # return

#     def __init__(self, in_size, bank_size, beta=1e-4):
#         super().__init__()

#         self.in_size = in_size
#         self.bank_size = bank_size
#         self.kernel = gaussianKernel

#         self.weights = torch.zeros((bank_size, bank_size, in_size, in_size))

#         self.beta = beta

#         self.initialize()


#     def initialize(self):

#         # 512^2 kernels. scale, two means, two variances, rotation
#         # = 6 parameters per kernel
#         params = torch.rand(self.bank_size, self.bank_size, 6)

#         self.params = nn.Parameter(params, requires_grad = True)
#         # print(self.params.device)
#         # enable autograd to accumulate across params


#     def computeCoefficients(self):

#         # NEEDS TO HAPPEN ON CPU!!!

#         # self.weights = torch.zeros((bank_size, bank_size, in_size, in_size)) (IN INIT)
#         weights = torch.zeros((self.bank_size, self.bank_size, self.in_size, self.in_size))
#         params = self.params.to("cpu")

#         # x = torch.linspace(-1, 1, self.in_size, device="cpu")

#         # I STILL EXPECT THIS TO BE VERY COSTLY.
#         # print(self.params.device)
#         for i in range(self.bank_size):
#             for j in range(self.bank_size):
#                 weights[i][j] = self.kernel(*params[i][j], in_size=self.in_size)

#         self.weights = weights.to(device) # EXPORT TO GPU ONLY WHEN COMPUTED

        

#     def denominator(self,x):

#         # neat trickery that enables weights matrix pointwise multiplication
#         # in the same fashion that the denominator sum is described
#         # in the original paper

#         self.computeCoefficients()
#         # re-computes the kernels accn. to current state of implicit
#         # trainable parameters
#         # how many times is this computed? -> autograd might struggle

#         expanded_images_tensor = x.unsqueeze(1).expand(-1, self.bank_size, -1, -1)
#         result_tensor = self.weights * expanded_images_tensor
#         summed = torch.sum(result_tensor, dim=0)
#         summed = summed + self.beta

#         return summed

#     def forward(self,x):

#         den = self.denominator(x)

#         return x / den


# def gaussianKernel(x, y, theta, v, w, rho, sigma, A):

#     Sigma = np.diag([rho, sigma])
#     mu = np.array([v,w])

#     x, y = np.meshgrid(x, y)

#     x_rot = x * np.cos(theta) + y * np.sin(theta)
#     y_rot = -x * np.sin(theta) + y * np.cos(theta)

#     pos = np.empty(x_rot.shape + (2,))
#     pos[:, :, 0] = x_rot
#     pos[:, :, 1] = y_rot

#     const = A / (2 * np.pi * rho * sigma)
    
#     Sigma_inv = np.linalg.inv(Sigma)

#     fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

#     return const * np.exp(-fac / 2)

# DIMENSIONS OF OUTPUT FROM VONEBLOCK MUST BE STORED SOMEWHERE!
# MAP FROM PARAMETERS TO FILTER!

# def torchify(data):

#     return torch.from_numpy(data.astype(np.float32))
#     # return torch.from_numpy(data.astype(np.float32)).to(device)
# #########

# class DNBlock(nn.Module):

#     # initialise all parameters
#     # compute denominator (bias plus params of kernels all trainable?)
#     # bank size, image size
#     # compute full expression
#     # return

#     def __init__(self, in_size, bank_size):
#         super().__init__()

#         self.in_size = in_size
#         self.bank_size = bank_size
#         self.kernel = gaussianKernel

#         self.weights = torch.zeros((bank_size, bank_size, in_size, in_size))


#     def initialize(self):

#         bias = np.array([0.5])
#         self.bias = nn.Parameter(torchify(bias), requires_grad=True)

#         bank_size = self.bank_size
#         in_size = self.in_size

#         # randomise between (0, pi)
#         theta = np.random.rand(bank_size, bank_size) * np.pi
#         theta = torchify(theta)
#         self.theta = nn.Parameter(theta, requires_grad = True) 

#         # (25th to 75th percentiles)
#         size = in_size // 4
#         sz = np.float32(np.arange(size, 3 * size + 1))

#         v = np.random.choice(sz, (bank_size, bank_size))
#         v = torchify(v)
#         self.v = nn.Parameter(v, requires_grad=True)

#         w = np.random.choice(sz, (bank_size, bank_size))
#         w = torchify(w)
#         self.w = nn.Parameter(w, requires_grad=True)

#         # start from (circa 1, size/2)
#         rho = (np.random.rand(bank_size, bank_size) + 1e-2) * in_size/2
#         rho = torchify(rho)
#         self.rho = nn.Parameter(rho, requires_grad=True)

#         # start from (circa 1, size/2)
#         sg = (np.random.rand(bank_size, bank_size) + 1e-2) * in_size/2
#         sg = torchify(sg)
#         self.sigma = nn.Parameter(sg, requires_grad=True)

#         # start from 1 / banksize
#         A = 1 / bank_size * np.ones((bank_size, bank_size))
#         A = torchify(A)
#         self.A = nn.Parameter(A, requires_grad=True)


#     def computeCoefficients(self):

#         x = torch.arange(0, self.in_size, device=device)
#         y = torch.arange(0, self.in_size, device=device)

#         for i in range(self.bank_size):
#             for j in range(self.bank_size):
                
#                 theta = self.theta[i][j]
#                 v = self.v[i][j]
#                 w = self.w[i][j]
#                 rho = self.rho[i][j]
#                 sigma = self.sigma[i][j]
#                 A = self.A[i][j]

#                 self.weights[i][j] = self.kernel(x, y, theta, v, w, rho, sigma, A)

#     def denominator(self,x):

#         self.computeCoefficients()

#         z = x.reshape(1,*x.shape)
#         xs = torch.cat([z]*self.bank_size, 0)

#         out = torch.sum(torch.mul(xs, self.weights), 1)

#         return self.bias + out
    

#     def forward(self,x):

#         den = self.denominator(x)

#         return x / den
    


  ##########  


# def gaussianKernel(x, y, theta, v, w, rho, sigma, A):

#     Sigma = np.diag([rho, sigma])
#     mu = np.array([v,w])

#     x, y = np.meshgrid(x, y)

#     x_rot = x * np.cos(theta) + y * np.sin(theta)
#     y_rot = -x * np.sin(theta) + y * np.cos(theta)

#     pos = np.empty(x_rot.shape + (2,))
#     pos[:, :, 0] = x_rot
#     pos[:, :, 1] = y_rot

#     const = A / (2 * np.pi * rho * sigma)
    
#     Sigma_inv = np.linalg.inv(Sigma)

#     fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

#     return const * np.exp(-fac / 2)

# DIMENSIONS OF OUTPUT FROM VONEBLOCK MUST BE STORED SOMEWHERE!
# MAP FROM PARAMETERS TO FILTER!

class VOneBlockDN(VOneBlock):

    def __init__(self, sf, theta, sigx, sigy, phase,
                 k_exc=25, noise_mode=None, noise_scale=1, noise_level=1,
                 simple_channels=128, complex_channels=128, ksize=25, stride=4, input_size=224,
                 cov_matrix = None, filters_r = None, filters_c = None, trainable = False, paper_implementation = False):

        super().__init__(sf, theta, sigx, sigy, phase,
                 k_exc, noise_mode, noise_scale, noise_level,
                 simple_channels, complex_channels, ksize, stride, input_size)

        if paper_implementation:
            self.dn = GaussianDNBlock(channels=simple_channels+complex_channels, in_size = input_size)
        else:
            if trainable:
                self.dn = DNBlockv2(channels=simple_channels+complex_channels)
            else:
                self.dn = DNBlock()
            
            self.dn.initialise(cov_matrix)

        self.dn.to(device)

        if filters_r != None:
            self.simple_conv_q0 = filters_r
        if filters_c != None:
            self.simple_conv_q1 = filters_c

        if filters_r != None or filters_c != None:

            self.ksize = self.simple_conv_q0.weight.shape[2]
            self.simple_conv_q0.padding = (self.ksize//2, self.ksize//2)
            self.simple_conv_q1.padding = (self.ksize//2, self.ksize//2)

    def forward(self, x):
        # Gabor activations [Batch, out_channels, H/stride, W/stride]
        x = self.gabors_f(x)
        # Noise [Batch, out_channels, H/stride, W/stride]
        x = self.noise_f(x)
        # V1 Block output: (Batch, out_channels, H/stride, W/stride)
        x = self.output(x)
        # DN output
        x = self.dn(x)

        return x