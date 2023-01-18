# third-party libraries
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# libraries
from SpectralConv2d_fast import SpectralConv2d_fast

class FNO2d(nn.Module):
    """
    The overall neural network:
    1. 
    2. 4 layers of the integral operators u' = (W + K)(u). W defined by self.w; K defined by self.conv
    3. Project from the channel space to the output space by self.fc1 and self.fc2
    """
    def __init__(self, modes1, modes2, width, mesh_x, mesh_y):
        """
        Initialization
            modes1, modes2: number of Fourier modes to apply (at most floor(N/2) + 1)
            width:          number of input and output channels
        """
        super(FNO2d, self).__init__()
        self.modes1    = modes1
        self.modes2    = modes2
        self.width     = width
        self.mesh_x    = mesh_x
        self.mesh_y    = mesh_y

        # pad the domain if input is non-periodic (number of cells to add at each side)
        #self.padding = 1

        # lift input to the desired channel dimension: 10 solutions of previous timesteps + 10 sol velocity + 2 locations x and y
        self.fc0 = nn.Linear(22, self.width)

        # FFT --> R --> inverse FFT
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)

        # convolution
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # batch normalization
        self.bn0 = th.nn.BatchNorm2d(self.width)
        self.bn1 = th.nn.BatchNorm2d(self.width)
        self.bn2 = th.nn.BatchNorm2d(self.width)
        self.bn3 = th.nn.BatchNorm2d(self.width)

        # local linear transform
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def get_grid(self, shape, device):
        """
        generate unform grid in x and y for domain [0,1]x[0,1]
        """
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        #gridx = np.linspace(0,1,size_x,endpoint=False)
        #gridy = np.linspace(0,1,size_y,endpoint=False)

        # shift grid
        #gridx = gridx + 0.5*(gridx[1] - gridx[0])
        #gridy = gridy + 0.5*(gridy[1] - gridy[0]) 
        

        # get tensors with needed dimensions
        #gridx = th.tensor(gridx, dtype=th.float)
        #gridy = th.tensor(gridy, dtype=th.float)
        gridx = self.mesh_x.reshape(1,size_x,1,1).repeat([batchsize,1,size_y,1])
        gridy = self.mesh_y.reshape(1,1,size_y,1).repeat([batchsize,size_x,1,1])

        return th.cat((gridx, gridy), dim=-1).to(device)

    def forward(self, x):
        """
        input  [batchsize,nx,ny,10]: solution of previous 10 timesteps
        output [batchsize,nx,ny4,1]: solution for the next time step
        """

        # add uniform grid in x and y: [batchsize,nx,ny,10] --> [batchsize,nx,ny,12]
        grid = self.get_grid(x.shape, x.device)     # [batchsize,nx,ny,2]
        x = th.cat((x, grid), dim=-1)               # [batchsize,nx,ny,12]

        # lift to a higher dimension channel space by a NN (P)
        x = self.fc0(x)             # [batchsize,nx,ny,12]    --> [batchsize,nx,ny,width]
        x = x.permute(0,3,1,2)      # [batchsize,nx,ny,width] --> [batchsize,width,nx,ny]

        # pad the domain if input is non-periodic
        #x = F.pad(x, [self.padding,self.padding,self.padding,self.padding])   # [batchsize,width,nx+2*padding,ny+2*padding]

        # set BCs
        #x[:,:,0,:]  = -x[:,:,1,:]
        #x[:,:,-1,:] = -x[:,:,-2,:]
        #x[:,:,:,0]  = -x[:,:,:,1]
        #x[:,:,:,-1] = 2.0*1.0 - x[:,:,:,-2]

        # FFT->R->iFFT --> convolution --> addition --> activation sigma
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # FFT->R->iFFT --> convolution --> addition --> activation sigma
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # FFT->R->iFFT --> convolution --> addition --> activation sigma
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # FFT->R->iFFT --> convolution --> addition --> activation sigma
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # pad the domain if input is non-periodic
        #x = x[..., self.padding:-self.padding, self.padding:-self.padding] 

        # project back to target dimension by a NN (Q)
        x = x.permute(0,2,3,1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x