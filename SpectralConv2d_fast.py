# third-party libraries
import torch as th
import torch.nn as nn

class SpectralConv2d_fast(nn.Module):
    """
    Part of the Fourier layer that does FFT --> linear transform R --> inverse FFT.
    FFT can be used only for uniform discretization.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Initialization
            in_channels:    number of inputs channels
            out_channels:   number of output channels
            modes1, modes2: number of Fourier modes to apply (at most floor(N/2) + 1)
        """
        super(SpectralConv2d_fast, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1       = modes1
        self.modes2       = modes2

        # initialize weights
        self.scale = 1.0/(in_channels*out_channels)
        self.weights1 = nn.Parameter(self.scale*th.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=th.cfloat))
        self.weights2 = nn.Parameter(self.scale*th.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=th.cfloat))

    def compl_mul2d(self, input, weights):
        """
        Complex multiplication: [batch,in_channel,x,y]*[in_channel,out_channel,x,y] --> [batch,out_channel,x,y]
        """
        return th.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        FFT --> linear transform --> inverse FFT
        """
        # compute Fourier coeffcients up to factor of e^(-constant)
        x_ft = th.fft.rfft2(x)

        # multiply relevant Fourier modes
        out_ft = th.zeros(x.shape[0], self.out_channels, x.size(-2), x.size(-1)//2+1, dtype=th.cfloat, device=x.device)
        out_ft[:,:,:self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:,:,:self.modes1, :self.modes2], self.weights1)
        out_ft[:,:,-self.modes1:,:self.modes2] = self.compl_mul2d(x_ft[:,:,-self.modes1:,:self.modes2], self.weights2)

        # return to physical space
        x = th.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        return x