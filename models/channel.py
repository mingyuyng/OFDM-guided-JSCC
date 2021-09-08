

from scipy.linalg import dft
from scipy.linalg import toeplitz
import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
import math

PI = math.pi

class BatchConv1DLayer(nn.Module):
    def __init__(self, stride=1,
                 padding=0, dilation=1):
        super(BatchConv1DLayer, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x, weight, bias=None):
        if bias is None:
            assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
        else:
            assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
                0], "dim=0 of bias must be equal in size to dim=0 of weight"

        b_i, b_j, c, h = x.shape
        b_i, out_channels, in_channels, kernel_width_size = weight.shape

        out = x.permute([1, 0, 2, 3]).contiguous().view(b_j, b_i * c, h)
        weight = weight.view(b_i * out_channels, in_channels, kernel_width_size)

        out = F.conv1d(out, weight=weight, bias=None, stride=self.stride, dilation=self.dilation, groups=b_i,
                       padding=self.padding)

        out = out.view(b_j, b_i, out_channels, out.shape[-1])

        out = out.permute([1, 0, 2, 3])

        if bias is not None:
            out = out + bias.unsqueeze(1).unsqueeze(3)

        return out


def Normalize(x, pwr=1):
    '''
    Normalization function
    '''
    power = torch.mean(x**2, (-2,-1), True)
    alpha = np.sqrt(pwr/2)/torch.sqrt(power)
    return alpha*x

class Clipping(nn.Module):
    '''
    Simulating the Clipping effect
    ''' 
    def __init__(self, opt):
        super(Clipping, self).__init__()
        self.CR = opt.CR  # Clipping ratio
    	
    def forward(self, x):
        # Calculating the scale vector for each element  

        amp = torch.sqrt(torch.sum(x**2, -1, True))
        sigma = torch.sqrt(torch.mean(x**2, (-2,-1), True) * 2)
        ratio = sigma*self.CR/amp
        scale = torch.min(ratio, torch.ones_like(ratio))

        with torch.no_grad():
            bias = x*scale - x

        return x + bias


class Add_CP(nn.Module): 
    '''
    Add cyclic prefix 
    '''
    def __init__(self, opt):
        super(Add_CP, self).__init__()
        self.opt = opt
    def forward(self, x):
        return torch.cat((x[...,-self.opt.K:,:], x), dim=-2)

class RM_CP(nn.Module):
    '''
    Remove cyclic prefix
    ''' 
    def __init__(self, opt):
        super(RM_CP, self).__init__()
        self.opt = opt
    def forward(self, x):
        return x[...,self.opt.K:, :]

class Add_CFO(nn.Module): 
    '''
    Simulating the CFO effect in baseband
    Ang: unit: (degree/sample)
    '''
    def __init__(self, opt):
        super(Add_CFO, self).__init__()
        self.opt = opt
    def forward(self, input):
        # Input size:  NxPxSx(M+K)x2
        N = input.shape[0]     # Input batch size

        if self.opt.is_cfo_random:
            angs = (torch.rand(N)*2-1)*self.opt.max_ang
        else:
            angs = torch.ones(N)*self.opt.ang 

        if self.opt.is_trick:
            index = torch.arange(-self.opt.K, self.opt.M+self.opt.N_pilot).float()
            angs_all = torch.ger(angs, index).repeat((1,self.opt.S+1)).view(N, self.opt.S+1, self.opt.M+self.opt.N_pilot+self.opt.K)    # Nx(S+1)x(M+K)
        else:
            index = torch.arange(0, (self.opt.S+1)*(self.opt.M+self.opt.N_pilot+self.opt.K)).float()
            angs_all = torch.ger(angs, index).view(N, self.opt.S+1, self.opt.M+self.opt.N_pilot+self.opt.K)    # Nx(S+1)x(M+K)

        real = torch.cos(angs_all/360*2*PI).unsqueeze(1).unsqueeze(-1)   # Nx1xSx(M+K)x1 
        imag = torch.sin(angs_all/360*2*PI).unsqueeze(1).unsqueeze(-1)   # Nx1xSx(M+K)x1

        real_in = input[...,0].unsqueeze(-1)    # NxPx(Sx(M+K))x1 
        imag_in = input[...,1].unsqueeze(-1)    # NxPx(Sx(M+K))x1

        # Perform complex multiplication
        real_out = real*real_in - imag*imag_in
        imag_out = real*imag_in + imag*real_in

        return torch.cat((real_out, imag_out), dim=4) 


class Channel(nn.Module):
    '''
    Realization of passing multi-path channel

    '''
    def __init__(self, opt, device):
        super(Channel, self).__init__()

        # Assign the power delay spectrum
        self.opt = opt
        SMK = (self.opt.S+1)*(self.opt.M+self.opt.N_pilot+self.opt.K)

        # Generate unit power profile
        power = torch.exp(-torch.arange(opt.L).float()/opt.decay).unsqueeze(0).unsqueeze(0).unsqueeze(3)  # 1x1xLx1
        self.power = power/torch.sum(power)
        self.device = device

        self.bconv1d = BatchConv1DLayer(padding=opt.L-1)  

    def sample(self, N, P, M, L):
        # Sample the channel coefficients
        cof = torch.sqrt(self.power/2) * torch.randn(N, P, L, 2)
        cof_true = torch.cat((cof, torch.zeros((N,P,M-L,2))), 2)
        H_true = torch.fft(cof_true, 1)

        return cof, H_true

    def forward(self, input, cof=None, def_index=True):
        # Input size:   NxPx(Sx(M+K))x2
        # Output size:  NxPx(L+Sx(M+K)-1)x2
        # Also return the true channel
        # Generate Channel Matrix

        N, P, SMK, _ = input.shape
        
        if cof is None:
            cof = torch.sqrt(self.power/2) * torch.randn(N, P, self.opt.L, 2)       # NxPxLx2

        cof_true = torch.cat((cof, torch.zeros((N,P,self.opt.M-self.opt.L,2))), 2)  
        H_true = torch.fft(cof_true, 1)  # NxPxLx2

        signal_real = input[...,0].view(N*P, 1, 1, -1)       # (NxP)x(Sx(M+K))x1
        signal_imag = input[...,1].view(N*P, 1, 1, -1)       # (NxP)x(Sx(M+K))x1

        ind = torch.linspace(self.opt.L-1, 0, self.opt.L).long()

        cof_real = cof[...,0][...,ind].view(N*P, 1, 1, -1).to(self.device) 
        cof_imag = cof[...,1][...,ind].view(N*P, 1, 1, -1).to(self.device)

        output_real = self.bconv1d(signal_real, cof_real) - self.bconv1d(signal_imag, cof_imag)   # (NxP)x(L+SMK-1)x1
        output_imag = self.bconv1d(signal_real, cof_imag) + self.bconv1d(signal_imag, cof_real)   # (NxP)x(L+SMK-1)x1

        output = torch.cat((output_real.view(N*P,-1,1), output_imag.view(N*P,-1,1)), -1)   # (NxP)x(L+SMK-1)x2

        return output.view(N,P,self.opt.L+SMK-1,2), H_true


def complex_division(no, de):
    a = no[...,0]
    b = no[...,1]
    c = de[...,0]
    d = de[...,1]

    out_real = (a*c+b*d)/(c**2+d**2)
    out_imag = (b*c-a*d)/(c**2+d**2)

    return torch.cat((out_real.unsqueeze(-1), out_imag.unsqueeze(-1)),-1)

def complex_multiplication(x1, x2):
    real1 = x1[...,0]
    imag1 = x1[...,1]
    real2 = x2[...,0]
    imag2 = x2[...,1]

    out_real = real1*real2 - imag1*imag2
    out_imag = real1*imag2 + imag1*real2

    return torch.cat((out_real.unsqueeze(-1), out_imag.unsqueeze(-1)),-1)

def complex_conjugate(x):
    out_real = x[...,0]
    out_imag = -x[...,1]
    return torch.cat((out_real.unsqueeze(-1), out_imag.unsqueeze(-1)),-1)

def complex_amp(x):
    real = x[...,0]
    imag = x[...,1]
    return torch.sqrt(real**2 + imag**2).unsqueeze(-1)

def ZadoffChu(order, length, index=0):
    cf = length % 2
    n = np.arange(length)
    arg = np.pi*order*n*(n+cf+2*index)/length
    zado = np.exp(-1j*arg)
    zado_real = torch.from_numpy(zado.real).unsqueeze(-1).float()
    zado_imag = torch.from_numpy(zado.imag).unsqueeze(-1).float()
    return torch.cat((zado_real, zado_imag), 1)

def MMSE_equalization(H_est, Y, noise_pwr):
    # H_est: NxPx1xMx2
    # Y: NxPxSxMx2  
    no = complex_multiplication(Y, complex_conjugate(H_est))
    de = complex_amp(H_est)**2 + noise_pwr.unsqueeze(-1) 
    return no/de

def LMMSE_channel_est(pilot_tx, pilot_rx, noise_pwr):
    # pilot_tx: NxPx1xMx2
    # pilot_rx: NxPxS'xMx2
    no = complex_multiplication(torch.mean(pilot_rx, 2, True), complex_conjugate(pilot_tx))
    de = 1+noise_pwr.unsqueeze(-1)/pilot_rx.shape[2]
    return no/de



class OFDM_channel(nn.Module):
    '''
    SImulating the end-to-end OFDM system with non-linearity
    '''
    def __init__(self, opt, device, pwr = 1):
        super(OFDM_channel, self).__init__()
        self.opt = opt

        # Setup the add & remove CP layers
        self.add_cp = Add_CP(opt)
        self.rm_cp = RM_CP(opt)

        # Setup the channel layer
        self.channel = Channel(opt, device)
        self.clip = Clipping(opt)

        # Generate the pilot signal
        if not os.path.exists('pilot.pt'):
            pilot = ZadoffChu(order=1, length=opt.M)
        else:
            pilot = torch.load('pilot.pt').squeeze()
        
        self.pilot = Normalize(pilot, pwr=pwr)
        self.pilot = self.pilot.to(device)
        self.pilot_cp = self.add_cp(torch.ifft(self.pilot,1)).repeat(opt.P, opt.N_pilot,1,1)         #1xMx2  => PxS'x(M+K)x2

        self.pwr = pwr

    def sample(self, N):
        return self.channel.sample(N, self.opt.P, self.opt.M, self.opt.L)

    def PAPR(self, x):
        power = torch.mean(x**2, (-2,-1))*2
        max_pwr, _ = torch.max(torch.sum(x**2, -1), -1)
        return max_pwr/power

    def forward(self, x, SNR, cof=None):
        # Input size: NxPxSxMx2   The information to be transmitted
        # cof denotes given channel coefficients
        N = x.shape[0]

        # Normalize the input power in frequency domain
        x = Normalize(x, pwr=self.pwr)
        
        # IFFT:                    NxPxSxMx2  => NxPxSxMx2
        x = torch.ifft(x, 1)

        # Add Cyclic Prefix:       NxPxSxMx2  => NxPxSx(M+K)x2
        x = self.add_cp(x)
        
        # Reshape:
        x = x.view(N, self.opt.P, self.opt.S*(self.opt.M+self.opt.K), 2)
        pilot = self.pilot_cp.repeat(N,1,1,1,1).view(N, self.opt.P, self.opt.N_pilot*(self.opt.M+self.opt.K), 2)
        
        # Signal clipping (optional)       
        if self.opt.is_clip:
            with torch.no_grad():
                pwr_pre = torch.mean(x**2, (-2,-1), True) * 2
            x = self.clip(x)
            with torch.no_grad():
                pwr = torch.mean(x**2, (-2,-1), True) * 2
                alpha = torch.sqrt(pwr_pre/2)/torch.sqrt(pwr/2)
            x = alpha*x            
        PAPR = self.PAPR(x)
            
        # Add pilot:               NxPxSx(M+K)x2  => NxPx(S+1)x(M+K)x2
        x = torch.cat((pilot, x), 2)    
        
        # Pass the Channel:        NxPx(S+1)(M+K)x2  =>  NxPx((S+1)(M+K)+L-1)x2
        y, H_true = self.channel(x, cof)
        
        # Calculate the power of received signal
        # 'ins': instantaeous noise calculated at the receiver
        # 'avg': average noise calculated at the transmitter
        with torch.no_grad(): 
            if self.opt.SNR_cal == 'ins':    
                pwr = torch.mean(y**2, (-2,-1), True) * 2
                noise_pwr = pwr*10**(-SNR/10)
            elif self.opt.SNR_cal == 'avg':
                pwr = torch.mean(y**2, (-2,-1), True) * 2
                noise_pwr = self.pwr*10**(-SNR/10)/self.opt.M
                noise_pwr = noise_pwr * torch.ones_like(pwr)

        # Generate random noise
        noise = torch.sqrt(noise_pwr/2) * torch.randn_like(y)
        y_noisy = y + noise

        # Peak Detection: (Perfect)    NxPx((S+S')(M+K)+L-1)x2  =>  NxPx(S+S')x(M+K)x2
        output = y_noisy[:,:,:(self.opt.S+self.opt.N_pilot)*(self.opt.M+self.opt.K),:].view(N, self.opt.P, self.opt.S+self.opt.N_pilot, self.opt.M+self.opt.K, 2)

        y_pilot = output[:,:,:self.opt.N_pilot,:,:]         # NxPxS'x(M+K)x2
        y_sig = output[:,:,self.opt.N_pilot:,:,:]           # NxPxSx(M+K)x2

        # Remove Cyclic Prefix":   
        info_pilot = self.rm_cp(y_pilot)    # NxPxS'xMx2
        info_sig = self.rm_cp(y_sig)        # NxPxSxMx2

        # FFT:                     
        info_pilot = torch.fft(info_pilot, 1)
        info_sig = torch.fft(info_sig, 1)

        return info_pilot, info_sig, H_true, noise_pwr, PAPR



