# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from channeltest import OFDM
from GDN import GDN

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

def deconv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding = 0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding = output_padding,bias=False)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv_block, self).__init__()
        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gdn = GDN(out_channels)
        self.prelu = nn.PReLU()
    def forward(self, x): 
        out = self.conv(x)
        out = self.gdn(out)
        out = self.prelu(out)
        return out

class deconv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding = 0):
        super(deconv_block, self).__init__()
        self.deconv = deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,  output_padding = output_padding)
        self.gdn = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, activate_func='prelu'): 
        out = self.deconv(x)
        out = self.gdn(out)
        if activate_func=='prelu':
            out = self.prelu(out)
        elif activate_func=='sigmoid':
            out = self.sigmoid(out)
        return out   
    
    
class conv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv1x1=False, kernel_size=3, stride=1, padding=1):
        super(conv_ResBlock, self).__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = conv(out_channels, out_channels, kernel_size=1, stride = 1, padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.use_conv1x1 = use_conv1x1
        if use_conv1x1 == True:
            self.conv3 = conv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
    def forward(self, x): 
        out = self.conv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.gdn2(out)
        if self.use_conv1x1 == True:
            x = self.conv3(x)
        out = out+x
        out = self.prelu(out)
        return out 
    
class deconv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_deconv1x1=False, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(deconv_ResBlock, self).__init__()
        self.deconv1 = deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.deconv2 = deconv(out_channels, out_channels, kernel_size=1, stride = 1, padding=0, output_padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_deconv1x1 = use_deconv1x1
        if use_deconv1x1 == True:
            self.deconv3 = deconv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=output_padding)
    def forward(self, x, activate_func='prelu'): 
        out = self.deconv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.deconv2(out)
        out = self.gdn2(out)
        if self.use_deconv1x1 == True:
            x = self.deconv3(x)
        out = out+x
        if activate_func=='prelu':
            out = self.prelu(out)
        elif activate_func=='sigmoid':
            out = self.sigmoid(out)
        return out 

def ratio2filtersize(x: torch.Tensor, ratio):
    if x.dim() == 4:
        before_size = torch.prod(torch.tensor(x.size()[1:]))
    elif x.dim() == 3:
        before_size = torch.prod(torch.tensor(x.size()))
    else:
        raise Exception('Unknown size of input')
    encoder_temp = _Encoder(is_temp=True)
    z_temp = encoder_temp(x)
    
    c = before_size * ratio / torch.prod(torch.tensor(z_temp.size()[-2:]))
    
    return int(c)



class _Encoder(nn.Module):
    def __init__(self, c=1, is_temp=False, P=1):
        """
        Args:
            c:       encoder输出通道数
            K:       有效子载波数
            is_temp: 是否为临时网络（用于 ratio2filtersize 的尺寸探测）
            P:       发送功率约束
            N:       IFFT 总点数（None 表示不进行 IFFT，直接输出频域特征）
        """
        super(_Encoder, self).__init__()
        self.is_temp = is_temp
        self.c=c

        self.conv1=conv_block(3,64,5,1,2)
        self.conv2=conv_block(64,128,5,2,2)
        self.conv3=conv_block(128,256,5,2,2)
        self.conv4=conv_ResBlock(256,256,kernel_size=5,stride=1,padding=2)
        self.conv5=conv_ResBlock(256,256,kernel_size=5,stride=1,padding=2)
        self.conv6=conv_block(256,2*c,5,1,2)

        self.norm = self._normalizationLayer(P=P)

        

    @staticmethod
    def _normalizationLayer(P=1):
        def _inner(z_hat: torch.Tensor):
            if z_hat.dim() == 4:
                batch_size = z_hat.size(0)
                k = torch.prod(torch.tensor(z_hat.size()[1:]))
            elif z_hat.dim() == 3:
                batch_size = 1
                k = torch.prod(torch.tensor(z_hat.size()))
            else:
                raise Exception('Unknown size of input')
            z_temp  = z_hat.reshape(batch_size, 1, 1, -1)
            z_trans = z_hat.reshape(batch_size, 1, -1, 1)
            tensor  = torch.sqrt(P * k) * z_hat / torch.sqrt(z_temp @ z_trans)
            return tensor.squeeze(0) if batch_size == 1 else tensor
        return _inner

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if self.is_temp:
            return x
        x = self.conv6(x)       # 频域特征 (Batch, 2*c, H, W)
        x = self.norm(x)        # 功率归一化
        return x


class _Decoder(nn.Module):
    def __init__(self, c=1):
        super(_Decoder, self).__init__()

        self.tconv1 = deconv_block(2*c,256,5,1,2)
        self.tconv2 = deconv_ResBlock(256,256,kernel_size=5,stride=1,padding=2)
        self.tconv3 = deconv_ResBlock(256,256,kernel_size=5,stride=1,padding=2)
        self.tconv4 = deconv_block(256,128,5,2,2,output_padding=1)
        self.tconv5 = deconv_block(128,64,5,2,2,output_padding=1)
        self.tconv6 = deconv_block(64,3,5,1,2)
        

    def forward(self, x):
        
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        x = self.tconv6(x,activate_func='sigmoid')
        return x



class DeepJSCC(nn.Module):
    def __init__(self, c,nfft=128,cp_len=16, snr=None, bindwitdh_ratio=None,cfd_std=0.1,
        pd_std=0.1,
        n_paths=10,
        timing_offset_range=(-3,3)):
        """
        Args:
            c:            压缩通道数
            channel_type: 信道类型
            snr:          信噪比
            br:            带宽占用比 (Bandwidth Ratio), 范围 (0, 1]。
            K:            有效子载波数
        """
        super(DeepJSCC, self).__init__()
        self.c = c
        self.br = bindwitdh_ratio
        self.N=nfft
        self.K=int(nfft*bindwitdh_ratio)
        self.cp_len=cp_len    
        self.snr=snr
        self.encoder = _Encoder(c=c)
        self.ofdm= OFDM(c=self.c,K=self.K,N=self.N,cp_len=self.cp_len,snr=self.snr,cfd_std=cfd_std,pd_std=pd_std,n_paths=n_paths,timing_offset_range=timing_offset_range)
        self.decoder = _Decoder(c=c)
        
    

    def forward(self, x):
        x = self.encoder(x)  # 输出频域特征 或 时域信号
        
 
        x,signal=self.ofdm(x)


        x_hat = self.decoder(x)
        
        return x_hat

    def change_channel(self, snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = OFDM(snr=snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None

    def loss(self, prd, gt):
        criterion = nn.MSELoss(reduction='mean')
        return criterion(prd, gt)



if __name__ == '__main__':
    
    x = torch.rand(2, 3, 32, 32)
    ratio=1/6
    c=ratio2filtersize(x,ratio)
    pilot_list=[3,10,17]
    
    model_time = DeepJSCC(c=c, cp_len=16,bindwitdh_ratio=0.25,snr=[12.0,20.0])
    
    print(f"当前时域模式的 IFFT 总点数 N = {model_time.N}")
    y = model_time(x)
    print("时域 OFDM 模式输出形状:", y.shape) 
