import torch
import torch.nn as nn
import math
import utils
from torch import vmap

fs=1920000
 
class OFDM(nn.Module):
    def __init__(self,c,K,N,cfd_std,
        pd_std,
        n_paths,
        timing_offset_range,cp_len=None,snr=None,
        ):
        super().__init__()
        self.c=c
        self.K=K
        self.N=N
        self.cp_len=cp_len
        self.snr=snr
        
        self.pilot_type='zc'
        self.pilot_power=1
        self.register_buffer("pilot", None)
        self.register_buffer("pilot_t", None)
        self.register_buffer("pilot_list", None)
        self.register_buffer("data_list", None)
        
        self.channel_estimation_method='ls_average'
        
        
        if snr is not None:
            self.channel=Channel(snr,cfd_std=cfd_std,pd_std=pd_std,n_paths=n_paths,timing_offset_range=timing_offset_range)
            
    
    
    def _insert_pilots(self, data_signal):
        """
        插入导频并自动注册固定导频位置
        Args:
            data_signal: 数据信号 (B, num_data_symbols, symbol_len)
        Returns:
            signal_with_pilots: 插入导频后的信号 (B, total_symbols, symbol_len)
        """
        batch, num_data_symbols, symbol_len = data_signal.shape
        device = data_signal.device
        
        if self.pilot is None:
            pilot=utils._generate_pilot(self.pilot_type,self.K,5,device,pilot_power=1)
            pilot_freq = pilot * self.pilot_power
            
            pilot_t_rmcp = utils.zeropaddingwithifft(pilot_freq, self.K, self.N)
            if self.cp_len is not None:
                pilot_t = utils.add_cp(pilot_t_rmcp, self.cp_len)
            else:
                pilot_t = pilot_t_rmcp
            pilot_list_python = utils.get_pilot_list(data_signal)
            pilot_list = torch.tensor(pilot_list_python, dtype=torch.long, device=device)
            
            num_pilots = len(pilot_list)
            total_symbols = 7*num_pilots
            all_indices = torch.arange(total_symbols, device=device)
            data_mask = torch.ones(total_symbols, dtype=torch.bool, device=device)
            data_mask[pilot_list] = False
            data_list = all_indices[data_mask][:num_data_symbols]
            
            self.pilot = pilot_freq
            self.pilot_t = pilot_t
            self.pilot_list = pilot_list
            self.data_list = data_list
                        
            print(f"✅ 导频信号生成完成，形状: {self.pilot.shape}")
            print(f"✅ 自动注册固定导频位置: {self.pilot_list.tolist()}")
            print(f"✅ 自动注册数据位置: {self.data_list.tolist()}")
        
        # 使用已注册的导频位置
        pilot_list = self.pilot_list
        data_list = self.data_list
        num_pilots = len(pilot_list)
        total_symbols = 7*num_pilots
        
        # 初始化输出张量
        signal_with_pilots = torch.zeros(
            batch, total_symbols, symbol_len,
            dtype=data_signal.dtype,
            device=device
        )
        
        # 插入导频和数据
        signal_with_pilots[:, pilot_list, :] = self.pilot_t.expand(batch, num_pilots, -1)
        signal_with_pilots[:, data_list, :] = data_signal
        
        return signal_with_pilots
    
    def _cfo_comp(self,pilot_list,x):
        B,total_symbols,M=x.shape
        device=x.device
        cp=self.cp_len
        
        p1_idx = pilot_list[0]
        p2_idx = pilot_list[1] if len(pilot_list) > 1 else pilot_list[0]
        
        rx_pilot_1 = x[:, p1_idx, :]  # (B, Nfft+cp)
        rx_pilot_2 = x[:, p2_idx, :]  # (B, Nfft+cp)
        
        z1=rx_pilot_1[:,cp:-cp].sum(dim=-1)
        z2=rx_pilot_2[:,cp:-cp].sum(dim=-1)
        
        corr=z2*torch.conj(z1)
        dphi = torch.angle(corr)
        
        symbol_duration = M / fs
        delta_f_est = dphi / (2 * math.pi * (p2_idx - p1_idx) * symbol_duration) # (B,)
        t = torch.arange(M*total_symbols, device=device).float()
        t = t.view(1, -1) 
        delta_f_est = delta_f_est.view(B, 1)
        comp_phase = torch.exp(-1j * 2 * math.pi * delta_f_est * t / fs)
        x=x.reshape(B, -1)  # (B, 总符号数 * N)
        
        x = x * comp_phase
        x=x.reshape(B, total_symbols, M)  # (B, 总符号数, N)
        
        return x
    
    def _ls_estimate_average(self, rx_pilot):
        """
        多导频平均LS信道估计（适用于时不变信道）
        Args:
            rx_pilot: 接收导频，形状 (B, num_pilots, K)
        Returns:
            H_est: 估计的信道响应，形状 (B, 1, K)
        """
        # 每个导频符号独立做LS估计
        H_per_pilot = rx_pilot / self.pilot  # (B, num_pilots, K)
        # 在导频维度取平均，降低噪声
        H_est = H_per_pilot.mean(dim=1, keepdim=True)  # (B, 1, K)
        return H_est
    

    
    def _ls_equalize(self, rx_data, H_est):
        """LS均衡"""
        return rx_data / (H_est + 1e-8)
    

    
    def _channel_estimation_and_equalization(self, rx_freq, noise_power):
        """
        统一的信道估计与均衡接口
        Args:
            rx_freq: 接收频域信号，形状 (B, total_symbols, K)
            noise_power: 噪声功率，形状 (B, 1, 1)
        Returns:
            rx_data_eq: 均衡后的数据，形状 (B, num_data_symbols, K)
            H_est: 估计的信道响应
        """
        # 分离导频和数据
        rx_pilot = rx_freq[:, self.pilot_list, :]  # (B, num_pilots, K)
        rx_data = rx_freq[:, self.data_list, :]     # (B, num_data_symbols, K)
        
        # 根据选择的方法进行信道估计
        if self.channel_estimation_method == "ls_average":
            H_est = self._ls_estimate_average(rx_pilot)
            rx_data_eq = self._ls_equalize(rx_data, H_est)
            
        return rx_data_eq, H_est
    
    def forward(self,x,rxsignal=None):
        
        #x频域向量（b,2c,h,w)转化成复向量（b,s,k)并zeropaddingifft转换成时域复向量（b,s,n)
        batch, c2, H, W = x.shape
        c=c2//2
        assert c==self.c,"nn通道数错误"
        K, N = self.K, self.N
        # 将实部和虚部组合为复数张量
        x_real = x[:, :c, :, :]   # (Batch, c, H, W)
        x_imag = x[:, c:, :, :]   # (Batch, c, H, W)
        x_complex = torch.complex(x_real, x_imag)  # (Batch, c, H, W)
        x_complex=x_complex.reshape(batch,-1,K)#(b,s,k)
        x=utils.zeropaddingwithifft(x_complex,K,N)
        
        #加cp
        if self.cp_len is not None:
            x=utils.add_cp(x,self.cp_len)
            
        b, s, m = x.shape
        device = x.device

        #加导频
        x=self._insert_pilots(x)
        
        signal=x


        #信道
        if hasattr(self, 'channel') and self.channel is not None:
            x,noise_power = self.channel(x)    
        
        if rxsignal is not None:
            x=rxsignal
            print('use rxsignal')
        # cfo补偿
        x=self._cfo_comp(self.pilot_list,x)
        
        #去cp
        x=utils.rm_cp(x,self.cp_len)#( b, s+npilot, n)
        
        #timecropfft 时频转换
        x=utils.timecropfft(x,self.K,self.N)
        
        #信道估计与均衡
        x,Hest=self._channel_estimation_and_equalization(x,noise_power)

        #转化为decoder接收的特征图
        x = x.reshape(b, c, H, W)
        x = torch.cat([x.real, x.imag], dim=1)
        
        return x,signal
        

class Channel(nn.Module):
    def __init__(
        self,
        snr=None,
        cfd_std=0.015,
        pd_std=0.1,
        n_paths=10,
        timing_offset_range=(-3,3),
        normalize_power=True,
        
    ):
        super().__init__()
        self.snr = snr
        self.cfd_std = cfd_std
        self.pd_std = pd_std
        self.n_paths = n_paths
        self.timing_offset_range = timing_offset_range
        

    """ def _normalize(self, x):
        p = (x.abs() ** 2).mean(dim=(-1, -2), keepdim=True)
        return x / torch.sqrt(p + 1e-8) """

    

    def _multipath(self,x):#无大尺度衰减
        if self.n_paths <= 1 :
            return x

        B, S, N = x.shape
        x=x.reshape(B,-1)
        device = x.device
        n_paths=self.n_paths

        delays = torch.arange(self.n_paths, device=device)
        tau = max(delays.max().item() / 3.0, 1.0)
        power = torch.exp(-delays.float() / tau)
        power = power / power.sum()

        h_real = torch.randn(B, n_paths, device=device)
        h_imag = torch.randn(B, n_paths, device=device)
        h = torch.complex(h_real, h_imag)
        h = h * torch.sqrt(power).view(1, -1)
        h = h / torch.sqrt((h.abs() ** 2).sum(dim=1, keepdim=True) + 1e-8)

        y = torch.zeros_like(x)
        for p in range(n_paths):
            d = int(delays[p].item())
            if d > 0:
                xp = torch.zeros_like(x)
                xp[:, d:] = x[:, :-d]
            else:
                xp = x
            y = y + h[:, p].view(B, 1) * xp
        y=y.reshape(B,S,N)

        return y
    

    def _timing_offset(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (B, S, N)")

        if not torch.is_complex(x):
            x = x.to(torch.complex64)

        B, S, N = x.shape
        device = x.device
        low, high = self.timing_offset_range

        shifts = torch.randint(low, high + 1, (B,), device=device)

        # 2. 展平为 (B, L) 并分离实部虚部
        x_r = x.real.flatten(1)
        x_i = x.imag.flatten(1)

        # ===================== 🔥 调用新的移位/外推函数 =====================
        y_r = utils._shift_linear_extrapolate_batch(x_r, shifts)
        y_i = utils._shift_linear_extrapolate_batch(x_i, shifts)

        # 恢复形状
        y_r = y_r.view(B, S, N)
        y_i = y_i.view(B, S, N)

        return torch.complex(y_r, y_i)
    
    def _cfo(self, x):
        if self.cfd_std <= 0:
            return x
        B, S, N = x.shape
        x=x.reshape(B,-1)
        B,N=x.shape
        device = x.device
        t = torch.arange(N, device=device).float().view(1, N)
        delta = torch.empty(B, 1,  device=device).uniform_(-self.cfd_std, self.cfd_std)*fs
        phase = 2 * math.pi * delta * t/fs #delta是deltaf
        
        x= x * torch.exp(1j * phase)
        x=x.reshape(B,S,-1)
        return x

    def _po(self, x):
        if self.pd_std <= 0:
            return x
        B, S, N = x.shape
        device = x.device
        phi0 = torch.empty(B, 1, 1, device=device).uniform_(-self.pd_std, self.pd_std)
        return x * torch.exp(1j * phi0)

    def _awgn(self, x):
        # 假设你的类初始化参数已从 self.snr 改为 self.snr_range = (low, high)
        if self.snr is None:
            return x

        B = x.shape[0]  # 获取 Batch size
        device = x.device
        low, high = self.snr

        # 1. 为每个 Batch 样本随机生成一个 SNR (dB)
        # 形状: (B,) -> 后续会广播为 (B, 1, 1)
        snr_db = low + (high - low) * torch.rand(B, device=device)
        
        # 2. 转换为线性域
        # 增加维度以便广播: (B,) -> (B, 1, 1)
        snr_lin = 10 ** (snr_db / 10.0)
        snr_lin = snr_lin.view(B, 1, 1) 

        # 3. 计算每个样本的信号功率
        # 结果形状: (B, 1, 1)，可以直接与 snr_lin 运算
        sig_power = (x.abs() ** 2).mean(dim=(-1, -2), keepdim=True)

        # 4. 计算每个样本对应的噪声功率
        noise_power = sig_power / snr_lin

        # 5. 生成高斯白噪声 (实部虚部独立)
        # torch.randn_like 已经包含了广播逻辑
        noise = torch.sqrt(noise_power / 2) * (
            torch.randn_like(x.real) + 1j * torch.randn_like(x.real)
        )

        return x + noise, noise_power

    def forward(self, x):
        if not torch.is_complex(x):
            raise TypeError("Channel expects complex input of shape (B, S, N).")

        

        x = self._multipath(x)
        x = self._timing_offset(x)
        x = self._cfo(x)
        x = self._po(x)
        x,noise_power = self._awgn(x)
        
        return x,noise_power
    
if __name__=='__main__' :
    device = "cuda" if torch.cuda.is_available() else "cpu"

    c = 4
    K = 32
    H = 8
    W = 8


    N = 128
    cp_len = 16
    batch = 16

    model = OFDM(
        c=c,
        K=K,
        N=N,
        cfd_std=0.015,
        pd_std=0.0,
        n_paths=8,
        timing_offset_range=(-1, -1),
        cp_len=cp_len,
        snr=(25, 25),
    ).to(device)

    x = torch.randn(batch, 2*c, H, W, device=device)
    y = model(x)

    mse_out = ((x - y)**2).mean().item()
    print(f"[LS TEST] output MSE = {mse_out:.6e}")