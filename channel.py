import torch
import torch.nn as nn
import math
import utils

fs = 1920000
 
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
        self.pilot_power=1.0
        self.register_buffer("pilot", None) # 频域导频
        self.register_buffer("pilot_t", None)
        self.register_buffer("pilot_list", None)
        self.register_buffer("data_list", None)
        self.register_buffer("R_H", None) # 修复：添加MMSE需要的协方差矩阵buffer
        
        self.channel_estimation_method='ls_average'
        
        if snr is not None:
            self.channel=Channel(snr,cfd_std=cfd_std,pd_std=pd_std,n_paths=n_paths,timing_offset_range=timing_offset_range)
            
    def _insert_pilots(self, data_signal):
        batch, num_data_symbols, symbol_len = data_signal.shape
        device = data_signal.device
        
        if self.pilot is None:
            print(f"🔧 正在生成{self.pilot_type.upper()}导频信号和位置...")
            # 修复1：变量名错误
            pilot_freq = utils._generate_pilot(self.pilot_type, self.K, 3, device, pilot_power=self.pilot_power)
            
            pilot_t_rmcp = utils.zeropaddingwithifft(pilot_freq, self.K, self.N)
            if self.cp_len is not None:
                pilot_t = utils.add_cp(pilot_t_rmcp, self.cp_len)
            else:
                pilot_t = pilot_t_rmcp
            
            pilot_list_python = utils.get_pilot_list(data_signal)
            pilot_list = torch.tensor(pilot_list_python, dtype=torch.long, device=device)
            
            # 修复2：使用刚生成的pilot_list而不是self.pilot_list
            num_pilots = len(pilot_list)
            total_symbols = max(num_data_symbols + num_pilots, pilot_list[-1].item() + 1)
            all_indices = torch.arange(total_symbols, device=device)
            data_mask = torch.ones(total_symbols, dtype=torch.bool, device=device)
            data_mask[pilot_list] = False
            data_list = all_indices[data_mask][:num_data_symbols]
            
            self.register_buffer("pilot", pilot_freq)
            self.register_buffer("pilot_t", pilot_t)
            self.register_buffer("pilot_list", pilot_list)
            self.register_buffer("data_list", data_list)
            
            print(f"✅ 导频信号形状: {self.pilot.shape}")
            print(f"✅ 导频位置: {self.pilot_list.tolist()}")
            print(f"✅ 数据位置: {self.data_list.tolist()}")
        
        pilot_list = self.pilot_list
        data_list = self.data_list
        num_pilots = len(pilot_list)
        total_symbols = max(num_data_symbols + num_pilots, pilot_list[-1].item() + 1)
        
        signal_with_pilots = torch.zeros(
            batch, total_symbols, symbol_len,
            dtype=data_signal.dtype,
            device=device
        )
        
        signal_with_pilots[:, pilot_list, :] = self.pilot_t.expand(batch, num_pilots, -1)
        signal_with_pilots[:, data_list, :] = data_signal
        
        return signal_with_pilots
    
    def _cfo_comp(self, pilot_list, x):
        B, total_symbols, M = x.shape
        device = x.device
        cp = self.cp_len
        
        p1_idx = pilot_list[0]
        p2_idx = pilot_list[1] if len(pilot_list) > 1 else pilot_list[0]
        
        rx_pilot_1 = x[:, p1_idx, :]
        rx_pilot_2 = x[:, p2_idx, :]
        
        z1 = rx_pilot_1[:, cp:-cp].sum(dim=-1)
        z2 = rx_pilot_2[:, cp:-cp].sum(dim=-1)
        
        corr = z2 * torch.conj(z1)
        dphi = torch.angle(corr)
        
        symbol_duration = M / fs
        delta_f_est = dphi / (2 * math.pi * (p2_idx - p1_idx) * symbol_duration)
        
        t = torch.arange(M * total_symbols, device=device).float().view(1, -1)
        delta_f_est = delta_f_est.view(B, 1)
        comp_phase = torch.exp(-1j * 2 * math.pi * delta_f_est * t / fs)
        
        x = x.reshape(B, -1) * comp_phase
        return x.reshape(B, total_symbols, M)
    
    def _ls_estimate_average(self, rx_pilot):
        H_per_pilot = rx_pilot / self.pilot
        H_est = H_per_pilot.mean(dim=1, keepdim=True)
        return H_est
    
    def _ls_estimate_linear_interp(self, rx_pilot, num_total_symbols):
        B, num_pilots, K = rx_pilot.shape
        device = rx_pilot.device
        
        H_pilot = rx_pilot / self.pilot
        H_est = torch.zeros(B, num_total_symbols, K, dtype=torch.complex64, device=device)
        H_est[:, self.pilot_list, :] = H_pilot
        
        for k in range(K):
            H_est[:, :, k] = torch.interp(
                torch.arange(num_total_symbols, device=device),
                self.pilot_list.float(),
                H_pilot[:, :, k].float()
            ).to(torch.complex64)
        
        H_est_data = H_est[:, self.data_list, :]
        return H_est_data
    
    def _mmse_estimate(self, rx_pilot, noise_power):
        H_ls_avg = (rx_pilot / self.pilot).mean(dim=1, keepdim=True)
        
        if self.R_H is None:
            print("🔧 正在估计信道协方差矩阵...")
            H_ls_all = rx_pilot / self.pilot
            H_centered = H_ls_all - H_ls_all.mean(dim=(0, 1), keepdim=True)
            self.R_H = torch.einsum('bnk,bnl->kl', H_centered, H_centered.conj()) / (rx_pilot.shape[0] * rx_pilot.shape[1] - 1)
            self.R_H = self.R_H + 1e-6 * torch.eye(self.K, device=self.R_H.device)
        
        pilot_power = (self.pilot.abs() ** 2).mean()
        W = self.R_H @ torch.linalg.inv(self.R_H + (noise_power / pilot_power) * torch.eye(self.K, device=rx_pilot.device))
        
        H_mmse = torch.einsum('kl,bnl->bnk', W, H_ls_avg)
        return H_mmse
    
    def _ls_equalize(self, rx_data, H_est):
        return rx_data / (H_est + 1e-8)
    
    def _mmse_equalize(self, rx_data, H_est, noise_power, data_power=1.0):
        H_conj = H_est.conj()
        H_sq = (H_est.abs() ** 2)
        W = H_conj / (H_sq + noise_power / data_power + 1e-8)
        return rx_data * W
    
    def _channel_estimation_and_equalization(self, rx_freq, noise_power):
        rx_pilot = rx_freq[:, self.pilot_list, :]
        rx_data = rx_freq[:, self.data_list, :]
        
        if self.channel_estimation_method == "ls_average":
            H_est = self._ls_estimate_average(rx_pilot)
            rx_data_eq = self._ls_equalize(rx_data, H_est)
        
        elif self.channel_estimation_method == "ls_interp":
            H_est = self._ls_estimate_linear_interp(rx_pilot, rx_freq.shape[1])
            rx_data_eq = self._ls_equalize(rx_data, H_est)
        
        elif self.channel_estimation_method == "mmse":
            H_est = self._mmse_estimate(rx_pilot, noise_power)
            rx_data_eq = self._mmse_equalize(rx_data, H_est, noise_power)
        
        else:
            raise ValueError(f"不支持的信道估计方法: {self.channel_estimation_method}")
        
        return rx_data_eq, H_est
    
    def forward(self, x):
        batch, c2, H, W = x.shape
        c = c2 // 2
        assert c == self.c, f"通道数不匹配：预期 {2*self.c}，实际 {c2}"
        K, N = self.K, self.N
        
        x_real = x[:, :c, :, :]
        x_imag = x[:, c:, :, :]
        x_complex = torch.complex(x_real, x_imag).reshape(batch, -1, K)
        
        x_time = utils.zeropaddingwithifft(x_complex, K, N)
        
        if self.cp_len is not None:
            x_time_with_cp = utils.add_cp(x_time, self.cp_len)
        else:
            x_time_with_cp = x_time
        
        signal_with_pilots = self._insert_pilots(x_time_with_cp)
        
        if hasattr(self, 'channel') and self.channel is not None:
            rx_signal, noise_power = self.channel(signal_with_pilots)
        else:
            rx_signal = signal_with_pilots
            noise_power = None
        
        # 保存原始信号用于对比
        self.original_signal = x_complex.detach()
        
        # CFO补偿
        rx_signal_compensated = self._cfo_comp(self.pilot_list, rx_signal)
        
        # 去CP
        rx_signal_rmcp = utils.rm_cp(rx_signal_compensated, self.cp_len)
        
        # 时频转换
        rx_freq = utils.timecropfft(rx_signal_rmcp, self.K, self.N)
        
        # 修复3：调用正确的信道估计方法并传入noise_power
        rx_data_eq, H_est = self._channel_estimation_and_equalization(rx_freq, noise_power)
        
        # 保存中间结果用于测试
        self.rx_data_before_eq = rx_freq[:, self.data_list, :].detach()
        self.rx_data_after_eq = rx_data_eq.detach()
        self.H_est = H_est.detach()
        
        # 转换回特征图
        x_out = rx_data_eq.reshape(batch, c, H, W)
        x_out = torch.cat([x_out.real, x_out.imag], dim=1)
        
        return x_out
        

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

    def _multipath(self,x):
        if self.n_paths <= 1 :
            return x

        B, S, N = x.shape
        x_flat = x.reshape(B,-1)
        device = x.device
        n_paths = self.n_paths

        delays = torch.arange(self.n_paths, device=device)
        tau = max(delays.max().item() / 3.0, 1.0)
        power = torch.exp(-delays.float() / tau)
        power = power / power.sum()

        h_real = torch.randn(B, n_paths, device=device)
        h_imag = torch.randn(B, n_paths, device=device)
        h = torch.complex(h_real, h_imag)
        h = h * power.view(1, -1)
        h = h / torch.sqrt((h.abs() ** 2).sum(dim=1, keepdim=True) + 1e-8)

        y = torch.zeros_like(x_flat)
        for p in range(n_paths):
            d = int(delays[p].item())
            if d > 0:
                xp = torch.zeros_like(x_flat)
                xp[:, d:] = x_flat[:, :-d]
            else:
                xp = x_flat
            y = y + h[:, p].view(B, 1) * xp
        y = y.reshape(B,S,N)

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

        x_r = x.real.flatten(1)
        x_i = x.imag.flatten(1)

        # 简化的移位函数（如果你的utils有这个函数可以替换）
        def _shift_linear_extrapolate_batch(x, shifts):
            B, L = x.shape
            y = torch.zeros_like(x)
            for i in range(B):
                s = shifts[i].item()
                if s > 0:
                    y[i, s:] = x[i, :-s]
                    y[i, :s] = x[i, 0]
                elif s < 0:
                    y[i, :s] = x[i, -s:]
                    y[i, s:] = x[i, -1]
                else:
                    y[i] = x[i]
            return y

        y_r = _shift_linear_extrapolate_batch(x_r, shifts)
        y_i = _shift_linear_extrapolate_batch(x_i, shifts)

        y_r = y_r.view(B, S, N)
        y_i = y_i.view(B, S, N)

        return torch.complex(y_r, y_i)
    
    def _cfo(self, x):
        if self.cfd_std <= 0:
            return x
        B, S, N = x.shape
        x_flat = x.reshape(B,-1)
        device = x.device
        t = torch.arange(x_flat.shape[1], device=device).float().view(1, -1)
        delta = torch.empty(B, 1, device=device).uniform_(-self.cfd_std, self.cfd_std)*fs
        phase = 2 * math.pi * delta * t / fs
        
        x_flat = x_flat * torch.exp(1j * phase)
        x = x_flat.reshape(B,S,-1)
        return x

    def _po(self, x):
        if self.pd_std <= 0:
            return x
        B, S, N = x.shape
        device = x.device
        phi0 = torch.empty(B, 1, 1, device=device).uniform_(-self.pd_std, self.pd_std)
        return x * torch.exp(1j * phi0)

    def _awgn(self, x):
        if self.snr is None:
            return x, None

        B = x.shape[0]
        device = x.device
        low, high = self.snr

        snr_db = low + (high - low) * torch.rand(B, device=device)
        snr_lin = 10 ** (snr_db / 10.0).view(B, 1, 1)
        
        sig_power = (x.abs() ** 2).mean(dim=(-1, -2), keepdim=True)
        noise_power = sig_power / snr_lin

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
        x, noise_power = self._awgn(x)
        
        return x, noise_power
    
    
    
# 固定随机种子保证可复现
torch.manual_seed(42)

# 测试参数
BATCH_SIZE = 1
C = 4          # 复数通道数
K = 32          # 有效子载波数
N = 128         # IFFT点数
CP_LEN = 16     # 循环前缀长度
SNR_RANGE = (12, 30)  # 固定10dB SNR
CFD_STD = 0.015 # 载波频偏标准差（约19.2kHz）
PD_STD = 0.01   # 关闭相位偏移
N_PATHS = 8    # 3条多径
TIMING_RANGE = (-3, 3) # 定时偏移范围

# 创建OFDM模型
ofdm = OFDM(
    c=C,
    K=K,
    N=N,
    cfd_std=CFD_STD,
    pd_std=PD_STD,
    n_paths=N_PATHS,
    timing_offset_range=TIMING_RANGE,
    cp_len=CP_LEN,
    snr=SNR_RANGE
)

# 生成你要求的输入张量 (1, 8, 8, 8)
# 注意：输入是实部和虚部拼接，所以通道数是2*C=2
input_tensor = torch.randn(BATCH_SIZE, 2*C, 8, 8)
print(f"输入张量形状: {input_tensor.shape}")

# 测试1：LS平均均衡 + 有CFO补偿
print("\n" + "="*50)
print("测试1：LS平均均衡 + 有CFO补偿")
print("="*50)
ofdm.channel_estimation_method = "ls_average"
output1 = ofdm(input_tensor)

# 计算MSE
original = ofdm.original_signal
before_eq = ofdm.rx_data_before_eq
after_eq = ofdm.rx_data_after_eq

mse_before_eq = torch.mean(torch.abs(original - before_eq) ** 2).item()
mse_after_eq = torch.mean(torch.abs(original - after_eq) ** 2).item()

print(f"均衡前MSE: {mse_before_eq:.4f}")
print(f"均衡后MSE: {mse_after_eq:.4f}")
print(f"均衡增益: {10 * math.log10(mse_before_eq / mse_after_eq):.2f} dB")

# 测试2：LS平均均衡 + 无CFO补偿
print("\n" + "="*50)
print("测试2：LS平均均衡 + 无CFO补偿")
print("="*50)
# 临时禁用CFO补偿
original_cfo_comp = ofdm._cfo_comp
ofdm._cfo_comp = lambda pilot_list, x: x

output2 = ofdm(input_tensor)

original = ofdm.original_signal
before_eq = ofdm.rx_data_before_eq
after_eq = ofdm.rx_data_after_eq

mse_before_eq_no_cfo = torch.mean(torch.abs(original - before_eq) ** 2).item()
mse_after_eq_no_cfo = torch.mean(torch.abs(original - after_eq) ** 2).item()

print(f"均衡前MSE: {mse_before_eq_no_cfo:.4f}")
print(f"均衡后MSE: {mse_after_eq_no_cfo:.4f}")
print(f"均衡增益: {10 * math.log10(mse_before_eq_no_cfo / mse_after_eq_no_cfo):.2f} dB")

# 恢复CFO补偿
ofdm._cfo_comp = original_cfo_comp

# # 测试3：MMSE均衡 + 有CFO补偿
# print("\n" + "="*50)
# print("测试3：MMSE均衡 + 有CFO补偿")
# print("="*50)
# ofdm.channel_estimation_method = "mmse"
# # 重置协方差矩阵
# ofdm.R_H = None

# output3 = ofdm(input_tensor)

# original = ofdm.original_signal
# before_eq = ofdm.rx_data_before_eq
# after_eq = ofdm.rx_data_after_eq

# mse_before_eq_mmse = torch.mean(torch.abs(original - before_eq) ** 2).item()
# mse_after_eq_mmse = torch.mean(torch.abs(original - after_eq) ** 2).item()

# print(f"均衡前MSE: {mse_before_eq_mmse:.4f}")
# print(f"均衡后MSE: {mse_after_eq_mmse:.4f}")
# print(f"均衡增益: {10 * math.log10(mse_before_eq_mmse / mse_after_eq_mmse):.2f} dB")

# 测试4：LS插值均衡 + 有CFO补偿
print("\n" + "="*50)
print("测试4：LS插值均衡 + 有CFO补偿")
print("="*50)
ofdm.channel_estimation_method = "ls_interp"

output4 = ofdm(input_tensor)

original = ofdm.original_signal
before_eq = ofdm.rx_data_before_eq
after_eq = ofdm.rx_data_after_eq

mse_before_eq_interp = torch.mean(torch.abs(original - before_eq) ** 2).item()
mse_after_eq_interp = torch.mean(torch.abs(original - after_eq) ** 2).item()

print(f"均衡前MSE: {mse_before_eq_interp:.4f}")
print(f"均衡后MSE: {mse_after_eq_interp:.4f}")
print(f"均衡增益: {10 * math.log10(mse_before_eq_interp / mse_after_eq_interp):.2f} dB")

# 最终对比总结
print("\n" + "="*50)
print("最终性能对比总结")
print("="*50)
print(f"{'方法':<25} {'均衡后MSE':<15} {'增益(dB)':<10}")
print("-"*50)
print(f"{'LS平均+有CFO补偿':<25} {mse_after_eq:<15.4f} {10*math.log10(mse_before_eq/mse_after_eq):<10.2f}")
print(f"{'LS平均+无CFO补偿':<25} {mse_after_eq_no_cfo:<15.4f} {10*math.log10(mse_before_eq_no_cfo/mse_after_eq_no_cfo):<10.2f}")
# print(f"{'MMSE+有CFO补偿':<25} {mse_after_eq_mmse:<15.4f} {10*math.log10(mse_before_eq_mmse/mse_after_eq_mmse):<10.2f}")
print(f"{'LS插值+有CFO补偿':<25} {mse_after_eq_interp:<15.4f} {10*math.log10(mse_before_eq_interp/mse_after_eq_interp):<10.2f}")