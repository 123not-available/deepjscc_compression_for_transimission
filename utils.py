import torch
import torch.nn.functional as F
import os
import numpy as np
import math


def image_normalization(norm_type):
    def _inner(tensor: torch.Tensor):
        if norm_type == 'normalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return tensor * 255.0
        else:
            raise Exception('Unknown type of normalization')
    return _inner


def get_psnr(image, gt, max_val=255, mse=None):
    if mse is None:
        mse = F.mse_loss(image, gt)
    mse = torch.tensor(mse) if not isinstance(mse, torch.Tensor) else mse

    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr


def save_model(model, dir, path):
    os.makedirs(dir, exist_ok=True)
    flag = 1
    while True:
        if os.path.exists(path):
            path = path + '_' + str(flag)
            flag += 1
        else:
            break
    torch.save(model.state_dict(), path)
    print("Model saved in {}".format(path))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def view_model_param(model):
    total_param = 0

    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    return total_param


def get_pilot_list(x):
        b,s,n=x.shape
        pilot_list=[3]
        while s>len(pilot_list)*7:
            pilot_list.append(pilot_list[-1]+7)
        return pilot_list
    
def _generate_zc_sequence( length, root=5, device=None):
        """
        生成Zadoff-Chu序列
        Args:
            length: 序列长度
            root: 根索引，必须与length互质
            device: 设备
        Returns:
            zc_seq: 归一化的ZC序列
        """
        if device is None:
            device = torch.device("cpu")
        
        # 确保root和length互质
        assert math.gcd(root, length) == 1, f"ZC根索引{root}必须与长度{length}互质"
        
        n = torch.arange(length, dtype=torch.float32, device=device)
        # ZC序列生成公式
        phase = torch.pi * root * n * (n + 1) / length
        zc_seq = torch.exp(1j * phase)
        
        # 单位功率归一化
        zc_seq = zc_seq / torch.sqrt((zc_seq.abs() ** 2).mean())
        
        return zc_seq.view(1, 1, length)
    
def _generate_pilot(pilot_type,N,zc_root,device,pilot_seed=42,pilot_power=1):
        """根据pilot_type生成对应的导频序列"""
        if pilot_type == "zc":
            # ZC序列（推荐用于实际硬件和真实信道）
            pilot = _generate_zc_sequence(N, zc_root, device)
        
        elif pilot_type == "qpsk":
            # QPSK随机导频（推荐用于仿真）
            old_rng_state = torch.get_rng_state()
            torch.manual_seed(pilot_seed)
            phase = 2 * torch.pi * torch.rand(1, 1, N, device=device)
            pilot = torch.exp(1j * phase)
            pilot = pilot / torch.sqrt((pilot.abs() ** 2).mean())
            torch.set_rng_state(old_rng_state)
        
        elif pilot_type == "all1":
            # 全1导频（仅用于调试）
            pilot = torch.ones(1, 1, N, dtype=torch.complex64, device=device)
        
        elif pilot_type == "alternating":
            # 交替±1导频（仅用于简单仿真）
            pilot = torch.ones(1, 1, N, dtype=torch.complex64, device=device)
            pilot[:, :, 1::2] = -1
        
        else:
            raise ValueError(f"不支持的导频类型: {pilot_type}")
        
        
        return pilot
    
def _shift_linear_extrapolate_batch(x, shift):
        """
        x: (B, L)  输入张量
        shift: (B,)  每个样本一个整数位移
            shift > 0 表示向右移，左侧缺失
            shift < 0 表示向左移，右侧缺失
        return: (B, L)
        """
        assert x.dim() == 2
        B, L = x.shape
        device = x.device
        dtype = x.dtype

        shift = shift.to(device=device, dtype=torch.long).view(B, 1)

        pos = torch.arange(L, device=device).view(1, L)  # (1, L)
        src = pos - shift                                  # (B, L)

        valid = (src >= 0) & (src < L)
        src_clamped = src.clamp(0, L - 1)

        y = torch.gather(x, 1, src_clamped)

        left_mask = src < 0
        right_mask = src >= L

        if L >= 2:
            d_left = x[:, 1] - x[:, 0]     # (B,)
            d_right = x[:, -1] - x[:, -2]  # (B,)

            # 左侧外推
            left_val = x[:, :1] + d_left[:, None] * src.to(dtype)
            # 右侧外推
            right_val = x[:, -1:] + d_right[:, None] * (src.to(dtype) - (L - 1))

            y = torch.where(left_mask, left_val, y)
            y = torch.where(right_mask, right_val, y)
        else:
            y = x.expand_as(y)

        return y
    
    
@staticmethod
def linear_interp(x, xp, fp):
    """
    原版公式完全一致 + 支持 [B,N,P] 批量GPU
    x: [L]
    xp: [P]
    fp: [B, N, P]
    return: [B, N, L]
    """
    P = xp.shape[0]
    if P == 1:
        return fp.expand(-1, -1, len(x))

    m = (fp[..., 1:] - fp[..., :-1]) / (xp[1:] - xp[:-1] + 1e-8)
    b = fp[..., :-1] - m * xp[:-1]

    indices = torch.searchsorted(xp, x, right=True)
    indices = torch.clamp(indices, 1, P - 1) - 1

    # 批量gather
    m = m.gather(dim=-1, index=indices.view(1,1,-1))
    b = b.gather(dim=-1, index=indices.view(1,1,-1))
    x = x.view(1, 1, -1)
    return m * x + b

@staticmethod
def ls_estimate_and_mmse_equalize(y_pilot, pilot, y_data, pilot_list, data_list, all_pos, noise_pwr):
    device = y_pilot.device
    B, P, K = y_pilot.shape

    # 全部就地转GPU tensor，杜绝CPU列表索引
    pilot_pos = torch.tensor(pilot_list, dtype=torch.float32, device=device)
    data_pos = torch.tensor(data_list, dtype=torch.int64, device=device)
    all_pos = torch.tensor(all_pos, dtype=torch.float32, device=device)
    L = all_pos.shape[0]

    # pilot: [1,1,K] 完美匹配导频符号
    H_pilot = y_pilot / (pilot.to(device) + 1e-8)  # [B, P, K]

    # 维度变换适配批量插值 [B,P,K] -> [B,K,P]
    H_pilot_t = H_pilot.permute(0, 2, 1)
    # 批量时间插值 【无循环】
    H_est_t = linear_interp(all_pos, pilot_pos, H_pilot_t)  # [B,K,L]
    # 换回你原版输出维度 [B, L, K]
    H_est = H_est_t.permute(0, 2, 1)

    # MMSE 均衡 完全复刻你公式
    H_data = H_est[:, data_pos, :]
    equalized_data = torch.conj(H_data) * y_data / (H_data.abs() ** 2 + noise_pwr)

    return equalized_data, H_est

""" @staticmethod
def _cubic_fill_1d_gpu(y, mask, x):
    
    B, L = y.shape
    device = y.device

    # 自然边界条件三对角矩阵
    A = torch.zeros(L, L, dtype=torch.float32, device=device)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0

    dx = x[1:] - x[:-1]
    for i in range(1, L-1):
        A[i, i-1] = dx[i-1]
        A[i, i] = 2 * (dx[i-1] + dx[i])
        A[i, i+1] = dx[i]

    # 批量求解（B 个样本一起算）
    y_known = y.clone()
    y_known[~mask] = 0.0

    b = torch.zeros(B, L, device=device)
    b[:, 1:-1] = 3 * ((y[:, 2:] - y[:, 1:-1]) / dx[1:] - (y[:, 1:-1] - y[:, :-2]) / dx[:-1])
    b = b * mask.float()

    # GPU 求解
    c = torch.linalg.solve(A.unsqueeze(0), b.unsqueeze(-1)).squeeze(-1)
    a = y_known
    d = (c[:, 1:] - c[:, :-1]) / (3 * dx + 1e-8)
    b_coeff = (a[:, 1:] - a[:, :-1])/dx - dx*(c[:, 1:] + 2*c[:, :-1])/3

    # 插值
    idx = torch.searchsorted(x[:-1], x, right=True) - 1
    idx = idx.clamp(0, L-2)

    dx_val = x - x[idx]
    out = a[:, idx] + b_coeff[:, idx]*dx_val + c[:, idx]*dx_val**2 + d[:, idx]*dx_val**3
    out[mask] = y[mask]
    return out """
    
@staticmethod
def _batch_cubic_fill(y, mask, x):
    """
    纯GPU批量三次样条填充
    输入：y [B,L], mask [B,L], x [L]
    输出：y_filled [B,L]
    """
    B, L = y.shape
    device = y.device

    # 自然边界三次样条矩阵（一次构建，全局复用）
    A = torch.zeros(L, L, dtype=torch.float32, device=device)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0

    dx = x[1:] - x[:-1]
    for i in range(1, L-1):
        A[i, i-1] = dx[i-1]
        A[i, i] = 2 * (dx[i-1] + dx[i])
        A[i, i+1] = dx[i]

    # 构造右侧项
    dy = (y[:, 2:] - y[:, 1:-1]) / (dx[1:] + 1e-8) - (y[:, 1:-1] - y[:, :-2]) / (dx[:-1] + 1e-8)
    b = torch.zeros(B, L, device=device)
    b[:, 1:-1] = 3 * dy

    # 求解
    c = torch.linalg.solve(A.unsqueeze(0), b.unsqueeze(-1)).squeeze(-1)
    a = y.clone()
    d = (c[:, 1:] - c[:, :-1]) / (3 * dx + 1e-8)
    b_coeff = (a[:, 1:] - a[:, :-1])/dx - dx*(c[:, 1:] + 2*c[:, :-1])/3

    # 插值
    idx = torch.searchsorted(x[:-1], x, right=True) - 1
    idx = idx.clamp(0, L-2)
    dx_val = x - x[idx]

    res = a[:, idx] + b_coeff[:, idx]*dx_val + c[:, idx]*dx_val**2 + d[:, idx]*dx_val**3
    res[mask] = y[mask]
    return res

def zeropaddingwithifft(z,K,N):
    
    batch,s,k=z.shape
    assert k==K,"zeropadding时错误"

    pad_total = N - K
    pad_left  = pad_total // 2        # 左侧（负频率）补零数
    pad_right = pad_total - pad_left  # 右侧（正频率）补零数

    z_padded = torch.zeros(
        batch, s, N,
        dtype=torch.complex64,
        device=z.device
    )

    z_padded[:,:, pad_left:pad_left + K] = z

    z_standard = torch.fft.ifftshift(z_padded, dim=-1)

    z_time = torch.fft.ifft(z_standard, n=N, dim=-1, norm="ortho")
    return z_time

def timecropfft(x,K,N):

    X = torch.fft.fft(x, n=N, dim=-1, norm="ortho")
    X = torch.fft.fftshift(X, dim=-1)

    pad_total = N - K
    pad_left = pad_total // 2
    X = X[..., pad_left:pad_left + K]   # (B, S, K)
    return X
    

def add_cp(x, cp_len):
    print(cp_len)
    return torch.cat((x[...,-cp_len:], x), dim=-1)

def rm_cp(x, cp_len):
    return x[...,cp_len:]

