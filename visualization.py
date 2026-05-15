# -*- coding: utf-8 -*-
import os
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from scipy.io import savemat,loadmat
from model import DeepJSCC, ratio2filtersize
import utils

"""
使用方式（在项目根目录）：
    python visualize_jscc.py --ckpt epoch_953.pkl  # 根据你实际的ckpt名修改
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str,
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset')),
                        help='dataset root (same as training)')
    parser.add_argument('--ckpt_dir', type=str,
                        default='/home/cj/Documents/Deep-JSCC-PyTorch-main/build/test5.13/lsestimate/checkpoints',
                        help='checkpoint root dir')
    parser.add_argument('--exp_name', type=str, default='CIFAR10_8_12.0_30.0_0.17_0.25_128_8',
                        help='experiment name, e.g. CIFAR10_8_12.0_30.0_0.17_0.25_128_10')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='checkpoint filename, e.g. epoch_953.pkl')
    parser.add_argument('--snr_low', type=float, default=12.0)
    parser.add_argument('--snr_high', type=float, default=30.0)
    parser.add_argument('--ratio', type=float, default=1/6)
    parser.add_argument('--bindwidth_ratio', type=float, default=0.25)
    parser.add_argument('--nfft', type=int, default=128)
    parser.add_argument('--cp_len', type=int, default=16)
    parser.add_argument('--npaths', type=int, default=8)
    parser.add_argument('--timeoffset_low', type=int, default=-3)
    parser.add_argument('--timeoffset_high', type=int, default=3)
    parser.add_argument('--cfd', type=float, default=0.015)
    parser.add_argument('--pd', type=float, default=0.1)
    parser.add_argument('--num_images', type=int, default=8)
    parser.add_argument('--out', type=str, default='jscc_vis.png')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument(
    '--indices',
    type=int,
    nargs='+',
    default=list(range(10)),
    help='要可视化对比的图片索引'
)
    return parser.parse_args()


def load_model(args, device):
    # 加载一张样本图像来计算 c
    transform = transforms.Compose([transforms.ToTensor()])
    cifar_root = os.path.abspath(os.path.join(args.dataset_root, ''))
    dataset = datasets.CIFAR10(root=cifar_root, train=False, download=True, transform=transform)
    x0, _ = dataset[0]
    x0 = x0.unsqueeze(0)  # (1,3,32,32)

    c = ratio2filtersize(x0, args.ratio)
    print(f"[INFO] inner channel c = {c}")

    snr_list = [args.snr_low, args.snr_high]

    model = DeepJSCC(
        c=c,
        nfft=args.nfft,
        cp_len=args.cp_len,
        snr=snr_list,
        bindwitdh_ratio=args.bindwidth_ratio,
        cfd_std=args.cfd,
        pd_std=args.pd,
        n_paths=args.npaths,
        timing_offset_range=(args.timeoffset_low, args.timeoffset_high),
    )

    model = model.to(device)

    # 加载 checkpoint
    ckpt_path = os.path.join(args.ckpt_dir, args.exp_name, args.ckpt)
    print(f"[INFO] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        state_dict = ckpt['model_state']
    else:
        state_dict = ckpt

    # 兼容 DataParallel
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_state = {}
        for k, v in state_dict.items():
            nk = k.replace('module.', '') if k.startswith('module.') else k
            new_state[nk] = v
        model.load_state_dict(new_state,strict=False)

    model.eval()
    return model

def encode(images,model,args,device):
    latent = model.encoder(images)
    print(latent.shape)
    rxlatent,signal=model.ofdm(latent)
    b,s,n=signal.shape
    signal=signal.reshape(b,-1)
    zeros = torch.zeros(b, 1000, device=signal.device, dtype=signal.dtype)
    signal_pad = torch.cat([signal, zeros], dim=1)
    signal=signal_pad.flatten()
    zeros = torch.zeros(10000,device=signal.device, dtype=signal.dtype)
    signal=torch.cat([zeros,signal])
    
    
    return latent,signal

def decode(latent,model,args,device):
    mat = loadmat('mat/signal_RX.mat')   # 读取mat文件
    rxsignal_np = mat['signal_RX']             # 取出变量，得到 numpy.ndarray
    rxsignal_tensor = torch.from_numpy(rxsignal_np).to(device)# 转成 tensor
    rxsignal_tensor = rxsignal_tensor.flatten()
    zeros=torch.zeros(1288,device=device)
    rxsignal=torch.cat([rxsignal_tensor,zeros])
    rxsignal=rxsignal.reshape(1000,-1)
    rxsignal=rxsignal[:,:144*21]
    rxsignal=rxsignal.reshape(1000,21,144)
    print(rxsignal.shape)
    rxlatent,signal=model.ofdm(latent,rxsignal=rxsignal)
    mse=torch.mean((rxlatent-latent)**2)
    print(mse)
    recon=model.decoder(rxlatent)
    return recon

def visualize_selected(images, recon, indices, out_path='selected_compare.png'):
    images = images.detach().cpu()
    recon = recon.detach().cpu()

    for idx in indices:
        if idx < 0 or idx >= images.size(0):
            raise ValueError(f'index {idx} out of range, valid range is [0, {images.size(0)-1}]')

    num = len(indices)
    fig, axes = plt.subplots(2, num, figsize=(2 * num, 4))

    if num == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(indices):
        img = images[idx].permute(1, 2, 0).numpy()
        rec = recon[idx].permute(1, 2, 0).numpy()

        axes[0, col].imshow(img)
        axes[0, col].set_title(f'Original #{idx}')
        axes[0, col].axis('off')

        axes[1, col].imshow(rec)
        axes[1, col].set_title(f'Recon #{idx}')
        axes[1, col].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] saved selected comparison to {out_path}")

def visualize(model, args, device,imgs=None,recon=None):
    if recon is None:
        # 准备数据
        transform = transforms.Compose([transforms.ToTensor()])
        cifar_root = os.path.abspath(os.path.join(args.dataset_root, ''))
        dataset = datasets.CIFAR10(root=cifar_root, train=False,
                                download=True, transform=transform)

        num = min(args.num_images, len(dataset))
        imgs = []
        for i in range(num):
            img, _ = dataset[i]
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0).to(device)  # (B,3,32,32)

        with torch.no_grad():
            # 直接送入模型：模型内部是 0-1 范围
            recon = model(imgs)

    # 你在训练里是 *255 后算loss，所以这里为了可视化，我们保持在0-1范围即可
    # 如果你想看和训练时一致的“0-255图像”，可以再乘 255 再 clamp/归一以方便显示

    # 拼接原图和重建图：先把 batch 做成网格
    grid_original = torchvision.utils.make_grid(imgs.cpu(), nrow=num, padding=2)
    grid_recon = torchvision.utils.make_grid(recon.cpu(), nrow=num, padding=2)

    # 可视化 & 保存
    fig, axes = plt.subplots(2, 1, figsize=(num * 2, 4))
    axes[0].imshow(grid_original.permute(1, 2, 0).numpy())
    axes[0].set_title('Original (0-1)')
    axes[0].axis('off')

    axes[1].imshow(grid_recon.permute(1, 2, 0).numpy())
    axes[1].set_title('Reconstructed by DeepJSCC (0-1)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[INFO] saved visualization to {args.out}")


def main():
    args = parse_args()
    utils.set_seed(43)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] using device: {device}")
    
    transform = transforms.Compose([transforms.ToTensor()])
    cifar_root = os.path.abspath(os.path.join(args.dataset_root, ''))
    dataset = datasets.CIFAR10(root=cifar_root, train=False,
                               download=True, transform=transform)
    images = torch.stack([dataset[i][0] for i in range(1000)], dim=0)
    images=images.to(device)
    
    model = load_model(args, device)
    
    visualize(model,args,device)
    
    latent,signal=encode(images,model,args,device)
    print(latent.shape,signal.shape)
    # signal_np = signal.detach().cpu().numpy()
    # out_mat = "mat/signal.mat"
    # savemat(out_mat, {"signal": signal_np}, oned_as='column')
    
    recon=decode(latent,model,args,device)
    
    visualize_selected(images, recon, args.indices, out_path='rxsignal_selected_compare.png')
    


if __name__ == '__main__':
    main()