# -*- coding: utf-8 -*-
"""
Created on Tue Dec  17:00:00 2023

@author: chun
"""
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import DeepJSCC, ratio2filtersize
from torch.nn.parallel import DataParallel
from utils import image_normalization, set_seed, save_model, view_model_param
from fractions import Fraction
from dataset import Vanilla
import numpy as np
import time
from tensorboardX import SummaryWriter
import glob
import yaml

class CosineAnnealingWarmupRestartsLR:
    """
    组合调度器：学习率预热 + 余弦退火周期性重启
    阶段1（预热）：学习率从0线性增长到初始LR (仅在第一个周期开始时执行)
    阶段2（退火+重启）：学习率按余弦函数衰减，每 T_i 个 Epoch 重置一次
    """
    def __init__(self, optimizer, T_0, T_mult, init_lr, min_lr, min_lr_decay=0.5):
        self.optimizer = optimizer
        
        self.T_0 = T_0          
        self.T_mult = T_mult    
        self.init_lr = init_lr  # 最大学习率 (保持不变)
        self.base_min_lr = min_lr # 初始最小学习率 (基准值)
        self.min_lr_decay = min_lr_decay # 每过一个周期，min_lr乘以这个系数
        
        self.last_epoch = -1
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        current_epoch = epoch
        cycle = 0
        cycle_start = 0
        T_i = self.T_0
        
        # 1. 动态计算当前周期 (Cycle) 和 周期起始位置
        while current_epoch >= cycle_start + T_i:
            cycle_start += T_i
            cycle += 1
            T_i = int(self.T_0 * (self.T_mult ** cycle))
            
        # 2. 计算当前周期的动态最小学习率和最大学习率
        current_min_lr = self.base_min_lr * (self.min_lr_decay ** cycle)
        current_max_lr = self.base_min_lr * (self.min_lr_decay ** (cycle-1))
        
        # 3. 计算相对位置
        rel_epoch = current_epoch - cycle_start
        
        # 4. 余弦退火计算
        cos_decay = 0.5 * (1 + np.cos(np.pi * rel_epoch / T_i))
        lr = current_min_lr + (current_max_lr - current_min_lr) * cos_decay
        
        # 5. 更新优化器
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


def train_epoch(model, optimizer, param, data_loader):
    model.train()
    epoch_loss = 0

    for iter, (images, _) in enumerate(data_loader):
        images = images.cuda() if param['parallel'] and torch.cuda.device_count(
        ) > 1 else images.to(param['device'])
        optimizer.zero_grad()
        outputs = model.forward(images)
        outputs = image_normalization('denormalization')(outputs)
        images = image_normalization('denormalization')(images)
        loss = model.loss(images, outputs) if not param['parallel'] else model.module.loss(
            images, outputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)

    return epoch_loss, optimizer


def evaluate_epoch(model, param, data_loader):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for iter, (images, _) in enumerate(data_loader):
            images = images.cuda() if param['parallel'] and torch.cuda.device_count(
            ) > 1 else images.to(param['device'])
            outputs = model.forward(images)
            outputs = image_normalization('denormalization')(outputs)
            images = image_normalization('denormalization')(images)
            loss = model.loss(images, outputs) if not param['parallel'] else model.module.loss(
                images, outputs)
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)

    return epoch_loss


def config_parser_pipeline():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'imagenet'], help='dataset')
    parser.add_argument('--out', default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'out')), type=str, help='out_path')
    parser.add_argument('--disable_tqdm', default=False, type=bool, help='disable_tqdm')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--parallel', default=False, type=bool, help='parallel')
    parser.add_argument('--snr_list', default=(12,30), help='snr_list')
    parser.add_argument('--ratio_list', default=['1/6'], nargs='+', help='ratio_list')
    parser.add_argument('--channel', default='AWGN', type=str,
                        choices=['AWGN', 'Rayleigh'], help='channel')
    parser.add_argument('--bindwidth_ratio',default=0.25,type=float,help='bindwidth_ratio')
    parser.add_argument('--nfft',default=128,help='nfft')
    parser.add_argument('--cp_len',default=16,help='cp_len')
    parser.add_argument('--npaths',default=8,help='npaths')
    parser.add_argument('--timeoffset_range',default=(-3,3),help='timeoffset_range')
    parser.add_argument('--cfd',default=0.015,help='cfd')
    parser.add_argument('--pd',default=0.1,help='pd')

    return parser.parse_args()


def main_pipeline():
    args = config_parser_pipeline()

    print("Training Start")
    dataset_name = args.dataset
    out_dir = args.out
    args.snr_list = list(map(float, args.snr_list))
    args.ratio_list = list(map(lambda x: float(Fraction(x)), args.ratio_list))
    params = {}
    params['disable_tqdm'] = args.disable_tqdm
    params['dataset'] = dataset_name
    params['out_dir'] = out_dir
    params['device'] = args.device
    params['snr_list'] = args.snr_list
    params['ratio_list'] = args.ratio_list
    params['channel'] = args.channel
    params['bindwidth_ratio']=args.bindwidth_ratio
    params['npaths']=args.npaths
    params['nfft']=args.nfft
    params['cp_len']=args.cp_len
    params['timeoffset_range']=args.timeoffset_range
    params['cfd']=args.cfd
    params['pd']=args.pd
    
    
    if dataset_name == 'cifar10':
        params['batch_size'] = 64  # 1024
        params['num_workers'] = 4
        # ===================== 修改：总 Epoch 数 = 2个周期 * 3000 =====================
        params['epochs'] = 2000 
        params['T_0'] = 1500      # 第一个周期的长度 (3000 epoch)
        params['T_mult'] = 1       # 周期长度倍数 (1表示以后每个周期也是3000)
        params['init_lr'] = 1e-3  # 1e-2
        params['weight_decay'] = 5e-4
        params['parallel'] = False
        params['if_scheduler'] = True
        params['step_size'] = 300
        params['gamma'] = 0.1
        params['ReduceLROnPlateau'] = False
        params['lr_reduce_factor'] = 0.5
        params['lr_schedule_patience'] = 25
        # ===================== 修改：开启周期性余弦退火 =====================
        params['cosine_annealing'] = False
        params['cosine_restart'] = True # 新增标志位
        params['seed'] = 42
        params['max_time'] = 48 # 增加最大训练时间，因为要跑6000轮
        params['min_lr'] = 1e-6
        params['warmup_epochs'] = 50 
        # ===================== 新增：周期性参数 =====================
        
        
    elif dataset_name == 'imagenet':
        params['batch_size'] = 32
        params['num_workers'] = 4
        params['epochs'] = 600
        params['init_lr'] = 1e-4
        params['weight_decay'] = 5e-4
        params['parallel'] = True
        params['if_scheduler'] = True
        params['gamma'] = 0.1
        params['seed'] = 42
        params['ReduceLROnPlateau'] = True
        params['lr_reduce_factor'] = 0.5
        params['lr_schedule_patience'] = 15
        params['max_time'] = 24
        params['min_lr'] = 1e-5
    else:
        raise Exception('Unknown dataset')

    set_seed(params['seed'])

    for ratio in params['ratio_list']:
            params['ratio'] = ratio
            params['snr']=params['snr_list']

            train_pipeline(params)


                
# add train_pipeline to with only dataset_name args
def train_pipeline(params):
    


    dataset_name = params['dataset']
    # dataset root is script-relative to avoid re-downloading when CWD differs
    
    dataset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset'))
    
    print(f"数据集根目录: {dataset_root}")
    # load data
    if dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), ])
        train_dataset = datasets.CIFAR10(root=dataset_root, train=True,
                                         download=True, transform=transform)

        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=params['batch_size'], num_workers=params['num_workers'])
        test_dataset = datasets.CIFAR10(root=dataset_root, train=False,
                                        download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=params['batch_size'], num_workers=params['num_workers'])

    elif dataset_name == 'imagenet':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((128, 128))])  # the size of paper is 128
        print("loading data of imagenet")
        train_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'ImageNet', 'train'), transform=transform)

        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=params['batch_size'], num_workers=params['num_workers'])
        test_dataset = Vanilla(root=os.path.join(dataset_root, 'ImageNet', 'val'), transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=params['batch_size'], num_workers=params['num_workers'])
    else:
        raise Exception('Unknown dataset')

    # create model
    image_fisrt = train_dataset.__getitem__(0)[0]
    image_fisrt = image_fisrt.unsqueeze(0)
    c = ratio2filtersize(image_fisrt, params['ratio'])
    print("The snr is {}, the inner channel is {}, the ratio is {:.2f}".format(
        params['snr'], c, params['ratio']))
    
    model = DeepJSCC(c=c, nfft=params['nfft'],cp_len=params['cp_len'],snr=params['snr'],bindwitdh_ratio=params['bindwidth_ratio'],cfd_std=params['cfd'],pd_std=params['pd'],n_paths=params['npaths'],timing_offset_range=params['timeoffset_range'])

    # init exp dir
    out_dir = params['out_dir']
    phaser = dataset_name.upper() + '_' + str(c) + '_' + str(params['snr'][0]) + '_' + str(params['snr'][1])+'_'+\
        "{:.2f}".format(params['ratio']) + '_' + str(params['bindwidth_ratio'])+'_'+str(params['nfft'])+'_'+str(params['npaths'])
   
    root_log_dir = out_dir + '/' + 'logs/' + phaser
    root_ckpt_dir = out_dir + '/' + 'checkpoints/' + phaser
    root_config_dir = out_dir + '/' + 'configs/' + phaser
    

    # model init
    device = torch.device(params['device'] if torch.cuda.is_available() else 'cpu')
    print(f"将使用设备: {device}")
    if params['parallel'] and torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model = model.cuda()
    else:
        model = model.to(device)
    
    # opt
    optimizer = optim.Adam(
        model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    
    # ===================== 修改：调度器初始化逻辑 =====================
    scheduler = None
    if params.get('cosine_annealing', False):
        if params.get('cosine_restart', False):
            print(f"[Info] 使用 CosineAnnealingWarmRestarts (周期性重启): T_0={params['T_0']}, T_mult={params['T_mult']}")
            scheduler = CosineAnnealingWarmupRestartsLR(
                optimizer,
                T_0=params['T_0'],
                T_mult=params['T_mult'],
                init_lr=params['init_lr'],
                min_lr=params['min_lr']
            )
        else:
            print(f"[Info] 使用普通 CosineAnnealing (单周期)")
            # 这里可以保留你原来的单周期类，为了简洁，这里暂略，可复用上面的类但设T_0很大
            scheduler = CosineAnnealingWarmupRestartsLR(
                optimizer,
                warmup_epochs=params['warmup_epochs'],
                T_0=params['epochs'], # 设为总轮数即不重启
                T_mult=1,
                init_lr=params['init_lr'],
                min_lr=params['min_lr']
            )
    elif params['ReduceLROnPlateau']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=params['lr_reduce_factor'],
                                                         patience=params['lr_schedule_patience'],
                                                         )
    elif params['if_scheduler']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=params['step_size'], gamma=params['gamma'])
    else:
        print("No scheduler")

    # ---- resume from checkpoint support ----
    start_epoch = 0
    if os.path.exists(root_ckpt_dir):
        ckpts = glob.glob(root_ckpt_dir + '/*.pkl')
        if len(ckpts) > 0:
            # pick latest by epoch number in filename 'epoch_{n}.pkl'
            def epoch_from_name(p):
                try:
                    name = os.path.basename(p)
                    num = name.split('_')[-1]
                    num = int(os.path.splitext(num)[0])
                    return num
                except Exception:
                    return -1

            latest = max(ckpts, key=epoch_from_name)

            try:
                print(f"Found checkpoint {latest}, attempting to resume...")
                ckpt = torch.load(latest, map_location='cpu', weights_only=False)
                # support both legacy (state_dict only) and new checkpoint dict
                if isinstance(ckpt, dict) and 'model_state' in ckpt:
                    model_state = ckpt['model_state']
                    opt_state = ckpt.get('optimizer_state', None)
                    sch_state = ckpt.get('scheduler_state', None)
                    start_epoch = epoch_from_name(latest) + 1
                else:
                    # legacy: entire file is model.state_dict()
                    model_state = ckpt
                    opt_state = None
                    sch_state = None
                    start_epoch = epoch_from_name(latest) + 1

                # try load model state (handle module. prefix differences)
                try:
                    model.load_state_dict(model_state,strict=False)
                except Exception:
                    # try stripping 'module.' prefix
                    new_state = {}
                    for k, v in model_state.items():
                        new_key = k.replace('module.', '') if k.startswith('module.') else k
                        new_state[new_key] = v
                    model.load_state_dict(new_state,strict=False)

                if opt_state is not None:
                    try:
                        optimizer.load_state_dict(opt_state)
                        # move optimizer state to correct device
                        for state in optimizer.state.values():
                            if isinstance(state, dict):
                                for k, v in list(state.items()):
                                    if isinstance(v, torch.Tensor):
                                        state[k] = v.to(device)
                    except Exception as e:
                        print('Warning: failed to load optimizer state:', e)


                # if scheduler is not None and sch_state is not None:
                #     try:
                #         scheduler.load_state_dict(sch_state)
                #     except Exception as e:
                #         print('Warning: failed to load scheduler state:', e)

                print(f"Resumed from epoch {start_epoch - 1}, will continue from epoch {start_epoch}.")
            except Exception as e:
                print(f"Failed to load checkpoint: {type(e).__name__}: {e}")
    # ---- end resume support ----
    if start_epoch==params['epochs']:
        return 0
    writer = SummaryWriter(log_dir=root_log_dir)
    writer.add_text('config', str(params))
    t0 = time.time()
    epoch_train_losses, epoch_val_losses = [], []
    per_epoch_time = []

    # train
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(start_epoch, params['epochs']), disable=params['disable_tqdm']) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, optimizer = train_epoch(
                    model, optimizer, params, train_loader)

                epoch_val_loss = evaluate_epoch(model, params, test_loader)

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss)

                per_epoch_time.append(time.time() - start)

                # Saving checkpoint

                if not os.path.exists(root_ckpt_dir):
                    os.makedirs(root_ckpt_dir)
                # save full checkpoint (model + optimizer + scheduler + epoch)
                ckpt = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                }
                torch.save(ckpt, os.path.join(root_ckpt_dir, f'epoch_{epoch}.pkl'))

                files = glob.glob(root_ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)

                # ===================== 修改：调度器 Step 逻辑 =====================
                if params.get('cosine_annealing', False) and scheduler is not None:
                    scheduler.step(epoch)
                elif params['ReduceLROnPlateau'] and scheduler is not None:
                    scheduler.step(epoch_val_loss)
                elif params['if_scheduler'] and scheduler is not None:
                    scheduler.step()
                
                # ===================== 移除：LR 低于最小值就停止的限制 (因为周期性会回升) =====================
                # if optimizer.param_groups[0]['lr'] < params['min_lr']:
                #     print("\n!! LR EQUAL TO MIN LR SET.")
                #     break

                # Stop training after params['max_time'] hours
                if time.time() - t0 > params['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(
                        params['max_time']))
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    test_loss = evaluate_epoch(model, params, test_loader)
    train_loss = evaluate_epoch(model, params, train_loader)
    print("Test Accuracy: {:.4f}".format(test_loss))
    print("Train Accuracy: {:.4f}".format(train_loss))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    """
        Write the results in out_dir/results folder
    """

    writer.add_text(tag='result', text_string="""Dataset: {}\nparams={}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST Loss: {:.4f}\nTRAIN Loss: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""
                    .format(dataset_name, params, view_model_param(model), np.mean(np.array(train_loss)),
                            np.mean(np.array(test_loss)), epoch, (time.time() - t0) / 3600, np.mean(per_epoch_time)))
    writer.close()
    if not os.path.exists(os.path.dirname(root_config_dir)):
        os.makedirs(os.path.dirname(root_config_dir))
    with open(root_config_dir + '.yaml', 'w') as f:
        dict_yaml = {'dataset_name': dataset_name, 'params': params,
                     'inner_channel': c, 'total_parameters': view_model_param(model)}

        yaml.dump(dict_yaml, f)

    del model, optimizer, scheduler, train_loader, test_loader
    del writer


if __name__ == '__main__':
    main_pipeline()