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
from modeltest import DeepJSCC, ratio2filtersize
from torch.nn.parallel import DataParallel
from utils import image_normalization, set_seed, save_model, view_model_param
from fractions import Fraction
from dataset import Vanilla
import numpy as np
import time
from tensorboardX import SummaryWriter
import glob


def train_epoch(model, optimizer, param, data_loader):
    model.train()
    epoch_loss = 0

    # 初始化各个阶段的累计时间
    time_data_load = 0.0
    time_to_device = 0.0
    time_forward = 0.0
    time_loss = 0.0
    time_backward_opt = 0.0

    # 定义一个同步闭包，确保 GPU 计时准确
    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    sync()
    t_start = time.time()

    for iter, (images, _) in enumerate(data_loader):
        # 1. 记录数据加载耗时
        sync()
        t_data_loaded = time.time()
        time_data_load += (t_data_loaded - t_start)

        # 2. 记录数据转移到 GPU 的耗时
        images = images.cuda() if param['parallel'] and torch.cuda.device_count() > 1 else images.to(param['device'])
        sync()
        t_to_device = time.time()
        time_to_device += (t_to_device - t_data_loaded)

        optimizer.zero_grad()

        # 3. 记录前向传播耗时 (包含模型计算和图像反归一化)
        outputs = model.forward(images)
        outputs = image_normalization('denormalization')(outputs)
        images = image_normalization('denormalization')(images)
        sync()
        t_forward = time.time()
        time_forward += (t_forward - t_to_device)

        # 4. 记录计算 Loss 的耗时
        loss = model.loss(images, outputs) if not param['parallel'] else model.module.loss(images, outputs)
        sync()
        t_loss = time.time()
        time_loss += (t_loss - t_forward)

        # 5. 记录反向传播和优化器参数更新耗时
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        
        sync()
        t_backward_opt = time.time()
        time_backward_opt += (t_backward_opt - t_loss)

        # 重置 t_start，为下一次迭代的数据加载做准备
        t_start = time.time()
        
     # 打印本 Epoch 的各个阶段耗时统计
    total_time = time_data_load + time_to_device + time_forward + time_loss + time_backward_opt
    print(f"\n--- Epoch 性能耗时分析 ({len(data_loader)} Batches) ---")
    print(f"1. 数据加载 (DataLoader):   {time_data_load:.4f} s ({time_data_load/total_time*100:.1f}%)")
    print(f"2. 数据至显存 (To Device):  {time_to_device:.4f} s ({time_to_device/total_time*100:.1f}%)")
    print(f"3. 前向传播 (Forward):      {time_forward:.4f} s ({time_forward/total_time*100:.1f}%)")
    print(f"4. 损失计算 (Loss):         {time_loss:.4f} s ({time_loss/total_time*100:.1f}%)")
    print(f"5. 反向与优化 (Backward):   {time_backward_opt:.4f} s ({time_backward_opt/total_time*100:.1f}%)")
    print(f"-> 总计耗时:                {total_time:.4f} s")
    print("-" * 50)
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
    parser.add_argument('--out', default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'testout')), type=str, help='out_path')
    parser.add_argument('--disable_tqdm', default=False, type=bool, help='disable_tqdm')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--parallel', default=False, type=bool, help='parallel')
    parser.add_argument('--snr_list', default=['19', '13',
                        '7', '4', '1'], nargs='+', help='snr_list')
    parser.add_argument('--ratio_list', default=['1/6', '1/12'], nargs='+', help='ratio_list')
    parser.add_argument('--channel', default='AWGN', type=str,
                        choices=['AWGN', 'Rayleigh'], help='channel')

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
    if dataset_name == 'cifar10':
        params['batch_size'] = 64  # 1024
        params['num_workers'] = 4
        params['epochs'] = 1000
        params['init_lr'] = 1e-3  # 1e-2
        params['weight_decay'] = 5e-4
        params['parallel'] = False
        params['if_scheduler'] = True
        params['step_size'] = 640
        params['gamma'] = 0.1
        params['seed'] = 42
        params['ReduceLROnPlateau'] = False
        params['lr_reduce_factor'] = 0.5
        params['lr_schedule_patience'] = 15
        params['max_time'] = 24
        params['min_lr'] = 1e-5
    elif dataset_name == 'imagenet':
        params['batch_size'] = 32
        params['num_workers'] = 4
        params['epochs'] = 300
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
        for snr in params['snr_list']:
            params['ratio'] = ratio
            params['snr'] = snr

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
                                  batch_size=params['batch_size'], 
                                  num_workers=params['num_workers'],
                                  pin_memory=True,
                                  persistent_workers=True)
        test_dataset = datasets.CIFAR10(root=dataset_root, train=False,
                                        download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=params['batch_size'], 
                                 num_workers=params['num_workers'],
                                 pin_memory=True,
                                 persistent_workers=True)

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
    c = ratio2filtersize(image_fisrt, params['ratio'])
    print("The snr is {}, the inner channel is {}, the ratio is {:.2f}".format(
        params['snr'], c, params['ratio']))
    model = DeepJSCC(c=c,cp_len=10,bindwitdh_ratio=0.25, snr=params['snr'])

    # init exp dir
    out_dir = params['out_dir']
    phaser = dataset_name.upper() + '_' + str(c) + '_' + str(params['snr']) + '_' + \
        "{:.2f}".format(params['ratio']) + '_' + str(params['channel']+'test') 
   
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
    if params['if_scheduler'] and not params['ReduceLROnPlateau']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=params['step_size'], gamma=params['gamma'])
    elif params['ReduceLROnPlateau']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=params['lr_reduce_factor'],
                                                         patience=params['lr_schedule_patience'],
                                                         verbose=False)
    else:
        print("No scheduler")
        scheduler = None

    
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
                ckpt = torch.load(latest, map_location='cpu')
                # support both legacy (state_dict only) and new checkpoint dict
                if isinstance(ckpt, dict) and 'model_state' in ckpt:
                    model_state = ckpt['model_state']
                    opt_state = ckpt.get('optimizer_state', None)
                    sch_state = ckpt.get('scheduler_state', None)
                    start_epoch = ckpt.get('epoch', 0) + 1
                else:
                    # legacy: entire file is model.state_dict()
                    model_state = ckpt
                    opt_state = None
                    sch_state = None
                    start_epoch = epoch_from_name(latest) + 1

                # try load model state (handle module. prefix differences)
                try:
                    model.load_state_dict(model_state)
                except Exception:
                    # try stripping 'module.' prefix
                    new_state = {}
                    for k, v in model_state.items():
                        new_key = k.replace('module.', '') if k.startswith('module.') else k
                        new_state[new_key] = v
                    model.load_state_dict(new_state)

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

                if scheduler is not None and sch_state is not None:
                    try:
                        scheduler.load_state_dict(sch_state)
                    except Exception as e:
                        print('Warning: failed to load scheduler state:', e)

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

                if params['ReduceLROnPlateau'] and scheduler is not None:
                    scheduler.step(epoch_val_loss)
                elif params['if_scheduler'] and not params['ReduceLROnPlateau']:
                    scheduler.step()  # use only information from the validation loss

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

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
        import yaml
        yaml.dump(dict_yaml, f)

    del model, optimizer, scheduler, train_loader, test_loader
    del writer


if __name__ == '__main__':
    main_pipeline()
    # main()
