import argparse
import math
import random 
import os
from util import *
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
from torch import nn, autograd
from torch import optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist

from torchvision import transforms, utils
from tqdm import tqdm
from torch.optim import lr_scheduler
import copy
import kornia.augmentation as K
import kornia
import lpips

from model import *
from dataset import ImageFolder
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

mse_criterion = nn.MSELoss()


def test(args, genA2B, genB2A, testA_loader, testB_loader, name, step):
    testA_loader = iter(testA_loader)
    testB_loader = iter(testB_loader)
    with torch.no_grad():
        test_sample_num = 16

        genA2B.eval(), genB2A.eval() 
        A2B = []
        B2A = []
        for i in range(test_sample_num):
            real_A = testA_loader.next()
            real_B = testB_loader.next()

            real_A, real_B = real_A.cuda(), real_B.cuda()

            A2B_content, A2B_style = genA2B.encode(real_A)
            B2A_content, B2A_style = genB2A.encode(real_B)

            if i % 2 == 0:
                A2B_mod1 = torch.randn([1, args.latent_dim]).cuda()
                B2A_mod1 = torch.randn([1, args.latent_dim]).cuda()
                A2B_mod2 = torch.randn([1, args.latent_dim]).cuda()
                B2A_mod2 = torch.randn([1, args.latent_dim]).cuda()

            fake_B2B, _, _ = genA2B(real_B)
            fake_A2A, _, _ = genB2A(real_A)

            colsA = [real_A, fake_A2A]
            colsB = [real_B, fake_B2B]
            
            fake_A2B_1 = genA2B.decode(A2B_content, A2B_mod1)
            fake_B2A_1 = genB2A.decode(B2A_content, B2A_mod1)

            fake_A2B_2 = genA2B.decode(A2B_content, A2B_mod2)
            fake_B2A_2 = genB2A.decode(B2A_content, B2A_mod2)

            fake_A2B_3 = genA2B.decode(A2B_content, B2A_style)
            fake_B2A_3 = genB2A.decode(B2A_content, A2B_style)

            colsA += [fake_A2B_3, fake_A2B_1, fake_A2B_2]
            colsB += [fake_B2A_3, fake_B2A_1, fake_B2A_2]

            fake_A2B2A, _,  _ = genB2A(fake_A2B_3, A2B_style)
            fake_B2A2B, _,  _ = genA2B(fake_B2A_3, B2A_style)
            colsA.append(fake_A2B2A)
            colsB.append(fake_B2A2B)

            fake_A2B2A, _,  _ = genB2A(fake_A2B_1, A2B_style)
            fake_B2A2B, _,  _ = genA2B(fake_B2A_1, B2A_style)
            colsA.append(fake_A2B2A)
            colsB.append(fake_B2A2B)

            fake_A2B2A, _,  _ = genB2A(fake_A2B_2, A2B_style)
            fake_B2A2B, _,  _ = genA2B(fake_B2A_2, B2A_style)
            colsA.append(fake_A2B2A)
            colsB.append(fake_B2A2B)

            fake_A2B2A, _, _ = genB2A(fake_A2B_1)
            fake_B2A2B, _, _ = genA2B(fake_B2A_1)
            colsA.append(fake_A2B2A)
            colsB.append(fake_B2A2B)

            colsA = torch.cat(colsA, 2).detach().cpu()
            colsB = torch.cat(colsB, 2).detach().cpu()

            A2B.append(colsA)
            B2A.append(colsB)
        A2B = torch.cat(A2B, 0)
        B2A = torch.cat(B2A, 0)

        utils.save_image(A2B, f'{im_path}/{name}_A2B_{str(step).zfill(6)}.jpg', normalize=True, range=(-1, 1), nrow=16)
        utils.save_image(B2A, f'{im_path}/{name}_B2A_{str(step).zfill(6)}.jpg', normalize=True, range=(-1, 1), nrow=16)

        genA2B.train(), genB2A.train()


def train(args, trainA_loader, trainB_loader, testA_loader, testB_loader, G_A2B, G_B2A, D_A, D_B, G_optim, D_optim, device):
    G_A2B.train(), G_B2A.train(), D_A.train(), D_B.train()
    trainA_loader = sample_data(trainA_loader)
    trainB_loader = sample_data(trainB_loader)
    G_scheduler = lr_scheduler.StepLR(G_optim, step_size=100000, gamma=0.5)
    D_scheduler = lr_scheduler.StepLR(D_optim, step_size=100000, gamma=0.5)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.1)

    loss_dict = {}
    mean_path_length_A2B = 0
    mean_path_length_B2A = 0

    if args.distributed:
        G_A2B_module = G_A2B.module
        G_B2A_module = G_B2A.module
        D_A_module = D_A.module
        D_B_module = D_B.module
        D_L_module = D_L.module

    else:
        G_A2B_module = G_A2B
        G_B2A_module = G_B2A
        D_A_module = D_A
        D_B_module = D_B
        D_L_module = D_L

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')
            break

        ori_A = next(trainA_loader)
        ori_B = next(trainB_loader)
        if isinstance(ori_A, list):
            ori_A = ori_A[0]
        if isinstance(ori_B, list):
            ori_B = ori_B[0]

        ori_A = ori_A.to(device)
        ori_B = ori_B.to(device)
        aug_A = aug(ori_A)
        aug_B = aug(ori_B)
        A = aug(ori_A[[np.random.randint(args.batch)]].expand_as(ori_A))
        B = aug(ori_B[[np.random.randint(args.batch)]].expand_as(ori_B))

        if i % args.d_reg_every == 0:
            aug_A.requires_grad = True
            aug_B.requires_grad = True
        
        A2B_content, A2B_style = G_A2B.encode(A)
        B2A_content, B2A_style = G_B2A.encode(B)

        # get new style
        aug_A2B_style = G_B2A.style_encode(aug_B)
        aug_B2A_style = G_A2B.style_encode(aug_A)
        rand_A2B_style = torch.randn([args.batch, args.latent_dim]).to(device).requires_grad_()
        rand_B2A_style = torch.randn([args.batch, args.latent_dim]).to(device).requires_grad_()

        # styles
        idx = torch.randperm(2*args.batch)
        input_A2B_style = torch.cat([rand_A2B_style, aug_A2B_style], 0)[idx][:args.batch]

        idx = torch.randperm(2*args.batch)
        input_B2A_style = torch.cat([rand_B2A_style, aug_B2A_style], 0)[idx][:args.batch]

        fake_A2B = G_A2B.decode(A2B_content, input_A2B_style)
        fake_B2A = G_B2A.decode(B2A_content, input_B2A_style)


        # train disc
        real_A_logit = D_A(aug_A)
        real_B_logit = D_B(aug_B)
        real_L_logit1 = D_L(rand_A2B_style)
        real_L_logit2 = D_L(rand_B2A_style)

        fake_B_logit = D_B(fake_A2B.detach())
        fake_A_logit = D_A(fake_B2A.detach())
        fake_L_logit1 = D_L(aug_A2B_style.detach())
        fake_L_logit2 = D_L(aug_B2A_style.detach())

        # global loss
        D_loss = d_logistic_loss(real_A_logit, fake_A_logit) +\
                 d_logistic_loss(real_B_logit, fake_B_logit) +\
                 d_logistic_loss(real_L_logit1, fake_L_logit1) +\
                 d_logistic_loss(real_L_logit2, fake_L_logit2)

        loss_dict['D_adv'] = D_loss

        if i % args.d_reg_every == 0:
            r1_A_loss = d_r1_loss(real_A_logit, aug_A)
            r1_B_loss = d_r1_loss(real_B_logit, aug_B)
            r1_L_loss = d_r1_loss(real_L_logit1, rand_A2B_style) + d_r1_loss(real_L_logit2, rand_B2A_style)
            r1_loss = r1_A_loss + r1_B_loss + r1_L_loss
            D_r1_loss = (args.r1 / 2 * r1_loss * args.d_reg_every)
            D_loss += D_r1_loss

        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()

        #Generator
        # adv loss
        fake_B_logit = D_B(fake_A2B)
        fake_A_logit = D_A(fake_B2A)
        fake_L_logit1 = D_L(aug_A2B_style)
        fake_L_logit2 = D_L(aug_B2A_style)

        lambda_adv = (1, 1, 1)
        G_adv_loss = 1 * (g_nonsaturating_loss(fake_A_logit, lambda_adv) +\
                         g_nonsaturating_loss(fake_B_logit, lambda_adv) +\
                         2*g_nonsaturating_loss(fake_L_logit1, (1,)) +\
                         2*g_nonsaturating_loss(fake_L_logit2, (1,)))

        # style consis loss
        G_con_loss = 50 * (A2B_style.var(0, unbiased=False).sum() + B2A_style.var(0, unbiased=False).sum())
                    
        # cycle recon
        A2B2A_content, A2B2A_style = G_B2A.encode(fake_A2B)
        B2A2B_content, B2A2B_style = G_A2B.encode(fake_B2A)
        fake_A2B2A = G_B2A.decode(A2B2A_content, shuffle_batch(A2B_style))
        fake_B2A2B = G_A2B.decode(B2A2B_content, shuffle_batch(B2A_style))

        G_cycle_loss = 20 * (F.mse_loss(fake_A2B2A, A) + F.mse_loss(fake_B2A2B, B))
        lpips_loss = 10 * (lpips_fn(fake_A2B2A, A).mean() + lpips_fn(fake_B2A2B, B).mean()) #10 for anime

        # style reconstruction
        G_style_loss = 5 * (mse_criterion(A2B2A_style, input_A2B_style) +\
                            mse_criterion(B2A2B_style, input_B2A_style))


        G_loss =  G_adv_loss + G_cycle_loss + G_con_loss + lpips_loss + G_style_loss

        loss_dict['G_adv'] = G_adv_loss
        loss_dict['G_con'] = G_con_loss
        loss_dict['G_cycle'] = G_cycle_loss
        loss_dict['lpips'] = lpips_loss

        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()

        G_scheduler.step()
        D_scheduler.step()

        accumulate(G_A2B_ema, G_A2B_module)
        accumulate(G_B2A_ema, G_B2A_module)

        loss_reduced = reduce_loss_dict(loss_dict)
        D_adv_loss_val = loss_reduced['D_adv'].mean().item()

        G_adv_loss_val = loss_reduced['G_adv'].mean().item()
        G_cycle_loss_val = loss_reduced['G_cycle'].mean().item()
        G_con_loss_val = loss_reduced['G_con'].mean().item()
        lpips_val = loss_reduced['lpips'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'Dadv: {D_adv_loss_val:.2f}; lpips: {lpips_val:.2f} '
                    f'Gadv: {G_adv_loss_val:.2f}; Gcycle: {G_cycle_loss_val:.2f}; GMS: {G_con_loss_val:.2f} {G_style_loss.item():.2f}'
                )
            )

            if i % 1000 == 0:
                with torch.no_grad():
                    test(args, G_A2B, G_B2A, testA_loader, testB_loader, 'normal', i)
                    test(args, G_A2B_ema, G_B2A_ema, testA_loader, testB_loader, 'ema', i)

            if (i+1) % 2000 == 0:
                torch.save(
                    {
                        'G_A2B': G_A2B_module.state_dict(),
                        'G_B2A': G_B2A_module.state_dict(),
                        'G_A2B_ema': G_A2B_ema.state_dict(),
                        'G_B2A_ema': G_B2A_ema.state_dict(),
                        'D_A': D_A_module.state_dict(),
                        'D_B': D_B_module.state_dict(),
                        'D_L': D_L_module.state_dict(),
                        'G_optim': G_optim.state_dict(),
                        'D_optim': D_optim.state_dict(),
                        'iter': i,
                    },
                    os.path.join(model_path, 'ck.pt'),
                )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--iter', type=int, default=300000)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--n_sample', type=int, default=64)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--lambda_cycle', type=int, default=1)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--mixing', type=float, default=0.9)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_down', type=int, default=3)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--d_path', type=str, required=True)
    parser.add_argument('--latent_dim', type=int, default=8)
    parser.add_argument('--lr_mlp', type=float, default=0.01)
    parser.add_argument('--n_res', type=int, default=1)

    args = parser.parse_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = False

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    save_path = f'./{args.name}'
    im_path = os.path.join(save_path, 'sample')
    model_path = os.path.join(save_path, 'checkpoint')
    os.makedirs(im_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    args.n_mlp = 5

    args.start_iter = 0

    G_A2B = Generator( args.size, args.num_down, args.latent_dim, args.n_mlp, lr_mlp=args.lr_mlp, n_res=args.n_res).to(device)
    D_A = Discriminator(args.size).to(device)
    G_B2A = Generator( args.size, args.num_down, args.latent_dim, args.n_mlp, lr_mlp=args.lr_mlp, n_res=args.n_res).to(device)
    D_B = Discriminator(args.size).to(device)
    D_L = LatDiscriminator(args.latent_dim).to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    G_A2B_ema = copy.deepcopy(G_A2B).to(device).eval()
    G_B2A_ema = copy.deepcopy(G_B2A).to(device).eval()

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    G_optim = optim.Adam( list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=args.lr, betas=(0, 0.99))
    D_optim = optim.Adam(
        list(D_L.parameters()) + list(D_A.parameters()) + list(D_B.parameters()),
        lr=args.lr, betas=(0**d_reg_ratio, 0.99**d_reg_ratio))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
            
        except ValueError:
            pass
            
        G_A2B.load_state_dict(ckpt['G_A2B'])
        G_B2A.load_state_dict(ckpt['G_B2A'])
        G_A2B_ema.load_state_dict(ckpt['G_A2B_ema'])
        G_B2A_ema.load_state_dict(ckpt['G_B2A_ema'])
        D_A.load_state_dict(ckpt['D_A'])
        D_B.load_state_dict(ckpt['D_B'])
        D_L.load_state_dict(ckpt['D_L'])

        G_optim.load_state_dict(ckpt['G_optim'])
        D_optim.load_state_dict(ckpt['D_optim'])
        args.start_iter = ckpt['iter']

    if args.distributed:
        G_A2B = nn.parallel.DistributedDataParallel(
            G_A2B,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        D_A = nn.parallel.DistributedDataParallel(
            D_A,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        G_B2A = nn.parallel.DistributedDataParallel(
            G_B2A,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        D_B = nn.parallel.DistributedDataParallel(
            D_B,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        D_L = nn.parallel.DistributedDataParallel(
            D_L,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
    ])

    aug = nn.Sequential(
        K.RandomAffine(degrees=(-20,20), scale=(0.8, 1.2), translate=(0.1, 0.1), shear=0.15),
        kornia.geometry.transform.Resize(256+30),
        K.RandomCrop((256,256)),
        K.RandomHorizontalFlip(),
    )


    d_path = args.d_path
    trainA = ImageFolder(os.path.join(d_path, 'trainA'), train_transform)
    trainB = ImageFolder(os.path.join(d_path, 'trainB'), train_transform)
    testA = ImageFolder(os.path.join(d_path, 'testA'), test_transform)
    testB = ImageFolder(os.path.join(d_path, 'testB'), test_transform)
    
    trainA_loader = data.DataLoader(trainA, batch_size=args.batch, 
            sampler=data_sampler(trainA, shuffle=True, distributed=args.distributed), drop_last=True, pin_memory=True, num_workers=5)
    trainB_loader = data.DataLoader(trainB, batch_size=args.batch, 
            sampler=data_sampler(trainB, shuffle=True, distributed=args.distributed), drop_last=True, pin_memory=True, num_workers=5)

    testA_loader = data.DataLoader(testA, batch_size=1, shuffle=False)
    testB_loader = data.DataLoader(testB, batch_size=1, shuffle=False)


    train(args, trainA_loader, trainB_loader, testA_loader, testB_loader, G_A2B, G_B2A, D_A, D_B, G_optim, D_optim, device)
