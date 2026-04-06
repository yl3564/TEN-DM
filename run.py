import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW, Adam
import argparse
from scipy.stats import kstest
import setproctitle
from torch.utils.tensorboard import SummaryWriter
import datetime
import pickle
import os
from tqdm import tqdm
import random
import json
import time
from DSTPP import GaussianDiffusion_ST, Transformer, Transformer_ST, Model_all, ST_Diffusion
from DSTPP.Dataset import get_dataloader
from DSTPP.Models import build_graph, get_non_pad_mask, CNN
import DSTPP.Constants as Constants


def init_seed(args):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def setup_init(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def model_name():
    TIME = int(time.time())
    TIME = time.localtime(TIME)
    return time.strftime("%Y-%m-%d %H:%M:%S",TIME)

def normalization(x,MAX,MIN):
    return (x-MIN)/(MAX-MIN)


def get_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--mode', type=str, default='train', help='')
    parser.add_argument('--total_epochs', type=int, default=1000, help='')
    parser.add_argument('--machine', type=str, default='none', help='')
    parser.add_argument('--loss_type', type=str, default='l2',choices=['l1','l2','Euclid'], help='')
    parser.add_argument('--beta_schedule', type=str, default='cosine',choices=['linear','cosine'], help='')
    parser.add_argument('--dim', type=int, default=2, help='', choices = [1,2,3])
    parser.add_argument('--dataset', type=str, default='JP_Earthquake',choices=['JP_Earthquake','COVID19','Thefts','311Service','US_Earthquake','Human_mobility','Wildfire','Twitter'], help='')
    parser.add_argument('--batch_size', type=int, default=64,help='')
    parser.add_argument('--timesteps', type=int, default=100, help='')
    parser.add_argument('--samplingsteps', type=int, default=100, help='')
    parser.add_argument('--cycle_len', type=int, default=30, help='Length of Temporal query')
    parser.add_argument('--objective', type=str, default='pred_noise', help='')
    parser.add_argument('--cuda_id', type=str, default='0', help='')
    parser.add_argument('--lr_init', type=float, default=3e-4)
    parser.add_argument('--lr_decay', action='store_true')  
    parser.add_argument('--lr_decay_step', type=str, default='50,100,150')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--type', type=str, default='train',choices=['train','val','test'], help='')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args

opt = get_args()
device = torch.device("cuda:{}".format(opt.cuda_id) if opt.cuda else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_id)

def add_graph_emb(data, path):

    embs = torch.load(path)
    data_with_emb = []

    for sequence, z in zip(data, embs):
        new_sequence = []

        for event, emb in zip(sequence, z):
            event_with_emb = event + emb.tolist()
            new_sequence.append(event_with_emb)
        data_with_emb.append(new_sequence)

    return data_with_emb
    

def data_loader(writer):

    f = open('dataset/{}/data_train.pkl'.format(opt.dataset),'rb')
    train_data = pickle.load(f)
    train_data = [[list(i) for i in u] for u in train_data]
    train_data = [[[i[0], (lambda dt: (1e-6 if dt == 0 else dt))(i[0] - u[idx-1][0] if idx > 0 else i[0])] + i[1:] for idx, i in enumerate(u)] for u in train_data]
    train_data = [[[(1e-6 if ev[0] == 0 else ev[0])] + ev[1:] for ev in seq] for seq in train_data]
    
    f = open('dataset/{}/data_val.pkl'.format(opt.dataset),'rb')
    val_data = pickle.load(f)
    val_data = [[list(i) for i in u] for u in val_data]
    val_data = [[[i[0], (lambda dt: (1e-6 if dt == 0 else dt))(i[0] - u[idx-1][0] if idx > 0 else i[0])] + i[1:] for idx, i in enumerate(u)] for u in val_data]
    val_data = [[[(1e-6 if ev[0] == 0 else ev[0])] + ev[1:] for ev in seq] for seq in val_data]

    f = open('dataset/{}/data_test.pkl'.format(opt.dataset),'rb')
    test_data = pickle.load(f)
    test_data = [[list(i) for i in u] for u in test_data]
    test_data = [[[i[0], (lambda dt: (1e-6 if dt == 0 else dt))(i[0] - u[idx-1][0] if idx > 0 else i[0])] + i[1:] for idx, i in enumerate(u)] for u in test_data]
    test_data = [[[(1e-6 if ev[0] == 0 else ev[0])] + ev[1:] for ev in seq] for seq in test_data]
    
    
    l_train = len(train_data)
    l_val = len(val_data)
    l_test = len(test_data)

    data_all = train_data+test_data+val_data

    Max, Min = [], []
    for m in range(opt.dim+2): # opt.dim is spatial dimension (e.g., 2), so +2 includes time and delta_time
        if m > 0:
            Max.append(max([i[m] for u in data_all for i in u]))
            Min.append(min([i[m] for u in data_all for i in u]))
        else:
            Max.append(1)
            Min.append(0)

    assert Min[1] > 0
    
    train_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in train_data]
    test_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in test_data]
    val_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in val_data]

    train_data = add_graph_emb(train_data, "GraphData/{}_graph_train_embs.pt".format(opt.dataset))
    test_data = add_graph_emb(test_data, "GraphData/{}_graph_test_embs.pt".format(opt.dataset))
    val_data = add_graph_emb(val_data, "GraphData/{}_graph_val_embs.pt".format(opt.dataset))

    train_pis = np.load(f"./PIS/{opt.dataset}/{opt.dataset}_train_pis.npz")['arr_0']
    train_pis = train_pis.reshape(len(train_data), 1, 50, 50)
    val_pis = np.load(f"./PIS/{opt.dataset}/{opt.dataset}_val_pis.npz")['arr_0']
    val_pis = val_pis.reshape(len(val_data), 1, 50, 50)
    test_pis = np.load(f"./PIS/{opt.dataset}/{opt.dataset}_test_pis.npz")['arr_0']
    test_pis = test_pis.reshape(len(test_data), 1, 50, 50)

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True,  pis=train_pis)
    testloader = get_dataloader(test_data, len(test_data) if len(test_data)<=1000 else 1000, shuffle=False, pis=test_pis)
    valloader = get_dataloader(val_data, len(val_data) if len(val_data)<=1000 else 1000, shuffle=False, pis=val_pis)

    return trainloader, testloader, valloader, (Max,Min), l_train, l_test, l_val


def Batch2toModel(batch, transformer):

    event_time_origin, event_time, lng, lat, emb1, emb2, emb3, emb4, emb5, emb6, emb7, emb8, pis = map(lambda x: x.to(device), batch)
    event_loc = torch.cat((lng.unsqueeze(dim=2),lat.unsqueeze(dim=2)),dim=-1)

    event_time = event_time.to(device)
    event_time_origin = event_time_origin.to(device)
    event_loc = event_loc.to(device)
    
    enc_out, mask = transformer(event_loc, event_time_origin, pis)

    enc_out_non_mask  = []
    event_time_non_mask = []
    event_loc_non_mask = []
    for index in range(mask.shape[0]):
        length = int(sum(mask[index]).item()) # number of valid events in sequence
        if length>1:
            enc_out_non_mask += [i.unsqueeze(dim=0) for i in enc_out[index][:length-1]]
            event_time_non_mask += [i.unsqueeze(dim=0) for i in event_time[index][1:length]]
            event_loc_non_mask += [i.unsqueeze(dim=0) for i in event_loc[index][1:length]]

    enc_out_non_mask = torch.cat(enc_out_non_mask,dim=0)
    event_time_non_mask = torch.cat(event_time_non_mask,dim=0)
    event_loc_non_mask = torch.cat(event_loc_non_mask,dim=0)

    event_time_non_mask = event_time_non_mask.reshape(-1,1,1)
    event_loc_non_mask = event_loc_non_mask.reshape(-1,1,opt.dim)
    
    enc_out_non_mask = enc_out_non_mask.reshape(event_time_non_mask.shape[0],1,-1)

    return event_time_non_mask, event_loc_non_mask, enc_out_non_mask


def emb_reshape(emb_list):

    graph_emb = None

    for emb in emb_list:
        mask = get_non_pad_mask(emb)
        emb_non_mask = []

        for index in range(mask.shape[0]):
            length = int(sum(mask[index]).item())
            if length>1:
                emb_non_mask += [i.unsqueeze(dim=0) for i in emb[index][:length-1]]

        emb_non_mask = torch.cat(emb_non_mask,dim=0)
        emb_non_mask = emb_non_mask.reshape(emb_non_mask.shape[0],1,-1)
        if graph_emb is None:
            graph_emb = emb_non_mask
        else:
            graph_emb = torch.cat([graph_emb, emb_non_mask], dim=-1)

    graph_emb = graph_emb.to(device)

    return graph_emb
        

def LR_warmup(lr, epoch_num, epoch_current):
    return lr * (epoch_current+1) / epoch_num


if __name__ == "__main__":
    setup_init(opt)
    init_seed(opt)
    setproctitle.setproctitle("Model-Training")

    print('dataset:{}'.format(opt.dataset))
    
    # Specify a directory for logging data 
    logdir = "./logs/Dataset_{}_timesteps_{}".format( opt.dataset, opt.timesteps)
    model_path = './ModelSave/Dataset_{}_timesteps_{}/'.format(opt.dataset, opt.timesteps) 

    if not os.path.exists('./ModelSave'):
            os.mkdir('./ModelSave')

    if 'train' in opt.mode and not os.path.exists(model_path):
        os.mkdir(model_path)

    writer = SummaryWriter(log_dir = logdir, flush_secs=5)

    model = ST_Diffusion(
        n_steps = opt.timesteps,
        dim = 1 + opt.dim,
        condition = True,
        cond_dim = 64
    ).to(device)

    diffusion = GaussianDiffusion_ST(
        model,
        loss_type = opt.loss_type,
        seq_length = 1+opt.dim,
        timesteps = opt.timesteps,
        sampling_timesteps = opt.samplingsteps,
        objective = opt.objective,
        beta_schedule = opt.beta_schedule
    ).to(device)

    transformer = Transformer_ST(
        d_model=64,
        d_rnn=256,
        d_inner=128,
        n_layers=4,
        n_head=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
        device=device,
        loc_dim = opt.dim,
        CosSin = True,
        cycle_len = opt.cycle_len
    ).to(device)

    Model = Model_all(transformer,diffusion)

    trainloader, testloader, valloader, (MAX,MIN), l_train, l_test, l_val = data_loader(writer)

    warmup_steps = 5
    
    # training
    optimizer = AdamW(Model.parameters(), lr = 3e-4, betas = (0.9, 0.99))
    step, early_stop = 0, 0
    min_loss_test = 1e20

    os.makedirs("./ModelResult", exist_ok=True)
    results_path = "./ModelResult/{}.txt".format(opt.dataset)
    
    with open(results_path, 'a') as f:
        f.write("Test Results\n")

    best_rmse_t, best_mae_s, best_rmse_mean_t, best_mae_mean_s, best_nll_t, best_nll_s = float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')
    best_rmse_t_epo, best_mae_s_epo, best_rmse_mean_t_epo, best_mae_mean_s_epo, best_nll_t_epo, best_nll_s_epo = 0, 0, 0, 0, 0, 0
    train_time = 0

    for itr in range(opt.total_epochs):

        print('epoch:{}'.format(itr))

        if itr % 10==0:
            print('Evaluate!')
            with torch.no_grad():
                
                Model.eval()
                
                # validation set
                loss_test_all, vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0, 0.0
                mae_temporal, rmse_temporal, mae_spatial, mae_lng, mae_lat, total_num, rmse_temporal_mean, mae_spatial_mean = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                
                for batch in valloader:
                    event_time, event_time_norm, lng, lat, *emb_list, pis = batch
                    graph_emb = emb_reshape(emb_list)
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)
                    enc_out_non_mask = torch.cat([enc_out_non_mask, graph_emb], dim=-1)
            
                    sampled_seq_temporal_all, sampled_seq_spatial_all = [], []
                    sampled_seq = Model.diffusion.sample(batch_size = event_time_non_mask.shape[0],cond=enc_out_non_mask)
                    loss = Model.diffusion(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)

                    vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask) # Negative log-likelihood
                    vb_test_all += vb
                    vb_test_temporal_all += vb_temporal
                    vb_test_spatial_all += vb_spatial

                    loss_test_all += loss.item() * event_time_non_mask.shape[0]
                    
                    real = event_time_non_mask[:,0,:].detach().cpu() * (MAX[1]-MIN[1]) + MIN[1]
                    gen = sampled_seq[:,0,:1].detach().cpu() * (MAX[1]-MIN[1])+ MIN[1]
                    assert real.shape==gen.shape
                    mae_temporal += torch.abs(real-gen).sum().item()
                    rmse_temporal += ((real-gen)**2).sum().item()

                    real = event_loc_non_mask[:,0,:].detach().cpu()
                    assert real.shape[1:] == torch.tensor(MIN[2:]).shape
                    real = real * (torch.tensor([MAX[2:]])-torch.tensor([MIN[2:]]))+ torch.tensor([MIN[2:]])
                    gen = sampled_seq[:,0,-opt.dim:].detach().cpu()
                    gen = gen * (torch.tensor([MAX[2:]])-torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])
                    assert real.shape==gen.shape
                    mae_spatial += torch.sqrt(torch.sum((real-gen)**2,dim=-1)).sum().item()
                    
                    total_num += gen.shape[0]

                    assert gen.shape[0] == event_time_non_mask.shape[0]

                print("Validation RMSE Temporal:", np.sqrt(rmse_temporal/total_num))
                print("Validation MAE Spatial:", mae_spatial/total_num)

                if loss_test_all > min_loss_test:
                    early_stop += 1
                    if early_stop >= 200:
                        break
                
                else:
                    early_stop = 0
                
                torch.save(Model.state_dict(), model_path+'model_{}.pkl'.format(itr))

                min_loss_test = min(min_loss_test, loss_test_all)

                writer.add_scalar(tag='Evaluation/loss_val',scalar_value=loss_test_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_val',scalar_value=vb_test_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_temporal_val',scalar_value=vb_test_temporal_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_spatial_val',scalar_value=vb_test_spatial_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/mae_temporal_val',scalar_value=mae_temporal/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/rmse_temporal_val',scalar_value=np.sqrt(rmse_temporal/total_num),global_step=itr)
                writer.add_scalar(tag='Evaluation/distance_spatial_val',scalar_value=mae_spatial/total_num,global_step=itr)
                
                # test set
                loss_test_all, vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0, 0.0
                mae_temporal, rmse_temporal, mae_spatial, mae_lng, mae_lat, total_num, rmse_temporal_mean, mae_spatial_mean = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                
                for batch in testloader:
                    event_time, event_time_norm, lng, lat, *emb_list, pis = batch
                    graph_emb = emb_reshape(emb_list)
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)
                    enc_out_non_mask = torch.cat([enc_out_non_mask, graph_emb], dim=-1)

                    sampled_seq_temporal_all, sampled_seq_spatial_all = [], []
                    for _ in range(100):
                        sampled_seq = Model.diffusion.sample(batch_size = event_time_non_mask.shape[0],cond=enc_out_non_mask)
                        sampled_seq_temporal_all.append(sampled_seq[:,0,:1].detach().cpu() * (MAX[1]-MIN[1])+ MIN[1])
                        sampled_seq_spatial_all.append((sampled_seq[:,0,-opt.dim:].detach().cpu() * (torch.tensor([MAX[2:]])-torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])).unsqueeze(dim=1))

                    sampled_seq = Model.diffusion.sample(batch_size = event_time_non_mask.shape[0],cond=enc_out_non_mask)

                    loss = Model.diffusion(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)

                    vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)
                    vb_test_all += vb
                    vb_test_temporal_all += vb_temporal
                    vb_test_spatial_all += vb_spatial

                    loss_test_all += loss.item() * event_time_non_mask.shape[0]

                    real = event_time_non_mask[:,0,:].detach().cpu() * (MAX[1]-MIN[1]) + MIN[1]
                    gen = sampled_seq[:,0,:1].detach().cpu() * (MAX[1]-MIN[1])+ MIN[1]
                    
                    mae_temporal += torch.abs(real-gen).sum().item()
                    rmse_temporal += ((real-gen)**2).sum().item()
                    sampled_seq_temporal_all = torch.stack(sampled_seq_temporal_all, dim=0)
                    rmse_temporal_mean += ((real-sampled_seq_temporal_all.mean(dim=0))**2).sum().item()

                    real = event_loc_non_mask[:,0,:].detach().cpu()
                    real = real * (torch.tensor([MAX[2:]])-torch.tensor([MIN[2:]]))+ torch.tensor([MIN[2:]])
                    gen = sampled_seq[:,0,-opt.dim:].detach().cpu()
                    gen = gen * (torch.tensor([MAX[2:]])-torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])
                    mae_spatial += torch.sqrt(torch.sum((real-gen)**2,dim=-1)).sum().item()
                    sampled_seq_spatial_all = torch.stack(sampled_seq_spatial_all, dim=0)
                    mae_spatial_mean += torch.sqrt(torch.sum((real-sampled_seq_spatial_all.mean(dim=0).squeeze(1))**2,dim=-1)).sum().item()

                    total_num += gen.shape[0]

                print("Test RMSE Temporal (Mean):", np.sqrt(rmse_temporal_mean/total_num))
                print("Test MAE Spatial (Mean):", mae_spatial_mean/total_num)
                print("Test NLL Temporal:", vb_test_temporal_all/total_num)
                print("Test NLL Spatial:", vb_test_spatial_all/total_num)

                if np.sqrt(rmse_temporal/total_num) < best_rmse_t:
                    best_rmse_t = np.sqrt(rmse_temporal/total_num)
                    best_rmse_t_epo = itr

                if mae_spatial/total_num < best_mae_s:
                    best_mae_s = mae_spatial/total_num
                    best_mae_s_epo = itr

                if np.sqrt(rmse_temporal_mean/total_num) < best_rmse_mean_t:
                    best_rmse_mean_t = np.sqrt(rmse_temporal_mean/total_num)
                    best_rmse_mean_t_epo = itr

                if mae_spatial_mean/total_num < best_mae_mean_s:
                    best_mae_mean_s = mae_spatial_mean/total_num
                    best_mae_mean_s_epo = itr

                if vb_test_temporal_all/total_num < best_nll_t:
                    best_nll_t = vb_test_temporal_all/total_num
                    best_nll_t_epo = itr

                if vb_test_spatial_all/total_num < best_nll_s:
                    best_nll_s = vb_test_spatial_all/total_num
                    best_nll_s_epo = itr

                with open(results_path, 'a') as f:
                    f.write(f"Epoch {itr}:\n")
                    f.write(f"Test RMSE Temporal (Mean): {np.sqrt(rmse_temporal_mean / total_num):.6f}\n")
                    f.write(f"Test MAE Spatial (Mean): {mae_spatial_mean / total_num:.6f}\n")
                    f.write(f"Test NLL Temporal: {vb_test_temporal_all/total_num:.6f}\n")
                    f.write(f"Test NLL Spatial: {vb_test_spatial_all/total_num:.6f}\n\n")

                writer.add_scalar(tag='Evaluation/loss_test',scalar_value=loss_test_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_test',scalar_value=vb_test_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_temporal_test',scalar_value=vb_test_temporal_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_spatial_test',scalar_value=vb_test_spatial_all/total_num,global_step=itr)
                
        if itr < warmup_steps:
            for param_group in optimizer.param_groups:
                lr = LR_warmup(1e-3, warmup_steps, itr)
                param_group["lr"] = lr

        else:
            for param_group in optimizer.param_groups:
                lr = 1e-3- (1e-3 - 5e-5)*(itr-warmup_steps)/opt.total_epochs
                param_group["lr"] = lr
                
        writer.add_scalar(tag='Statistics/lr',scalar_value=lr,global_step=itr)

        Model.train()

        loss_all, vb_all, vb_temporal_all, vb_spatial_all, total_num = 0.0, 0.0, 0.0, 0.0, 0.0

        start_time = time.time()
        for batch in trainloader:
            event_time, event_time_norm, lng, lat, *emb_list, pis = batch
            graph_emb = emb_reshape(emb_list) # [N, 1, 8]
            
            event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)
            enc_out_non_mask = torch.cat([enc_out_non_mask, graph_emb], dim=-1)
            loss = Model.diffusion(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1),enc_out_non_mask)

            optimizer.zero_grad()
            loss.backward()

            loss_all += loss.item() * event_time_non_mask.shape[0]
            vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)

            vb_all += vb
            vb_temporal_all += vb_temporal
            vb_spatial_all += vb_spatial

            writer.add_scalar(tag='Training/loss_step',scalar_value=loss.item(),global_step=step)

            torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.)
            optimizer.step() 
            
            step += 1

            total_num += event_time_non_mask.shape[0]

        end_time = time.time()
        train_time += (end_time - start_time)

        with torch.cuda.device("cuda:{}".format(opt.cuda_id)):
            torch.cuda.empty_cache()

        writer.add_scalar(tag='Training/loss_epoch',scalar_value=loss_all/total_num,global_step=itr)
        writer.add_scalar(tag='Training/NLL_epoch',scalar_value=vb_all/total_num,global_step=itr)
        writer.add_scalar(tag='Training/NLL_temporal_epoch',scalar_value=vb_temporal_all/total_num,global_step=itr)
        writer.add_scalar(tag='Training/NLL_spatial_epoch',scalar_value=vb_spatial_all/total_num,global_step=itr)

    with open(results_path, 'a') as f:
        f.write("Performance Evaluation:\n")
        f.write(f"{opt.dataset} --timesteps {opt.timesteps} --samplingsteps {opt.samplingsteps} --batch_size {opt.batch_size}\n") 
        f.write(f"--total_epochs {opt.total_epochs} --loss_type {opt.loss_type} --seed {opt.seed} --cycle_len {opt.cycle_len}\n") 
        f.write(f"Total Train Time: {train_time}\n")
        f.write(f"Test RMSE Temporal (Mean): {best_rmse_mean_t:.4f} Epoch: {best_rmse_mean_t_epo}\n")
        f.write(f"Test MAE Spatial (Mean): {best_mae_mean_s:.4f} Epoch: {best_mae_mean_s_epo}\n")
        f.write(f"Test NLL Temporal: {best_nll_t:.4f} Epoch: {best_nll_t_epo}\n")
        f.write(f"Test NLL Spatial: {best_nll_s:.4f} Epoch: {best_nll_s_epo}\n")
