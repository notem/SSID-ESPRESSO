import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from os.path import join
from tqdm import tqdm
import json
import time
import argparse
import math
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.optim.lr_scheduler import StepLR

from utils.nets.transdfnet import DFNet
from utils.nets.espressonet import EspressoNet
from utils.nets.dcfnet import Conv1DModel
from utils.layers import Mlp
from utils.processor import DataProcessor
from utils.data import *
from utils.loss import *


# enable if NaN or other odd behavior appears
#torch.autograd.set_detect_anomaly(True)
# disable any unnecessary logging / debugging
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# auto-optimize cudnn ops
torch.backends.cudnn.benchmark = True

class TripletDataset(Dataset):
    def __init__(self, inflow_data, outflow_data, chain_targets):
        self.positive_top = True
        self.inflow_data = inflow_data
        self.outflow_data = outflow_data
        self.all_indices = list(range(len(self.inflow_data)))
        random.shuffle(self.all_indices)
        self.chain_targets = chain_targets

        # Divide the shuffled indices into two partitions
        cutoff = len(self.all_indices) // 2
        self.partition_1 = self.all_indices[:cutoff]
        self.partition_2 = self.all_indices[cutoff:]

    def __len__(self):
        return len(self.inflow_data)

    def __getitem__(self, idx):
        # Choose a positive from partition 1 and a negative from partition 2 (or vice versa)
        if self.positive_top:
            idx = random.choice(self.partition_1)
            negative_idx = random.choice([j for j in self.partition_2 if j != idx])
        else:
            idx = random.choice(self.partition_2)
            negative_idx = random.choice([j for j in self.partition_1 if j != idx])

        anchor = self.inflow_data[idx]
        positive = self.outflow_data[idx]
        target = self.chain_targets[idx]
        negative = self.outflow_data[negative_idx]

        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            torch.tensor(negative, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )

    def reset_split(self):
        self.positive_top = not self.positive_top

        # Reshuffle the indices at the start of each epoch
        random.shuffle(self.all_indices)

        # Re-divide the shuffled indices into two partitions
        cutoff = len(self.all_indices) // 2
        self.partition_1 = self.all_indices[:cutoff]
        self.partition_2 = self.all_indices[cutoff:]


class OnlineTripletDataset(Dataset):
    def __init__(self, inflow_data, outflow_data, chain_targets):
        self.positive_top = True
        self.inflow_data = inflow_data
        self.outflow_data = outflow_data
        self.all_indices = list(range(len(self.inflow_data)))
        random.shuffle(self.all_indices)
        self.size = len(self.all_indices)
        self.chain_targets = chain_targets

    def __len__(self):
        return len(self.inflow_data)

    def __getitem__(self, trace_idx):
        # Pick a random inflow, outflow pair
        #trace_idx = torch.randint(low=0, high=self.size, size=(1,), dtype=torch.int32)
        anchor = self.inflow_data[trace_idx]
        positive = self.outflow_data[trace_idx]
        target = self.chain_targets[trace_idx]

        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )

    def reset_split(self):
        self.positive_top = not self.positive_top

        # Reshuffle the indices at the start of each epoch
        random.shuffle(self.all_indices)

        # Re-divide the shuffled indices into two partitions
        cutoff = len(self.all_indices) // 2
        self.partition_1 = self.all_indices[:cutoff]
        self.partition_2 = self.all_indices[cutoff:]


def parse_args():
    parser = argparse.ArgumentParser(
                        #prog = 'WF Benchmark',
                        #description = 'Train & evaluate WF attack model.',
                        #epilog = 'Text at the bottom of help'
                    )

    # Main experiment configuration options
    parser.add_argument('--data', required = True, type = str,
                        help = "Path to dataset pickle file.",
                    )
    parser.add_argument('--mode', 
                        type = str, default = 'network-ends',
                        choices=['same-host','network-ends','network-all'],
                        help = "Data mode to use for training.", 
                    )
    parser.add_argument('--ckpt_dir',
                        default = './checkpoint', type = str,
                        help = "Set directory for model checkpoints."
                    )
    parser.add_argument('--cache_dir',
                        default = './cache', type = str,
                        help = "Directory to use to store cached feature files."
                    )
    parser.add_argument('--results_dir', 
                        default = './results',type = str,
                        help = "Set directory for result logs."
                    )
    parser.add_argument('--ckpt', 
                        default = None, type = str,
                        help = "Resume from checkpoint path."
                    )
    parser.add_argument('--exp_name',
                        type = str,
                        default = f'{time.strftime("%Y%m%d-%H%M%S")}',
                        help = ""
                    )
    parser.add_argument('--ckpt_epoch',
                        type=int,
                        default=40)
    
    # Loss-related arguments
    parser.add_argument('--online', 
                        default = False, action = 'store_true',
                        help = "Use online semi-hard triplet mining."
                    )
    parser.add_argument('--hard', 
                        default = False, action = 'store_true',
                        help = "Use hard triplet mining."
                    )
    parser.add_argument('--host', 
                        default = False, action = 'store_true',
                        help = "Use hard triplet mining."
                    )
    #parser.add_argument('--switch_loss_type', 
    #                    type = int, default = 50, 
    #                    help ='Switch from triplet loss to online hard triplet loss'
    #                )
    parser.add_argument('--margin', default = 0.5, type = float,
                        help = "Loss margin for triplet learning."
                    )
    parser.add_argument('--w', default = 0.0, type = float,
                        help = "Weight placed on the chain-loss of multi-task loss."
                    )
    parser.add_argument(
        '--temporal_alignment', action='store_true', help='Use temporal alignment loss'
    )

    # Model architecture options
    parser.add_argument('--config',
                        default = None, type = str,
                        help = "Set model config (as JSON file)")
    parser.add_argument('--single_fen', 
                        default = False, action = 'store_true',
                        help = 'Use the same FEN for in and out flows.')

    return parser.parse_args()


if __name__ == "__main__":
    """
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()

    # load checkpoint (if it exists)
    checkpoint_path = args.ckpt
    checkpoint_fname = None
    resumed = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        resumed = torch.load(checkpoint_path)
        checkpoint_fname = os.path.basename(os.path.dirname(checkpoint_path))
    # else: checkpoint path and fname will be defined later if missing

    checkpoint_dir = args.ckpt_dir
    results_dir = args.results_dir

    # # # # # #
    # finetune config
    # # # # # #
    batch_size = 64   # samples to fit on GPU
    # # # # # #
    ckpt_period     = args.ckpt_epoch
    epochs          = args.ckpt_epoch * 5
    opt_lr          = 1e-4
    opt_betas       = (0.9, 0.999)
    opt_wd          = 0.001
    save_best_epoch = True
    use_same_fen    = args.single_fen
    loss_margin     = args.margin
    loss_delta      = float(args.w)
    multitask       = loss_delta > 0.
    # # # # # #
    train_params = {
        'batch_size': batch_size,
        'opt_lr': opt_lr,
        'opt_betas': opt_betas,
        'opt_wd': opt_wd,
        'use_same_fen': use_same_fen,
        'loss_margin': loss_margin,
        'loss_delta': loss_delta,
        #'steplr_step': steplr_step,
        #'steplr_gamma': steplr_gamma,
        #'semihard_loss': semihard_loss
    }

    # all trainable network parameters
    params = []

    # DF model config
    if resumed:
        model_config = resumed['config']
    else:
        with open(args.config, 'r') as fi:
            model_config = json.load(fi)

    features = model_config['features']
    feature_dim = model_config['feature_dim']
    window_kwargs = model_config['window_kwargs']
    
    processor = DataProcessor(features, **model_config)
    
    model_config['input_channels'] = processor.input_channels
    #model_config['input_channels'] = 8

    print("==> Model configuration:")
    print(json.dumps(model_config, indent=4))
    
    model_arch = model_config['model']
    if model_arch.lower() == "espresso":
        inflow_fen = EspressoNet(**model_config)
    elif model_arch.lower() == 'dcf':
        inflow_fen = Conv1DModel(input_size = model_config.get('inflow_size', 1000),
                                **model_config)
    else:
        print(f"Invalid model architecture name \'{model_arch}\'!")
        import sys
        sys.exit(-1)

    inflow_fen = inflow_fen.to(device)
    if resumed:
        inflow_fen.load_state_dict(resumed['inflow_fen'])
    params += inflow_fen.parameters()

    if use_same_fen:
        outflow_fen = inflow_fen
    else:
        if model_arch.lower() == "espresso":
            outflow_fen = EspressoNet(**model_config)
        elif model_arch.lower() == 'dcf':
            outflow_fen = Conv1DModel(input_size = model_config.get('inflow_size', 1000),
                                    **model_config)
        outflow_fen = outflow_fen.to(device)
        params += outflow_fen.parameters()
    
    # chain length prediction head
    if multitask:
        in_dim = feature_dim * model_config.get('special_toks',1) * 2
        head = Mlp(dim=in_dim, out_features=1)
        head = head.to(device)
        if resumed:
            head_state_dict = resumed['chain_head']
            head.load_state_dict(head_state_dict)
        params += head.parameters()

    # # # # # #
    # print parameter count of metaformer model (head not included)
    param_count = sum(p.numel() for p in params if p.requires_grad)
    param_count /= 1000000
    param_count = round(param_count, 2)
    print(f'=> Model is {param_count}m parameters large.')
    # # # # # #
    
        
    # # # # # #
    # create data loaders
    # # # # # #
    pklpath = args.data
    
    idx = np.arange(0,10000)
    np.random.seed(42)
    np.random.shuffle(idx)
    te_idx = idx[0:500]
    va_idx = idx[500:1000]
    tr_idx = idx[1000:10000]
    
    def build_dataset(pklpath, processor, in_idx=1, out_idx=-1):
        chains = load_dataset(pklpath)#, sample_idx)
        inflow, outflow = [],[]
        targets = []
        for chain in tqdm(chains):
            targets.append(len(chain)-3)
            s1 = processor(chain[in_idx])#.numpy(force=True)
            s2 = processor(chain[out_idx])#.numpy(force=True)
            inflow.append(s1)
            outflow.append(s2)
        inflow = np.stack(inflow)#.transpose((0,2,1))
        outflow = np.stack(outflow)#.transpose((0,2,1))
        targets = np.array(targets)
        return inflow, outflow, targets
    
    if args.host:
        out_idx = 2
        in_idx = 1
    else:
        out_idx = -1
        in_idx = 1
    inflow, outflow, targets = build_dataset(pklpath, processor, out_idx=out_idx, in_idx=in_idx)
    #tr_inflow, tr_outflow, tr_targets = build_dataset(pklpath, processor, tr_idx)
    va_inflow = inflow[va_idx]
    va_outflow = outflow[va_idx]
    tr_inflow = inflow[tr_idx]
    tr_outflow = outflow[tr_idx]
    tr_targets = targets[tr_idx]
    va_targets = targets[va_idx]

    if args.online:
        # Define the datasets
        tr_data = OnlineTripletDataset(tr_inflow, tr_outflow, tr_targets)
        va_data = OnlineTripletDataset(va_inflow, va_outflow, va_targets)
    #trainloader, tr_data 
    else:
        tr_data = TripletDataset(tr_inflow, tr_outflow, tr_targets)
        va_data = TripletDataset(va_inflow, va_outflow, va_targets)

    trainloader = DataLoader(
        tr_data,
        batch_size=batch_size,
        num_workers=16,
        shuffle=True,
        pin_memory=True,
    )
    validationloader = DataLoader(
        va_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    """

    if args.mode == 'same-host':
        data_kwargs = {
                    'host_only': True,
                }
    elif args.mode == 'network-ends':
        data_kwargs = {
                'ends_only': True,
                'stream_ID_range': (1,-1),
                }
    elif args.mode == 'network-all':
        data_kwargs = {
                'stream_ID_range': (1,float('inf')),
                }

    def make_dataloader(idx, shuffle=True, 
                        quad_sampler=False, **kwargs):
        dataset = BaseDataset(pklpath, processor,
                            window_kwargs = window_kwargs,
                            preproc_feats = False,
                            sample_idx = idx,
                            save_to_dir = args.cache_dir,
                            **kwargs,
                            )
        if args.online:
            dataset = OnlineDataset(dataset, k=2)

        else:
            dataset = TripletDataset(dataset)
        if quad_sampler:
            loader = DataLoader(dataset,
                                sampler=QuadrupleSampler(dataset),
                                batch_size=batch_size, 
                                collate_fn=dataset.batchify,
                                #shuffle=shuffle
                                )
        else:
            loader = DataLoader(dataset,
                                batch_size=batch_size, 
                                collate_fn=dataset.batchify,
                                shuffle=shuffle)
        return loader, dataset

    # prepare data loaders
    trainloader, tr_data = make_dataloader(tr_idx, 
                                           shuffle=True, 
                                           **data_kwargs)
    validationloader, va_data = make_dataloader(va_idx, 
                                                shuffle=False, 
                                                **data_kwargs)
    """

    # # # # # #
    # optimizer and params, reload from resume is possible
    # # # # # #
    optimizer = optim.AdamW(params,
            lr=opt_lr, betas=opt_betas, weight_decay=opt_wd)
    if resumed and resumed.get('opt', None):
        opt_state_dict = resumed['opt']
        optimizer.load_state_dict(opt_state_dict)

    last_epoch = -1
    if resumed and resumed['epoch']:    # if resuming from a finetuning checkpoint
        last_epoch = resumed['epoch']
    #scheduler = StepLR(optimizer, 
    #                    step_size = ckpt_period * 2, 
    #                    gamma = 0.7,
    #                    last_epoch=last_epoch
    #                )
    def lr_lambda(current_epoch, warmup_epochs=10):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        else:
            return 0.5 * (
                1
                + math.cos(
                    math.pi
                    * (current_epoch - warmup_epochs)
                    / (epochs - warmup_epochs)
                )
            )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # define checkpoint fname if not provided
    if not checkpoint_fname:
        #checkpoint_fname = f'{model_arch}'
        checkpoint_fname = f'{args.exp_name}'

    # create checkpoint directory if necesary
    if not os.path.exists(f'{checkpoint_dir}/{checkpoint_fname}/'):
        try:
            os.makedirs(f'{checkpoint_dir}/{checkpoint_fname}/')
        except:
            pass
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except:
            pass

    reg_criterion = nn.SmoothL1Loss(beta=1.0)
    triplet_hard = OnlineHardCosineTripletLoss(margin = loss_margin, 
                                               semihard = True, 
                                               hard_negative_loss = True
                                               ) 
    triplet_easy = OnlineCosineTripletLoss(margin = loss_margin, 
                                           semihard = True,
                                        )
    triplet_basic = CosineTripletLoss(margin = loss_margin)


    def epoch_iter(dataloader, 
                    eval_only = False, 
                    desc = f"Epoch"):
        """
        Step through one epoch on the dataset
        """
        tot_loss = 0.
        trip_loss = 0.
        loss_a = 0.

        acc = 0
        up_acc = 0
        down_acc = 0

        n = 0
        with tqdm(dataloader,
                desc = desc,
                dynamic_ncols = True) as pbar:
            for batch_idx, data in enumerate(pbar):

                # online loss variant
                if args.online:
                    #inputs, labels, targets = data
                    #inputs = inputs.to(device)
                    #N, S, F = inputs.shape
                    #labels = labels.to(device)
                    #targets = targets.to(device)
                    inflow_inputs, outflow_inputs, targets = data
                    targets = targets.to(device)
                    N = targets.size(0)
                    inflow_inputs = inflow_inputs.to(device)
                    outflow_inputs = outflow_inputs.to(device)
                    
                    """
                    inflow_inputs = inflow_inputs[:,3:,:]
                    outflow_inputs = outflow_inputs[:,3:,:]
                    
                    # Reshape sorted_inputs to (num_unique_labels, 2, sequence_length, features)
                    sorted_labels, sorted_indices = torch.sort(labels, stable=True)
                    sorted_inputs = inputs[sorted_indices]
                    reshaped_inputs = sorted_inputs.view(N//2, 2, S, F)
                    sorted_labels = sorted_labels.view(N//2, 2)
                    assert torch.all(sorted_labels[:,0] == sorted_labels[:,1])
                    
                    # Split into inflow and outflow
                    inflow_inputs = reshaped_inputs[:, 0, :, :]  # First instance of each label
                    outflow_inputs = reshaped_inputs[:, 1, :, :]  # Second instance of each label
                    """

                    # Apply FEN on flows
                    if multitask:
                        inflow_embed, inflow_chain = inflow_fen(inflow_inputs, return_toks=True)
                        outflow_embed, outflow_chain = outflow_fen(outflow_inputs, return_toks=True)
                    else:
                        inflow_embed = inflow_fen(inflow_inputs, return_toks=False)
                        outflow_embed = outflow_fen(outflow_inputs, return_toks=False)
                        
                    #inflow_embed = inflow_embed.permute(0,2,1)
                    #outflow_embed = outflow_embed.permute(0,2,1)
                    
                    # Remerge flows back into a sorted matrix
                    #stacked_embed = torch.stack((inflow_embed, outflow_embed), dim=1)
                    #sorted_embed = stacked_embed.view(N, *stacked_embed.shape[2:])
                    
                    # Calculate triplet loss
                    #triplet_loss = triplet_easy(sorted_embed, sorted_labels)
                    #triplet_loss = triplet_easy(inflow_embed, outflow_embed)
                    #triplet_loss = (0.34*triplet_loss) + (0.66*triplet_hard(inflow_embed, outflow_embed))
                    #if epoch < args.ckpt_epoch:
                    #    triplet_loss = triplet_easy(inflow_embed, outflow_embed)
                    #    triplet_loss += triplet_easy(inflow_embed.permute(0,2,1), outflow_embed.permute(0,2,1))
                    #else:
                    #triplet_loss = triplet_hard(inflow_embed, outflow_embed)
                    #triplet_loss = triplet_hard(inflow_embed.permute(0,2,1), outflow_embed.permute(0,2,1))
                    #triplet_loss /= 2
                    
                    if args.hard:
                        triplet_criterion = triplet_hard
                    else:
                        triplet_criterion = triplet_easy
                    
                    triplet_loss_f = triplet_criterion(inflow_embed, outflow_embed)
                    
                    if args.temporal_alignment:
                        triplet_loss_w = triplet_criterion(inflow_embed.permute(0,2,1), 
                                                           outflow_embed.permute(0,2,1))
                        triplet_loss_a = ((triplet_loss_w - triplet_loss_f)**2).mean()
                        triplet_loss_w = torch.sum(triplet_loss_w) / (torch.count_nonzero(triplet_loss_w) + 1e-6)
                    
                    triplet_loss_f = torch.sum(triplet_loss_f) / (torch.count_nonzero(triplet_loss_f) + 1e-6)
                    
                    if args.temporal_alignment:
                        triplet_loss = (triplet_loss_f + triplet_loss_w) / 2  + triplet_loss_a
                    else:
                        triplet_loss = triplet_loss_f
                    trip_loss += triplet_loss.item()
                    if args.temporal_alignment:
                        loss_a += triplet_loss_a.item()

                    # Calculate chain loss
                    if multitask:
                        #stacked_chain = torch.stack((inflow_chain, outflow_chain), dim=1)
                        #sorted_chain = stacked_chain.view(N, stacked_chain.size(-1))
                        
                        #sorted_targets = targets[sorted_targets]
                        #feature_count = sorted_chain.size(-1)
                        #concatenated = sorted_chain.view(-1, feature_count*2)
                        concatenated = torch.concat((inflow_chain, outflow_chain), dim=-1)
                        pred = head(concatenated)
                        
                        chain_loss = reg_criterion(pred, targets)
                        

                # offline / random triplets
                else:
                    inputs_anc, inputs_pos, inputs_neg, targets = data
                    inputs_anc = inputs_anc.to(device)
                    inputs_pos = inputs_pos.to(device)
                    inputs_neg = inputs_neg.to(device)
                    targets = targets.to(device)
                    #targets = torch.ones(inputs_anc.size(0)).to(device)

                    # # # # # #
                    # generate traffic feature vectors & run triplet loss
                    if multitask:
                        anc_embed, anc_chain = inflow_fen(inputs_anc, return_toks=True)
                        pos_embed, pos_chain = outflow_fen(inputs_pos, return_toks=True)
                        neg_embed, _ = outflow_fen(inputs_neg, return_toks=True)
                    else:
                        anc_embed = inflow_fen(inputs_anc, return_toks=False)
                        pos_embed = outflow_fen(inputs_pos, return_toks=False)
                        neg_embed = outflow_fen(inputs_neg, return_toks=False)
                        
                    #inflow_embed = inflow_embed.permute(0,2,1)
                    #outflow_embed = outflow_embed.permute(0,2,1)
                    triplet_criterion = triplet_basic
                        
                    triplet_loss = triplet_criterion(anc_embed, pos_embed, neg_embed)
                    trip_loss += triplet_loss.item()

                    # # # # #
                    # predict chain length with head & run loss
                    if multitask:
                        pred = head(torch.cat((anc_chain, pos_chain), dim=-1))
                        #pred = head(anc_chain)
                        chain_loss = reg_criterion(pred, targets)

                # combined multi-task loss
                loss = triplet_loss 
                if multitask:
                    loss += (loss_delta * chain_loss)
                tot_loss += loss.item()

                #
                # accuracy predicted as all-or-nothing
                if multitask:
                    #length_pred = torch.round(pred[:,0])
                    #up_acc += torch.sum(length_pred == targets[:,0]).item()

                    #length_pred = torch.round(pred[:,1])
                    #down_acc += torch.sum(length_pred == targets[:,1]).item()

                    #acc = (up_acc + down_acc)/2
                    length_pred = torch.round(pred)
                    acc += torch.sum(length_pred.flatten() == targets.flatten()).item()
                # # # # #

                n += len(targets)

                if not eval_only:
                    # update model weights
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # pbar stuff
                postfix = {
                            'triplet': trip_loss/(batch_idx+1),
                        }
                if loss_delta > 0.:
                    postfix.update({
                            #'up_acc': up_acc/n,
                            #'down_acc': down_acc/n,
                            'acc': acc/n,
                            'tot_loss': tot_loss/(batch_idx+1),
                    })
                if args.temporal_alignment:
                    postfix.update({
                            'triplet_a': loss_a/(batch_idx+1),
                    })
                pbar.set_postfix(postfix)
                                
                last_lr = scheduler.get_last_lr()
                if last_lr and not eval_only:
                    mod_desc = desc + f'[lr={last_lr[0]}]'
                else:
                    mod_desc = desc
                pbar.set_description(mod_desc)
                
            if args.online:
                sims = compute_sim(inflow_embed, outflow_embed, mean=False)
                sims = sims.permute(1,2,0).reshape(-1, sims.size(0))
                print(sims[0])
                sims = compute_sim(inflow_embed.permute(0,2,1), outflow_embed.permute(0,2,1), mean=False)
                sims = sims.permute(1,2,0).reshape(-1, sims.size(0))
                print(sims[0])

                #print(pred)

                
        trip_loss /= batch_idx + 1
        acc /= n
        return trip_loss

    # do training
    history = {}
    try:
        for epoch in range(last_epoch+1, epochs):

            # train and update model using training data
            inflow_fen.train()
            outflow_fen.train()
            if multitask:
                head.train()
            train_loss = epoch_iter(trainloader, 
                                    desc = f"Epoch {epoch} Train")
            metrics = {'tr_loss': train_loss}
            #if not args.online:
            tr_data.reset_split()
            scheduler.step()

            # Evaluate on hold-out data
            inflow_fen.eval()
            outflow_fen.eval()
            if multitask:
                head.eval()
            if validationloader is not None:
                with torch.no_grad():
                    va_loss = epoch_iter(validationloader, 
                                            eval_only = True, 
                                            desc = f"Epoch {epoch} Val.")
                metrics.update({'va_loss': va_loss})

            # Save model
            if (epoch % ckpt_period) == (ckpt_period-1):
                # save last checkpoint before restart
                checkpoint_path_epoch = f"{checkpoint_dir}/{checkpoint_fname}/e{epoch}.pth"
                print(f"Saving end-of-cycle checkpoint to {checkpoint_path_epoch}...")
                torch.save({
                                "epoch": epoch,
                                "inflow_fen": inflow_fen.state_dict(),
                                "outflow_fen": outflow_fen.state_dict() if not use_same_fen else None,
                                "chain_head": head.state_dict() if multitask else None,
                                "opt": optimizer.state_dict(),
                                "config": model_config,
                                "train_config": train_params,
                        }, checkpoint_path_epoch)

            if save_best_epoch:
                best_val_loss = min([999]+[metrics['va_loss'] for metrics in history.values()])
                if metrics['va_loss'] < best_val_loss:
                    checkpoint_path_epoch = f"{checkpoint_dir}/{checkpoint_fname}/best.pth"
                    print(f"Saving new best model to {checkpoint_path_epoch}...")
                    torch.save({
                                    "epoch": epoch,
                                    "inflow_fen": inflow_fen.state_dict(),
                                    "outflow_fen": outflow_fen.state_dict() if not use_same_fen else None,
                                    "chain_head": head.state_dict() if multitask else None,
                                    "opt": optimizer.state_dict(),
                                    "config": model_config,
                                    "train_config": train_params,
                            }, checkpoint_path_epoch)

            history[epoch] = metrics

    except KeyboardInterrupt:
        pass

    finally:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        results_fp = f'{results_dir}/{checkpoint_fname}.txt'
        with open(results_fp, 'w') as fi:
            json.dump(history, fi, indent='\t')
            
        # save a final checkpoint
        checkpoint_path_epoch = f"{checkpoint_dir}/{checkpoint_fname}/final.pth"
        print(f"Saving final checkpoint to {checkpoint_path_epoch}...")
        torch.save({
                        "epoch": epoch,
                        "inflow_fen": inflow_fen.state_dict(),
                        "outflow_fen": outflow_fen.state_dict() if not use_same_fen else None,
                        "chain_head": head.state_dict() if multitask else None,
                        "opt": optimizer.state_dict(),
                        "config": model_config,
                        "train_config": train_params,
                }, checkpoint_path_epoch)

