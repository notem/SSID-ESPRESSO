import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import sys
import os
from os.path import join
from tqdm import tqdm
import json
import argparse
from torch.utils.data import DataLoader, Dataset

from utils.nets.transdfnet import DFNet
from utils.nets.espressonet import EspressoNet
from utils.nets.dcfnet import Conv1DModel
from utils.layers import Mlp
from utils.data import BaseDataset, PairwiseDataset, load_dataset
from utils.data import build_dataset, OnlineTripletDataset
from utils.processor import DataProcessor
from sklearn.metrics.pairwise import pairwise_distances



# enable if NaN or other odd behavior appears
#torch.autograd.set_detect_anomaly(True)
# disable any unnecessary logging / debugging
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# auto-optimize cudnn ops
torch.backends.cudnn.benchmark = True



def parse_args():
    parser = argparse.ArgumentParser(
                        #prog = 'WF Benchmark',
                        #description = 'Train & evaluate WF attack model.',
                        #epilog = 'Text at the bottom of help'
                        )

    # experiment configuration options
    parser.add_argument('--ckpt', 
                        default = None, 
                        type = str,
                        help = "Resume from checkpoint path.", 
                        required=True)
    parser.add_argument('--data', 
                        type = str,
                        help = "Path to dataset pickle file.",
                        required = True)
    parser.add_argument('--outfile',
                        type=str,
                        help = 'Output file for results.',
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    """
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()

    # load checkpoint (if it exists)
    checkpoint_path = args.ckpt
    checkpoint_fname = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        resumed = torch.load(checkpoint_path)
        checkpoint_fname = os.path.basename(os.path.dirname(checkpoint_path))
    else:
        print("Failed to load model checkpoint!")
        sys.exit(-1)
    # else: checkpoint path and fname will be defined later if missing
    
    dirpath = os.path.dirname(args.outfile)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    model_config = resumed['config']
    model_name = model_config.get('model', 'dcf')
    features = model_config['features']
    feature_dim = model_config['feature_dim']
    inflow_size = model_config.get('inflow_size', 1000)
    outflow_size = model_config.get('outflow_size', 1000)
    
    print(json.dumps(model_config, indent=4))

    # traffic feature extractor
    if model_name.lower() == "espresso":
        inflow_fen = EspressoNet(input_channels=len(features),
                                **model_config)
    elif model_name.lower() == 'dcf':
        inflow_fen = Conv1DModel(input_channels=len(features),
                                input_size = model_config.get('inflow_size', 1000),
                                **model_config)
    inflow_fen = inflow_fen.to(device)
    inflow_fen.load_state_dict(resumed['inflow_fen'])
    inflow_fen.eval()

    if model_name.lower() == "espresso":
        outflow_fen = EspressoNet(input_channels=len(features),
                                **model_config)
    elif model_name.lower() == "dcf":
        outflow_fen = Conv1DModel(input_channels=len(features),
                                input_size = model_config.get('outflow_size', 1600),
                                **model_config)
    outflow_fen = outflow_fen.to(device)
    outflow_fen.load_state_dict(resumed['outflow_fen'])
    outflow_fen.eval()

    # chain length prediction head
    head = Mlp(dim=feature_dim, out_features=2)
    head = head.to(device)
    head_state_dict = resumed['chain_head']
    head.load_state_dict(head_state_dict)
    head.eval()

    # # # # # #
    # create data loaders
    # # # # # #

    # multi-channel feature processor
    processor = DataProcessor(features)
    
    idx = np.arange(0,10000)
    np.random.seed(42)
    np.random.shuffle(idx)
    te_idx = idx[0:500]
    va_idx = idx[500:1000]
    tr_idx = idx[1000:10000]
    
    if args.host:
        out_idx = 2
        in_idx = 1
    else:
        out_idx = -1
        in_idx = 1
    inflow, outflow, targets = build_dataset(args.data, processor, 
                                             inflow_size, outflow_size, 
                                             in_idx = in_idx)
    outflow = np.array([x[out_idx] for x in outflow])
    va_inflow = inflow[va_idx]
    va_outflow = outflow[va_idx]
    va_targets = targets[va_idx]
    te_inflow = inflow[te_idx]
    te_outflow = outflow[te_idx]
    te_targets = targets[te_idx]
    print(va_inflow.shape, va_outflow.shape)
    print(te_inflow.shape, te_outflow.shape)
    
    batch_size = 128
    va_data = OnlineTripletDataset(va_inflow, va_outflow, va_targets)
    valoader = DataLoader(
        va_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    te_data = OnlineTripletDataset(te_inflow, te_outflow, te_targets)
    teloader = DataLoader(
        va_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    
    
    def iter(dataloader, desc=""):
        with tqdm(dataloader,
                desc = desc,
                dynamic_ncols = True) as pbar:
            
            n = 0
            all_res = {'targets': [], 'preds': []}
            for batch_idx, data in enumerate(pbar):

                inflow_inputs, outflow_inputs, targets = data
                targets = targets.to(device)
                N = targets.size(0)
                inflow_inputs = inflow_inputs.to(device)
                outflow_inputs = outflow_inputs.to(device)
                
                # Apply FEN on flows
                _, inflow_chain = inflow_fen(inflow_inputs, return_toks=True)
                _, outflow_chain = outflow_fen(outflow_inputs, return_toks=True)
                
                concatenated = torch.concat((inflow_chain, outflow_chain), dim=-1)
                pred = head(concatenated)
                
                all_res['targets'].append(targets.cpu().numpy())
                all_res['preds'].append(pred.cpu().numpy())

                up_pred = pred[:,1]
                up_targets = targets[:,1]
                up_length_pred = torch.round(up_pred)
                up_acc += torch.sum(up_length_pred == up_targets).item()

                down_pred = pred[:,0]
                down_targets = targets[:,0]
                down_length_pred = torch.round(down_pred)
                down_acc += torch.sum(down_length_pred == down_targets).item()
                
                n += len(targets)
                
            all_targets = np.concatenate(all_res['targets'])
            all_preds = np.concatenate(all_res['preds'])
            all_error = all_targets - all_preds
            print(f'Error: {np.mean(all_error):0.3f}|{np.std(all_error):0.3f}')
            # up accuracy
            print(f'Up accuracy: {up_acc/n:.4f}')
            # down accuracy
            print(f'Down accuracy: {down_acc/n:.4f}')
        
            metrics = {'up_acc': up_acc/n, 'down_acc': down_acc/n}
            # add error and per-chain accuracy to metrics
            metrics.update({'error': np.mean(all_error), 'error_std': np.std(all_error)})
        
            # for both up and down, print the accuracy calculated for each unique chain length
            for chain_length in np.unique(all_targets[:,1]):
                idx = all_targets[:,1] == chain_length
                up_acc = np.sum(np.round(all_preds[idx,1]) == all_targets[idx,1]) / np.sum(idx)
                print(f'Up accuracy for chain length {chain_length}: {up_acc:.4f}')
                # print error
                error = all_targets[idx,1] - all_preds[idx,1]
                print(f'Error for chain length {chain_length}: {np.mean(error):0.3f}|{np.std(error):0.3f}')
                metrics.update({f'up_acc_{chain_length}': up_acc, f'up_error_{chain_length}': np.mean(error), f'up_error_std_{chain_length}': np.std(error)})
            for chain_length in np.unique(all_targets[:,0]):
                idx = all_targets[:,0] == chain_length
                down_acc = np.sum(np.round(all_preds[idx,0]) == all_targets[idx,0]) / np.sum(idx)
                print(f'Down accuracy for chain length {chain_length}: {down_acc:.4f}')
                # print error
                error = all_targets[idx,0] - all_preds[idx,0]
                print(f'Error for chain length {chain_length}: {np.mean(error):0.3f}|{np.std(error):0.3f}')
                metrics.update({f'down_acc_{chain_length}': down_acc, f'down_error_{chain_length}': np.mean(error), f'down_error_std_{chain_length}': np.std(error)})

    va_metrics = iter(valoader, desc="Validation")
    te_metrics = iter(teloader, desc="Test")
    
    print(json.dumps(te_metrics, indent=4))
    
    with open(args.outfile, 'w') as f:
        json.dump(te_metrics, f, indent=4)