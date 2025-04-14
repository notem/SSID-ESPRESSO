import torch
import torch.nn.functional as F
import numpy as np
import os
from os.path import join
import pickle as pkl
from tqdm import tqdm
import json
import argparse
import sys

#from transdfnet import DFNet
from utils.nets.transdfnet import DFNet
from utils.nets.espressonet import EspressoNet
from utils.nets.dcfnet import Conv1DModel
from utils.data import BaseDataset, PairwiseDataset, load_dataset
from utils.data import build_dataset
from utils.processor import *
from train_dcf import DFModel




# enable if NaN or other odd behavior appears
#torch.autograd.set_detect_anomaly(True)
# disable any unnecessary logging / debugging
#torch.autograd.set_detect_anomaly(False)
#torch.autograd.profiler.profile(False)
#torch.autograd.profiler.emit_nvtx(False)

# auto-optimize cudnn ops
#torch.backends.cudnn.benchmark = True


def find_closest(packet_time, other_flow_times):
    """
    """
    if other_flow_times.size == 0:
        return 0
    idx = np.searchsorted(other_flow_times, packet_time)
    if idx == len(other_flow_times):
        return other_flow_times[-1]
    elif idx == 0:
        return other_flow_times[0]
    else:
        before = other_flow_times[idx - 1]
        after = other_flow_times[idx]
        return before if abs(packet_time - before) < abs(packet_time - after) else after


def calculate_proportions(outflow, inflow, 
                          thresholds = (0.0001, 0.001, 0.01, 0.03, 0.08, 0.16, 0.32, 0.64)):
    """
    """
    sizes_trace1 = np.abs(outflow[:,1])
    sizes_trace2 = np.abs(inflow[:,1])
    timedirs_trace1 = outflow[:,0]
    timedirs_trace2 = inflow[:,0]

    size_threshold = 50
    sizes_trace1 = np.array(sizes_trace1)
    sizes_trace2 = np.array(sizes_trace2)

    mask1 = (timedirs_trace1 < 0) & (sizes_trace1 >= size_threshold)
    flow1 = -timedirs_trace1[mask1.to(bool)]
    mask2 = (timedirs_trace2 < 0) & (sizes_trace2 >= size_threshold)
    flow2 = -timedirs_trace2[mask2.to(bool)]

    flow1 = np.sort(flow1)
    flow2 = np.sort(flow2)

    if len(flow2) == 0:
        return np.zeros(len(thresholds))
    
    time_diffs = []
    for time in flow1:
        closest_time = find_closest(time, flow2)
        if closest_time is not None:
            time_diffs.append(abs(time - closest_time))
        else:
            time_diffs.append(float(1))

    if np.count_nonzero(flow1) == 0:
        return np.array([0]*8)

    proportions = []
    for threshold in thresholds:
        count = np.sum(np.array(time_diffs) < threshold)
        proportion = count / np.count_nonzero(flow1)
        proportions.append(proportion)

    return np.array(proportions)


def create_drift_features(inflow, outflow):
    """Generate 'drift' features to improve correlation performance.
    """
    download_time_diff = calculate_proportions(outflow, inflow)
    upload_time_diff = calculate_proportions(outflow * -1, inflow * -1)
    return np.concatenate((download_time_diff, 
                           upload_time_diff))


def parse_args():
    parser = argparse.ArgumentParser(
                        #prog = 'WF Benchmark',
                        #description = 'Train & evaluate WF attack model.',
                        #epilog = 'Text at the bottom of help'
                        )

    # experiment configuration options
    parser.add_argument('--data', 
                        type = str,
                        help = "Path to dataset pickle file.",
                        required = True)
    parser.add_argument('--sims_file', 
                        type = str,
                        help = "Path to store calculated similarity vectors.",
                        required = True)
    parser.add_argument('--ckpt', 
                        type = str,
                        help = "Resume from checkpoint path.", 
                        required = True)
    parser.add_argument('--mode', 
                        type = str,
                        default = 'network-ends',
                        choices=['same-host','network-ends','network-all'],
                        help = "Resume from checkpoint path.", 
                        )
    parser.add_argument('--cache_dir',
                        default = './cache', type = str,
                        help = "Directory to use to store cached feature files."
                    )
    #parser.add_argument('--drift_features',
    #                    action='store_true',
    #                    default=False)
    parser.add_argument('--host', 
                        default = False, action = 'store_true',
                        help = "Use hard triplet mining."
                    )
    parser.add_argument('--temporal_alignment', 
                        action = 'store_true', 
                        help = 'Use temporal alignment loss'
                    )

    return parser.parse_args()


if __name__ == "__main__":
    """
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()
    
    sims_dir = os.path.dirname(args.sims_file)
    if not os.path.exists(sims_dir):
        os.makedirs(sims_dir)

    checkpoint_path = args.ckpt
    checkpoint_fname = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        resumed = torch.load(checkpoint_path)
        checkpoint_fname = os.path.basename(os.path.dirname(checkpoint_path))
    else:
        print(f"Failed to load model checkpoint! {checkpoint_path}")
        sys.exit(-1)
    # else: checkpoint path and fname will be defined later if missing

    model_config = resumed['config']
    model_name = model_config.get('model', 'dcf')
    features = model_config['features']
    feature_dim = model_config.get('feature_dim', 64)
    inflow_size = model_config.get('inflow_size', 1000)
    outflow_size = model_config.get('outflow_size', 1000)
    
    # traffic feature extractor
    if model_name.lower() == "espresso":
        inflow_fen = EspressoNet(**model_config)
    elif model_name.lower() == 'dcf':
        #inflow_fen = Conv1DModel(feature_dim, len(features), 
        #                         input_size=inflow_size)
        inflow_fen = DFModel(input_shape=(len(features),1000), 
                             emb_size=feature_dim)
    inflow_fen = inflow_fen.to(device)
    inflow_fen.load_state_dict(resumed['inflow_fen'])
    inflow_fen.eval()

    if resumed['outflow_fen'] is not None:
        if model_name.lower() == "espresso":
            outflow_fen = EspressoNet(**model_config)
        elif model_name.lower() == "dcf":
            #outflow_fen = Conv1DModel(feature_dim, len(features), 
            #                          input_size=outflow_size)
            outflow_fen = DFModel(input_shape=(len(features),1000), 
                                  emb_size=feature_dim)
        outflow_fen = outflow_fen.to(device)
        outflow_fen.load_state_dict(resumed['outflow_fen'])
        outflow_fen.eval()
    else:
        outflow_fen = inflow_fen

    # # # # # #
    # create data loaders
    # # # # # #

    # multi-channel feature processor
    processor = DataProcessor(features, **model_config)
    pklpath = args.data

    # stream window definitions
    window_kwargs = model_config.get('window_kwargs',{
        "window_count": 11,
        "window_width": 5,
        "window_overlap": 3,
        "include_all_window": False
    })
    
    #if args.mode == 'same-host':
    #    # host
    #    data_kwargs = {
    #                'host_only': True,
    #            }
    #elif args.mode == 'network-ends':
    #    # network -- ends-only
    #    data_kwargs = {
    #            'ends_only': True,
    #            'stream_ID_range': (0,-1),
    #            }
    #elif args.mode == 'network-all':
    #    data_kwargs = {
    #            'stream_ID_range': (1,float('inf')),
    #            }
        
    if args.host:
        out_idx = 2
    else:
        out_idx = -1
    inflow, outflow, targets = build_dataset(pklpath, processor, 
                                             inflow_size, outflow_size, 
                                             in_idx = 1,
                                             window_kwargs = window_kwargs,
                                             )
    outflow = np.array([x[out_idx] for x in outflow])
    
    idx = np.arange(0,min(10000,len(inflow)))
    np.random.seed(42)
    np.random.shuffle(idx)
    te_idx = idx[0:500]
    va_idx = idx[500:1000]
    tr_idx = idx[1000:10000]
    va_inflow = inflow[va_idx]
    va_outflow = outflow[va_idx]
    te_inflow = inflow[te_idx]
    te_outflow = outflow[te_idx]
    print(va_inflow.shape, va_outflow.shape)
    print(te_inflow.shape, te_outflow.shape)
    
    """
    # train dataloader
    va_data = BaseDataset(pklpath, processor,
                        window_kwargs = window_kwargs,
                        preproc_feats = False,
                        sample_idx = va_idx,
                        save_to_dir = args.cache_dir,
                        **data_kwargs
                        )
    va_data = PairwiseDataset(va_data)
    va_ratio = len(va_data.uncorrelated_pairs) / len(va_data.correlated_pairs)
    print(f'Tr. data ratio: {va_ratio}')

    # test dataloader
    te_data = BaseDataset(pklpath, processor,
                        window_kwargs = window_kwargs,
                        preproc_feats = False,
                        sample_idx = te_idx,
                        save_to_dir = args.cache_dir,
                        **data_kwargs
                        )
    te_data = PairwiseDataset(te_data)
    te_ratio = len(te_data.uncorrelated_pairs) / len(te_data.correlated_pairs)
    print(f'Te. data ratio: {te_ratio}', 
          len(te_data.uncorrelated_pairs), 
          len(te_data.correlated_pairs))
    """
    
    """
    """
    
    
    #va_data.set_fen(inflow_fen, func, outflow_fen = outflow_fen)
    #te_data.set_fen(inflow_fen, func, outflow_fen = outflow_fen)
    
    def make_embeds(flows, fen):
        with torch.no_grad():
            embeds = []
            for flow in flows:
                flow = torch.tensor(flow).float()
                flow = flow.float().to(device)
                if window_kwargs is None:
                    flow = flow.unsqueeze(0)
                    #flow = flow.view(flow.size(0)*flow.size(1), 
                    #            flow.size(2), flow.size(3))
                e = fen(flow)
                if window_kwargs is not None:
                    #e = e.view(1, flow.size(0), -1)
                    e = e.unsqueeze(0)
                embeds.append(e.detach().cpu()[0])
        embeds = torch.stack(embeds)
        return embeds
    
    #def make_embeds(dataset, fen):
    #    inflow_embeds, outflow_embeds = [],[]
    #    for sample1, sample2, corr in tqdm(dataset):
    #        if not corr: continue
    #        inflow_embeds.append(fen(func(sample1[0][0])).detach().cpu()[0])
    #        outflow_embeds.append(fen(func(sample2[0][0])).detach().cpu()[0])
    #    return inflow_embeds, outflow_embeds
    
    va_inflow_embeds, va_outflow_embeds = make_embeds(va_inflow, inflow_fen), make_embeds(va_outflow, outflow_fen)
    te_inflow_embeds, te_outflow_embeds = make_embeds(te_inflow, inflow_fen), make_embeds(te_outflow, outflow_fen)
    print(va_inflow_embeds.shape, va_outflow_embeds.shape)
    #va_inflow_embeds, va_outflow_embeds = make_embeds(va_data, outflow_fen)
    #te_inflow_embeds, te_outflow_embeds = make_embeds(te_data, outflow_fen)

    #def build_sims(dataset):
    def build_sims(inflow_embeds, outflow_embeds):
        """Iterate over the pair-wise dataset and 
            generate similarity vectors for the embeddings
        """
        sims = []
            
        #print(np.mean(avg_corr), np.mean(avg_uncorr))
        
        # Normalize embeddings
        #inflow_embeddings_norm = inflow_embeds / inflow_embeds.norm(dim=-1, keepdim=True)
        #outflow_embeddings_norm = outflow_embeds / outflow_embeds.norm(dim=-1, keepdim=True)

        # Compute cosine similarities in batch
        # Resulting shape: [num_inflows, num_outflows, 92]
        #similarities = torch.einsum('ikd,jkd->ijk', 
        #                            inflow_embeddings_norm, 
        #                            outflow_embeddings_norm)
        #similarities = similarities.reshape((-1, similarities.shape[-1]))
        
        
        #"""
        #"""
        #inflow_embeds = inflow_embeds.permute(0,2,1)
        #outflow_embeds = outflow_embeds.permute(0,2,1)
        #inflow_embeddings_norm = inflow_embeds / inflow_embeds.norm(dim=-1, keepdim=True)
        #outflow_embeddings_norm = outflow_embeds / outflow_embeds.norm(dim=-1, keepdim=True)
        
        #similarities2 = torch.einsum('ikd,jkd->ijk', 
        #                            inflow_embeddings_norm, 
        #                            outflow_embeddings_norm)
        #similarities2 = similarities2.reshape((-1, similarities2.shape[-1]))
        #similarities = torch.cat((similarities, similarities2), dim=1)
        
        #num_inflows = inflow_embeddings_norm.shape[0]
        #num_outflows = outflow_embeddings_norm.shape[0]
        #inflow_indices = torch.arange(num_inflows).unsqueeze(1)
        #outflow_indices = torch.arange(num_outflows).unsqueeze(0)
        #indicator = (inflow_indices == outflow_indices).float().reshape(-1)
        
        #print(similarities[0])
        
        #return similarities.numpy(), indicator.numpy()
        #return np.stack(sims), np.array(labels)
        
        def compute_sim(in_emb, out_emb):
            """
            """
            # Normalize each vector (element) to have unit norm
            norms = torch.norm(in_emb, p=2, dim=-1, keepdim=True)  # Compute L2 norms
            in_emb = in_emb / norms  # Divide by norms to normalize

            norms = torch.norm(out_emb, p=2, dim=-1, keepdim=True)  # Compute L2 norms
            out_emb = out_emb / norms  # Divide by norms to normalize

            # Compute pairwise cosine similarity of windows
            all_sim = torch.bmm(in_emb.permute(1,0,2), out_emb.permute(1,2,0))
            window_count = all_sim.shape[0]
            all_sim = all_sim.permute(1,2,0).reshape(-1, window_count)

            return all_sim

        sims = compute_sim(inflow_embeds, outflow_embeds)
        if args.temporal_alignment:
            sims2 = compute_sim(inflow_embeds.permute(0,2,1), 
                                outflow_embeds.permute(0,2,1))
            sims_both = torch.concat((sims, sims2), dim=-1)
        else:
            sims_both = sims
        
        num_inflows = inflow_embeds.shape[0]
        num_outflows = outflow_embeds.shape[0]
        inflow_indices = torch.arange(num_inflows).unsqueeze(1)
        outflow_indices = torch.arange(num_outflows).unsqueeze(0)
        indicator = (inflow_indices == outflow_indices).float().reshape(-1)
        
        return sims_both.numpy(), indicator.numpy()

    #va_sims, va_labels = build_sims(va_data)
    #te_sims, te_labels = build_sims(te_data)
    
    va_sims, va_labels = build_sims(va_inflow_embeds, va_outflow_embeds)
    te_sims, te_labels = build_sims(te_inflow_embeds, te_outflow_embeds)
    print(va_sims)
    
    
    """
    if args.drift_features:
        
        # dataloaders for drift 
        #  (loading is deterministic so samples should be 
        #       in the same order as the previous datasets)
        drift_processor = DataProcessor(('time_dirs', 'sizes'))
        va_data_drift = BaseDataset(pklpath, drift_processor,
                                window_kwargs = None,
                                preproc_feats = False,
                                sample_idx = va_idx,
                                **data_kwargs
                            )
        va_data_drift = PairwiseDataset(va_data_drift)
        te_data_drift = BaseDataset(pklpath, drift_processor,
                                window_kwargs = None,
                                preproc_feats = False,
                                sample_idx = te_idx,
                                **data_kwargs
                            )
        te_data_drift = PairwiseDataset(te_data_drift)
        
        def build_feats(dataset):
            feats = []
            for sample1, sample2, _ in tqdm(dataset):
                feats.append(create_drift_features(sample1[0][0], 
                                                   sample2[0][0]))
            return np.stack(feats)
        
        # generate features and concatenate them to the sims vector
        va_feats = build_feats(va_data_drift)
        va_sims = np.concatenate((va_sims, va_feats), axis=1)
        te_feats = build_feats(te_data_drift)
        te_sims = np.concatenate((te_sims, te_feats), axis=1)
    """
    
    # store data to file for later benchmarking
    with open(args.sims_file, 'wb') as fi:
        pkl.dump({'va_sims': va_sims.astype(np.float16), 
                  'te_sims': te_sims.astype(np.float16),
                  'va_labels': va_labels.astype(bool),
                  'te_labels': te_labels.astype(bool)}, fi)