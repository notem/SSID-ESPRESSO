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
from utils.data import build_dataset, create_windows
from utils.processor import *
from train_dcf import DFModel


# Hack in istarmap to multiprocessing for Python 3.8+
import multiprocessing.pool as mpp
from multiprocessing import cpu_count

def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = istarmap


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
    parser.add_argument('--cache_dir',
                        default = './cache', type = str,
                        help = "Directory to use to store cached feature files."
                    )
    parser.add_argument(
        '--temporal_alignment', action='store_true', help='Use temporal alignment loss'
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
        print("Failed to load model checkpoint!")
        sys.exit(-1)
    # else: checkpoint path and fname will be defined later if missing

    model_config = resumed['config']
    model_name = model_config.get('model', 'dcf')
    features = model_config['features']
    feature_dim = model_config['feature_dim']
    input_size = 200
    inflow_size = model_config.get('inflow_size', input_size)
    outflow_size = model_config.get('outflow_size', input_size)
    
    # traffic feature extractor
    if model_name.lower() == "espresso":
        inflow_fen = EspressoNet(**model_config)
    elif model_name.lower() == 'dcf':
        #inflow_fen = DFNet(feature_dim, len(features), **model_config)
        inflow_fen = DFModel(input_shape=(len(features),inflow_size), 
                             emb_size=feature_dim)
    inflow_fen = inflow_fen.to(device)
    inflow_fen.load_state_dict(resumed['inflow_fen'])
    inflow_fen.eval()

    if resumed['outflow_fen'] is not None:
        if model_name.lower() == "espresso":
            outflow_fen = EspressoNet(**model_config)
        elif model_name.lower() == "dcf":
            #outflow_fen = DFNet(feature_dim, len(features),
            #                    **model_config)
            outflow_fen = DFModel(input_shape=(len(features),outflow_size), 
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
    
    window_kwargs = model_config.get('window_kwargs',{
        "window_count": 11,
        "window_width": 5,
        "window_overlap": 3,
        "include_all_window": False
    })

    # stream window definitions
    #window_kwargs = model_config['window_kwargs']
    
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
    
    # Helper function to process a single chain
    def process_chain(i, chain, size, processor):
        target = len(chain) - 2
        
        if isinstance(chain[1], list):
            s = [[processor(chain[j][k]) for k in range(len(chain[j]))] 
                    for j in range(1, len(chain))]
            
            for j in range(len(s)):
                for k in range(len(s[j])):
                    if len(s[j][k]) < size:
                        s[j][k] = np.pad(s[j][k], ((0, size - len(s[j][k])), (0, 0)))
                    else:
                        s[j][k] = s[j][k][:size]
        else:
            s = [processor(chain[j]) for j in range(1, len(chain)-1)]

            # Pad or trim each element in s2 to match outflow_size
            for j in range(len(s)):
                if len(s[j]) < size:
                    s[j] = np.pad(s[j], ((0, size - len(s[j])), (0, 0)))
                else:
                    s[j] = s[j][:size]

        return s, target

    def build_dataset(pklpath, processor, size, n_jobs=8):
        # load_dataset must be defined elsewhere
        chains = load_dataset(pklpath)
        
        if window_kwargs is not None:
            windowized_chains = []
            for chain in chains:
                sample_windows = []
                for sample in chain:
                    times = sample[:, 0]
                    windows = create_windows(times, sample, **window_kwargs)
                    sample_windows.append(windows)
                windowized_chains.append(sample_windows)
            chains = windowized_chains

        # Prepare arguments for each chain; limit to 10000 chains
        tasks = []
        for i, chain in enumerate(chains):
            if i >= 10000:
                break
            tasks.append((i, chain, size, processor))

        # Process chains in parallel using a multiprocessing Pool
        flows, chain_id, host_id = [], [], []
        j = 0
        with mpp.Pool(min(cpu_count(), n_jobs)) as pool:
            for s2, t in tqdm(pool.istarmap(process_chain, tasks, chunksize=16), 
                                  total = len(tasks)):
                flows.append(np.stack(s2))#.transpose((0,2,1)))
                chain_id.append(np.ones(len(s2))*j)
                host_id.append(np.arange(len(s2))//2)
                j += 1
                #targets.append(t)

        # Stack and transpose as needed
        #flows = np.array([np.stack(samples).transpose((0, 2, 1)) for samples in flows], dtype=object)
        #flows = np.stack(flows).transpose((0, 2, 1))
        #targets = np.array(targets)
        if window_kwargs is not None:
            # Stack and transpose as needed
            flows = np.array([np.stack(samples).transpose((0, 1, 3, 2)) for samples in flows], dtype=object)
        else:
            flows = np.array([np.stack(samples).transpose((0, 2, 1)) for samples in flows], dtype=object)

        #targets = np.array(targets,dtype=object)

        return flows, np.array(chain_id,dtype=object), np.array(host_id, dtype=object)
        #return np.array(flows,dtype=object), np.array(chain_id,dtype=object), np.array(host_id,dtype=object)

    flows, chain_id, host_id = build_dataset(pklpath, processor, max(inflow_size, outflow_size))
    
    idx = np.arange(0,min(10000, len(flows)))
    np.random.seed(42)
    np.random.shuffle(idx)
    te_idx = idx[0:500]
    va_idx = idx[500:1000]
    tr_idx = idx[1000:10000]
    
    va_chain_flows = flows[va_idx].tolist()
    va_chain_labels = chain_id[va_idx].tolist()
    va_host_labels = host_id[va_idx].tolist()
    te_chain_flows = flows[te_idx].tolist()
    te_chain_labels = chain_id[te_idx].tolist()
    te_host_labels = host_id[te_idx].tolist()
    
    va_chain_flows = np.concatenate(va_chain_flows)
    te_chain_flows = np.concatenate(te_chain_flows)
    va_chain_labels = np.concatenate(va_chain_labels)
    te_chain_labels = np.concatenate(te_chain_labels)
    va_host_labels = np.concatenate(va_host_labels)
    te_host_labels = np.concatenate(te_host_labels)
    
    # generate embeddings
    def func(t):
        """Custom func. to handle padding and batching samples for set_fen()"""
        #t = torch.nn.utils.rnn.pad_sequence(t, 
        #                                batch_first=True, 
        #                                padding_value=0.)
        t = torch.tensor(t).float()
        t = t.unsqueeze(0)
        #return t.permute(0,2,1).float().to(device)
        return t.float().to(device)
    
    #va_data.set_fen(inflow_fen, func, outflow_fen = outflow_fen)
    #te_data.set_fen(inflow_fen, func, outflow_fen = outflow_fen)
    
    #def make_embeds(flows, fen):
    #    with torch.no_grad():
    #        embeds = []
    #        for flow in flows:
    #            flow = func(flow)
    #            embeds.append(fen(flow).detach().cpu()[0])
    #    return embeds
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
    
    va_inflow_embeds, va_outflow_embeds = make_embeds(va_chain_flows, inflow_fen), make_embeds(va_chain_flows, outflow_fen)
    te_inflow_embeds, te_outflow_embeds = make_embeds(te_chain_flows, inflow_fen), make_embeds(te_chain_flows, outflow_fen)

    #def build_sims(dataset):
    def build_sims(inflow_embeds, outflow_embeds):
        """Iterate over the pair-wise dataset and 
            generate similarity vectors for the embeddings
        """
        sims = []
            
        #inflow_embeds = torch.stack(inflow_embeds)
        #outflow_embeds = torch.stack(outflow_embeds)
        
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
            all_sim = all_sim.permute(1,2,0)

            return all_sim

        # build sim vectors for all pairs
        sims = compute_sim(inflow_embeds, outflow_embeds)
        if args.temporal_alignment:
            sims2 = compute_sim(inflow_embeds.permute(0,2,1), 
                                outflow_embeds.permute(0,2,1))
            sims_both = torch.concat((sims, sims2), dim=-1)
        else:
            sims_both = sims
        
        return sims_both.numpy()
    
    
    va_sims = build_sims(va_inflow_embeds, va_outflow_embeds)
    te_sims = build_sims(te_inflow_embeds, te_outflow_embeds)
    
    
    # store data to file for later benchmarking
    with open(args.sims_file, 'wb') as fi:
        data = {'va_sims': va_sims.astype(np.float16), 
                  'te_sims': te_sims.astype(np.float16),
                  'va_chain_labels': va_chain_labels.astype(np.float32),
                  'te_chain_labels': te_chain_labels.astype(np.float32), 
                  'va_host_labels': va_host_labels.astype(np.float32),
                  'te_host_labels': te_host_labels.astype(np.float32)}
        pkl.dump(data, fi)
        print(len(np.unique(data['te_chain_labels'])))