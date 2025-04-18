import numpy as np
import torch
from torch.utils import data
from utils.processor import DataProcessor
from tqdm import tqdm
import pickle
import itertools
import json
from numpy import random
import hashlib
import os
from os.path import join, exists

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

def get_split_idx(available=10000, limit=10000):
    """Get indices for train, validation, and test splits using a fixed seed 
        for the same split and reproducibility between scripts.
    """
    idx = np.arange(0, min(available, limit))
    np.random.seed(42)
    np.random.shuffle(idx)
    te_idx = idx[0:500]
    va_idx = idx[500:1000]
    tr_idx = idx[1000:10000]
    return te_idx, va_idx, tr_idx

class TripletDataset(data.Dataset):
    def __init__(self, inflow_data, outflow_data, chain_targets=None, as_window=False):
        self.positive_top = True
        self.inflow_data = inflow_data
        self.outflow_data = outflow_data
        self.all_indices = list(range(len(self.inflow_data)))
        random.shuffle(self.all_indices)
        self.chain_targets = chain_targets
        self.as_windows = as_window

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
        
        sample_idx = torch.randint(low=0, high=len(self.outflow_data[idx]), size=(1,), dtype=torch.int32)
        positive = self.outflow_data[idx][sample_idx]
        
        sample_idx = torch.randint(low=0, high=len(self.outflow_data[negative_idx]), size=(1,), dtype=torch.int32)
        negative = self.outflow_data[negative_idx][sample_idx]
        
        if self.as_windows:
            window_idx = torch.randint(low=0, high=len(anchor), size=(1,), dtype=torch.int32)
            anchor = anchor[window_idx]
            positive = positive[window_idx]
            negative = negative[window_idx]
            
            #negative_window = None
            #max_iter = 20
            #i = 0
            #while negative_window is None or len(negative_window) == 0 or i == max_iter:
            #    window_idx = torch.randint(low=0, high=len(negative), size=(1,), dtype=torch.int32)
            #    negative_window = negative[window_idx]
            #    i += 1
            #negative = negative_window
        
        if self.chain_targets is not None:
            target = torch.tensor([sample_idx, len(self.outflow_data[negative_idx])-sample_idx], dtype=torch.float32) 
            return (
                torch.tensor(anchor, dtype=torch.float32),
                torch.tensor(positive, dtype=torch.float32),
                torch.tensor(negative, dtype=torch.float32),
                target,
            )

        
        #positive = self.outflow_data[idx]
        #target = self.chain_targets[idx]
        #negative = self.outflow_data[negative_idx]

        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            torch.tensor(negative, dtype=torch.float32),
        )
        
    def reset_split(self):
        self.positive_top = not self.positive_top

        # Reshuffle the indices at the start of each epoch
        random.shuffle(self.all_indices)

        # Re-divide the shuffled indices into two partitions
        cutoff = len(self.all_indices) // 2
        self.partition_1 = self.all_indices[:cutoff]
        self.partition_2 = self.all_indices[cutoff:]


class OnlineTripletDataset(data.Dataset):
    def __init__(self, inflow_data, outflow_data, chain_targets, as_window=False):
        self.positive_top = True
        self.inflow_data = inflow_data
        self.outflow_data = outflow_data
        self.all_indices = list(range(len(self.inflow_data)))
        random.shuffle(self.all_indices)
        self.size = len(self.all_indices)
        self.chain_targets = chain_targets
        self.as_window = as_window

    def __len__(self):
        return len(self.inflow_data)

    def __getitem__(self, trace_idx):
        # Pick a random inflow, outflow pair
        #trace_idx = torch.randint(low=0, high=self.size, size=(1,), dtype=torch.int32)
        anchor = self.inflow_data[trace_idx]
        
        sample_idx = torch.randint(low=0, high=len(self.outflow_data[trace_idx]), size=(1,), dtype=torch.int32)
        positive = self.outflow_data[trace_idx][sample_idx]
        
        if self.as_window:
            window_idx = torch.randint(low=0, high=len(anchor), size=(1,), dtype=torch.int32)
            anchor = anchor[window_idx]
            positive = positive[window_idx]
        
        target = torch.tensor([sample_idx, len(self.outflow_data[trace_idx])-sample_idx], dtype=torch.float32) 
        #self.chain_targets[trace_idx]

        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            target,
        )

    def reset_split(self):
        self.positive_top = not self.positive_top

        # Reshuffle the indices at the start of each epoch
        random.shuffle(self.all_indices)

        # Re-divide the shuffled indices into two partitions
        cutoff = len(self.all_indices) // 2
        self.partition_1 = self.all_indices[:cutoff]
        self.partition_2 = self.all_indices[cutoff:]


PROTO_MAP = {'ssh': 0, 'socat': 1, 'icmp': 2, 'dns': 3}


def check_filter(idx, idx_rev, upper, lower, ends_only):
    # filter out streams that does not match range criteria
    if ends_only:
        # select only lower and upper indices
        low_bound_1 = lower >= 0 and idx != lower
        low_bound_2 = lower < 0 and idx_rev != lower

        up_bound_1 = upper >= 0 and idx != upper
        up_bound_2 = upper < 0 and idx_rev != upper

        low_bound = low_bound_1 or low_bound_2
        up_bound = up_bound_1 or up_bound_2
        return low_bound and up_bound
    else:
        # select any idx between (inclusive) lower and upper indices
        low_bound_1 = lower >= 0 and idx < lower     # less than lower
        low_bound_2 = lower < 0 and idx_rev < lower  # rev is greater than lower

        up_bound_1 = upper >= 0 and idx > upper     # greater than upper
        up_bound_2 = upper < 0 and idx_rev > upper  # rev is less than upper

        low_bound = low_bound_1 or low_bound_2
        up_bound = up_bound_1 or up_bound_2
        return low_bound or up_bound


class BaseDataset(data.Dataset):
    """
    Dataset object to process and hold windowed samples in a convenient package.

    Attributes
    ----------
    data_ID_tuples : list
        List of stream-level identifiers as unique tuples, (chain_no, stream_no)
    data_windows : dict
        Dictionary containing lists of stream windows for streams within each chain
    data_chainlengths : dict
        Dictionary containing lists of chain-length 'labels' for use in chain-length prediction
    data_chain_IDs : dict
        Dictonary that maps chain numbers to all correlated stream-level IDs
    """
    def __init__(self, filepath, 
                       stream_processor,
                       sample_idx = None,
                       window_kwargs = dict(),
                       stream_ID_range = (0, float('inf')),
                       ends_only = False,
                       host_only = False,
                       preproc_feats = False,
                       save_to_dir = None,
                       zero_time = False,
                    ):
        """
        Load the metadata for samples collected in our SSID data. 

        Parameters
        ----------
        filepath : str
            The path to the data dictionary saved as a pickle file.
        stream_processor : DataProcessor
            The processor object that convert raw samples into their feature representations.
        window_kwargs : dict
            Dictionary containing the keyword arguments for the window processing function.
        stream_ID_range : 2-tuple
            Tuple of int or floats that can be used to control the stream hops loaded.
            First value is lower range. Second value is upper range (inclusive).
        ends_only : bool
            If True, select only the ends of stream_ID_range.
        host_only : bool
            If True, use host-only configuration (positive pairs were captured on the same host).
        preproc_feats : bool
            If True, the data processor will be applied on samples before windowing.
        """
        self.name = os.path.basename(filepath) 
        self.name += str(stream_processor.process_options) 
        self.name += str(stream_processor.interval_size)
        self.name += str(sample_idx) 
        self.name += json.dumps(window_kwargs,sort_keys=True) 
        self.name += str(stream_ID_range)
        self.name = hashlib.sha256(self.name.encode('utf-8'), usedforsecurity=True).hexdigest()
        
        self.ends_only = ends_only
        if host_only:
            stream_ID_range = (1, -2)  # include only streams from stepping-stones
        self.host_only = host_only
        
        loaded_dataset = False
        if save_to_dir is not None and exists(join(save_to_dir, self.name+'.pkl')):
            with open(join(save_to_dir, self.name+'.pkl'), 'rb') as fi:
                tmp = pickle.load(fi)
            self.data_ID_tuples = tmp['data_ID_tuples']
            self.data_chain_IDs = tmp['data_chain_IDs']
            self.data_chainlengths = tmp['data_chainlengths']
            self.data_windows = tmp['data_windows']
            #self.data_protocols = tmp['data_protocols']
            loaded_dataset = True
        
        if not loaded_dataset:
            self.data_ID_tuples = []
            self.data_windows = dict()
            self.data_chainlengths = dict()
            self.data_chain_IDs = dict()
            #self.data_protocols = dict()

            times_processor = DataProcessor(('times',))

            # load and enumeratechains in the dataset
            #chains, protocols = load_dataset(filepath, sample_idx)
            chains = load_dataset(filepath, sample_idx, zero_time=zero_time)
            for chain_ID, chain in tqdm(enumerate(chains)):

                # total hops in chain
                hops = (len(chain) // 2) + 1

                stream_ID_list = []

                # enumerate each stream in the chain
                # Note: streams are expected to be ordered from attacker (0) to stepping-stones (1<->(n-1)) to victim (n)
                for stream_ID, stream in enumerate(chain):

                    # filter out 
                    filtered = check_filter(idx = stream_ID, 
                                            idx_rev = stream_ID - len(chain), 
                                            lower = stream_ID_range[0], 
                                            upper = stream_ID_range[1],
                                            ends_only = ends_only)
                    if filtered: continue

                    # sample ID definitions
                    sample_ID = (chain_ID, stream_ID)
                    self.data_ID_tuples.append(sample_ID)
                    stream_ID_list.append(stream_ID)

                    if preproc_feats:
                        # multi-channel feature representation of stream
                        stream = stream_processor(stream)

                    # chunk stream into windows
                    if window_kwargs is not None:
                        # time-only representation
                        times = times_processor(stream)

                        windows = create_windows(times, stream, adjust_times=not preproc_feats, **window_kwargs)

                        if not preproc_feats:
                            # create multi-channel feature representation of windows independently
                            for i in range(len(windows)):
                                if len(windows[i]) <= 0:
                                    windows[i] = torch.empty((0, stream_processor.input_channels))
                                else:
                                    windows[i] = stream_processor(windows[i])

                        self.data_windows[sample_ID] = windows

                    else:
                        if not preproc_feats:  # apply processing if not yet performed
                            stream = stream_processor(stream)
                        self.data_windows[sample_ID] = [stream]

                    # chain-length label
                    downstream_hops = (stream_ID+1) // 2
                    upstream_hops = hops - downstream_hops - 1
                    self.data_chainlengths[sample_ID] = (downstream_hops, upstream_hops)
                    #self.data_protocols[sample_ID] = protocols[chain_ID][stream_ID]

                if len(stream_ID_list) > 1:
                    self.data_chain_IDs[chain_ID] = stream_ID_list
                    
            # convert all numpy arrays to torch tensors
            #self.data_windows = {k: [torch.tensor(x) for x in v] for k,v in self.data_windows.items()}
            #self.data_chainlengths = {k: torch.tensor(v) for k,v in self.data_chainlengths.items()}

            if save_to_dir is not None:
                if not exists(save_to_dir):
                    os.makedirs(save_to_dir)
                with open(join(save_to_dir, self.name+'.pkl'), 'wb') as fi:
                    pickle.dump({
                        'name': self.name,
                        'data_ID_tuples': self.data_ID_tuples,
                        'data_windows': self.data_windows,
                        'data_chainlengths': self.data_chainlengths,
                        'data_chain_IDs': self.data_chain_IDs,
                        #'data_protocols': self.data_protocols,
                    }, fi)
                
    def __len__(self):
        """
        Count of all streams within the dataset.
        """
        return len(self.data_ID_tuples)

    def __getitem__(self, index):
        """
        Generate a Triplet sample.

        Parameters
        ----------
        index : int
            The index of the sample to use as the anchor. 
            Note: Index values to sample mappings change after every reset_split()

        Returns
        -------
        windows : list
            List of windows for stream as torch tensors
        chain_label : 2-tuple
            A tuple containing ints that represent the downstream and upstream hop counts.
            Downstream represents the number of hosts between current host and victim.
            Upstream represents the number of hosts between current host and attacker.
        sample_ID : 2-tuple
            A tuple containing ints that acts as the sample ID.
            First value is the chain number. 
            Second value is the stream number within the chain.
        """
        sample_ID = self.data_ID_tuples[index]
        windows = self.data_windows[sample_ID]
        chain_label = self.data_chainlengths[sample_ID]
        return windows, chain_label, sample_ID


class PairwiseDataset(BaseDataset):
    """
    """
    def __init__(self, dataset, 
            sample_mode = 'oversample', 
            sample_ratio = None,
            sample_seed = 0):
        """
        """
        vars(self).update(vars(dataset))  # load dataset attributes

        self.return_embeds = False
        self.all_chain_IDs = np.array(list(self.data_chain_IDs.keys()))

        # create all possible combinations of chains within the dataset
        self.all_combos = list(itertools.combinations_with_replacement(self.all_chain_IDs, 2))

        # build complete list of sample ID pairs
        self.correlated_pairs = []
        self.uncorrelated_pairs = []
        for chain1_ID, chain2_ID in self.all_combos:
            # build all possible pairs of streams between the two chains

            if chain1_ID == chain2_ID:
                correlated = True
            else:
                correlated = False

            # for positives, consider only samples on the same host
            if self.host_only and correlated:
                # consider only host-wise pairs if host_only is enabled
                stream_IDs = self.data_chain_IDs[chain1_ID]
                for i in range(0, len(stream_IDs), 2):
                    hostwise_pair = ((chain1_ID, stream_IDs[i]), 
                                     (chain1_ID, stream_IDs[i+1]), 
                                     correlated)
                    self.correlated_pairs.append(hostwise_pair)

            # consider only first and last sample pairs
            elif self.ends_only:
                a = (chain1_ID, self.data_chain_IDs[chain1_ID][0])
                b = (chain1_ID, self.data_chain_IDs[chain1_ID][-1])
                c = (chain2_ID, self.data_chain_IDs[chain2_ID][0])
                d = (chain2_ID, self.data_chain_IDs[chain2_ID][-1])
                if correlated:
                    self.correlated_pairs.append((a, b, correlated))
                else:
                    self.uncorrelated_pairs.append((a, d, correlated))
                    self.uncorrelated_pairs.append((c, b, correlated))

            # all possible stream combinations between chains (with sole exception to same streams)
            else:
                chain1_ID_tuples = [(chain1_ID, stream_ID) for stream_ID in self.data_chain_IDs[chain1_ID]]
                chain2_ID_tuples = [(chain2_ID, stream_ID) for stream_ID in self.data_chain_IDs[chain2_ID]]

                # create all sample pairs
                for ID1 in chain1_ID_tuples:
                    for ID2 in chain2_ID_tuples:
                        # if correlated, don't include same-stream pairs
                        if correlated and ID1[1] != ID2[1]:
                            self.correlated_pairs.append((ID1, ID2, correlated))
                        # no additional filtering needed on uncorrelated pairs
                        elif not correlated:
                            self.uncorrelated_pairs.append((ID1, ID2, correlated))

        if sample_ratio is not None:
            random.seed(sample_seed)

            if sample_mode == 'oversample':
                k = int(len(self.uncorrelated_pairs) * sample_ratio)
                idx = random.choice(np.arange(len(self.correlated_pairs)), 
                                    size=k, replace=False)
                self.correlated_pairs = np.array(self.correlated_pairs, dtype=object)[idx]
                self.uncorrelated_pairs = np.array(self.uncorrelated_pairs, dtype=object)

            elif sample_mode == 'undersample':
                k = int(len(self.correlated_pairs) * sample_ratio)
                idx = random.choice(np.arange(len(self.uncorrelated_pairs)), 
                                    size=k, replace=False)
                self.uncorrelated_pairs = np.array(self.uncorrelated_pairs, dtype=object)[idx]
                self.correlated_pairs = np.array(self.correlated_pairs, dtype=object)

            self.all_pairs = np.concatenate((self.correlated_pairs, self.uncorrelated_pairs))

            self.all_sample_IDs = set()
            for sample1,sample2,_ in self.all_pairs:
                self.all_sample_IDs.add(sample1)
                self.all_sample_IDs.add(sample2)
                
        else:
            self.all_pairs = np.array(self.correlated_pairs + self.uncorrelated_pairs, dtype=object)
            self.all_sample_IDs = self.data_ID_tuples


    def set_fen(self, inflow_fen, proc, outflow_fen=None):
        """Attach a FEN model to the dataset for feature extraction. 
        Pre-compute the embeddings when attaching the FEN.
        """
        self.return_embeds = True
        self.computed_embeds = dict()
        
        inflow_fen.eval()
        if outflow_fen is not None:
            outflow_fen.eval()
            
        for sample_ID in tqdm(self.all_sample_IDs):
            windows = self.data_windows[sample_ID]
            embeds = inflow_fen(proc(windows)).detach().cpu()
            self.computed_embeds[str(sample_ID) + 'inflow'] = embeds

            if outflow_fen is not None:
                embeds = outflow_fen(proc(windows)).detach().cpu()
                self.computed_embeds[str(sample_ID) + 'outflow'] = embeds
            else:
                self.computed_embeds[str(sample_ID) + 'outflow'] = embeds

    def unset_fen(self):
        """Remove attached FEN
        """
        self.return_embeds = False
        del self.computed_embeds

    def __len__(self):
        """
        """
        return len(self.all_pairs)

    def __getitem__(self, index):
        """
        """
        sample1_ID, sample2_ID, correlated = self.all_pairs[index]

        if not self.return_embeds:
            sample1_windows = self.data_windows[sample1_ID]
            sample2_windows = self.data_windows[sample2_ID]
        else:
            sample1_windows = self.computed_embeds[str(sample1_ID)+'inflow']
            sample2_windows = self.computed_embeds[str(sample2_ID)+'outflow']

        sample1_chainlength = self.data_chainlengths[sample1_ID]
        sample2_chainlength = self.data_chainlengths[sample2_ID]

        sample1 = (sample1_windows, sample1_chainlength, sample1_ID)
        sample2 = (sample2_windows, sample2_chainlength, sample2_ID)

        return sample1, sample2, correlated


class OfflineDataset(BaseDataset):
    """
    Dataset object for generating triplets for triplet learning.
    """
    def __init__(self, dataset):
        """
        Initialize triplet dataset from an existing SSI dataset object.
        """
        vars(self).update(vars(dataset))  # load dataset attributes

        # create and shuffle indices for samples
        self.all_indices = np.array(list(self.data_chain_IDs.keys()))
        np.random.shuffle(self.all_indices)

        # divide indices into two partitions for triplet generation
        cutoff = len(self.all_indices) // 2
        self.partition_1 = self.all_indices[:cutoff]
        self.partition_2 = self.all_indices[cutoff:]

    def __len__(self):
        """
        An epoch of the TripletDataset iterates over all samples within partition_1
        """
        return len(self.partition_1)

    def __getitem__(self, index):
        """
        Generate a Triplet sample.

        Anchor samples are selected from parition_1
        Positive samples are selected at random from the chain of the anchor sample
        Negative samples are selected from partition_2

        Parameters
        ----------
        index : int
            The index of the sample to use as the anchor. 
            Note: Index values to sample mappings change after every reset_split()

        Returns
        -------
        anc : tensor
            Window to represent the anchor sample
        pos : tensor
            Correlated window to represent positive sample
        neg : tensor
            Uncorrelated window to represent the negative sample
        """
        # chain ID for anchor & positive
        anc_chain_ID = self.partition_1[index]

        if self.ends_only:
            anc_stream_IDs = [self.data_chain_IDs[anc_chain_ID][0], 
                              self.data_chain_IDs[anc_chain_ID][-1]]
        elif self.host_only:
            # randomly select a pair of streams, both of which were collected on the same host (e.g. host-based SSID)
            anc_stream_IDs = self.data_chain_IDs[anc_chain_ID]
            host_num = np.random.randint(0, len(anc_stream_IDs)//2)
            anc_stream_IDs = [anc_stream_IDs[host_num], anc_stream_IDs[host_num+1]]
            np.random.shuffle(anc_stream_IDs)
        else:
            # randomly sample (w/o replacement) two streams from the chain for anchor & positive (e.g. network-based SSID)
            anc_stream_IDs = self.data_chain_IDs[anc_chain_ID]
            anc_stream_IDs = np.random.choice(anc_stream_IDs, size=2, replace=False)

        anc_ID = (anc_chain_ID, anc_stream_IDs[0])
        pos_ID = (anc_chain_ID, anc_stream_IDs[1])

        # randomly select chain from partition 2 to be the negative
        neg_chain_ID = np.random.choice(self.partition_2)
        # randomly sample a stream from the negative stream
        neg_stream_ID = np.random.choice(self.data_chain_IDs[neg_chain_ID])
        neg_ID = (neg_chain_ID, neg_stream_ID)

        # get windows for anc, pos, neg
        anc = self.data_windows[anc_ID]
        pos = self.data_windows[pos_ID]
        neg = self.data_windows[neg_ID]

        # randomly select a window for the triplet
        if True:
            # ignore windows without packets
            candidate_idx = [i for i,window in enumerate(anc) if len(window) > 0]
            window_idx = np.random.choice(candidate_idx)
        if False:
            window_idx = np.random.randint(0, len(anc)-1)

        anc_tup = (anc[window_idx], self.data_chainlengths[anc_ID], anc_ID)
        pos_tup = (pos[window_idx], self.data_chainlengths[pos_ID], pos_ID)
        neg_tup = (neg[window_idx], self.data_chainlengths[neg_ID], neg_ID)
        return anc_tup, pos_tup, neg_tup
                    

    def reset_split(self):
        """
        Reshuffle sample indices and select a new split for triplet generation.
        """
        # Reshuffle the indices at the start of each epoch.
        np.random.shuffle(self.all_indices)

        # Re-divide the shuffled indices into two partitions.
        cutoff = len(self.all_indices) // 2
        self.partition_1 = self.all_indices[:cutoff]
        self.partition_2 = self.all_indices[cutoff:]

    @staticmethod
    def batchify(batch):
        """
        convert samples to tensors and pad samples to equal length
        """
        # convert labels to tensor and get sequence lengths
        #batch_anc, batch_neg, batch_pos = zip(*batch)
        batch_anc = []
        batch_neg = []
        batch_pos = []
        for i in range(len(batch)):
            anc, pos, neg = batch[i]
            batch_anc.append(anc)
            batch_neg.append(neg)
            batch_pos.append(pos)
    
        # batch windows for anchor, positive, and negative samples
        batch_x_anc = [tup[0] for tup in batch_anc]
        batch_x_pos = [tup[0] for tup in batch_pos]
        batch_x_neg = [tup[0] for tup in batch_neg]
        batch_x = [batch_x_anc, batch_x_pos, batch_x_neg]
    
        batch_y_anc = [tup[1] for tup in batch_anc]
        batch_y_anc = torch.tensor(batch_y_anc)
    
        # pad and fix dimension
        batch_x_tensors = []
        for batch_x_n in batch_x:
            batch_x_n = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in batch_x_n], 
                                                        batch_first=True, padding_value=0.)
            if len(batch_x_n.shape) < 3:  # add channel dimension if missing
                batch_x_n = batch_x_n.unsqueeze(-1)
            batch_x_n = batch_x_n.permute(0,2,1)
            batch_x_n = batch_x_n.float()
            batch_x_tensors.append(batch_x_n)
    
        return *batch_x_tensors, batch_y_anc.long()


class OnlineDataset(BaseDataset):
    """
    """
    def __init__(self, dataset, k=2):
        """
        Initialize triplet dataset from an existing SSI dataset object.
        """
        vars(self).update(vars(dataset))  # load dataset attributes

        # create and shuffle indices for samples
        self.all_indices = np.array(list(self.data_chain_IDs.keys()))
        np.random.shuffle(self.all_indices)

        self.k = k

    def __len__(self):
        """
        An epoch of the TripletDataset iterates over all samples within partition_1
        """
        return len(self.all_indices)

    def __getitem__(self, index):
        """
        """
        chain_ID = self.all_indices[index]

        if self.ends_only:
            stream_IDs = [self.data_chain_IDs[chain_ID][0], 
                          self.data_chain_IDs[chain_ID][-1]]
        elif not self.host_only:
            # randomly select a pair of streams, both of which were collected on the same host (e.g. host-based SSID)
            stream_IDs = self.data_chain_IDs[chain_ID]
            host_num = np.random.randint(0, len(stream_IDs)//2)
            stream_IDs = [stream_IDs[host_num], stream_IDs[host_num+1]]
            np.random.shuffle(stream_IDs)
        else:
            # randomly sample (w/o replacement) two streams from the chain for anchor & positive (e.g. network-based SSID)
            stream_IDs = self.data_chain_IDs[chain_ID]
            stream_IDs = np.random.choice(stream_IDs, size=self.k, replace=False)

        samples = []
        chain_lengths = []
        #protocols = []
        for stream_ID in stream_IDs:
            ID = (chain_ID, stream_ID)
            sample = self.data_windows[ID]
            samples.append(sample)
            chain_lengths.append(self.data_chainlengths[ID])
            #protocols.append(self.data_protocols[ID])

        return samples, chain_lengths#, protocols

    @staticmethod
    def batchify(batch):
        """
        convert samples to tensors and pad samples to equal length
        """
        batch_x = []
        batch_y = []
        chain_lengths = []
        #chain_protocols = []
        cur_label = 0
        for i in range(len(batch)):
            # add correlated samples to batch
            batch_x.extend(batch[i][0])
            # add label information for correlated samples
            batch_y.extend([cur_label] * len(batch[i][0]))
            chain_lengths.extend(batch[i][1])
            #chain_protocols.extend(batch[i][2])
            cur_label += 1

        # pick a random window to return
        window_count = len(batch_x[0])
        window_idx = np.random.choice(range(window_count))
        for i in range(len(batch_x)):
            window = batch_x[i][window_idx]
            while len(window) < 0:
                new_idx = np.random.choice(range(window_count))
                window = batch_x[i][new_idx]
            batch_x[i] = window
        #batch_x = [x[window_idx] for x in batch_x]

        # pad batches and fix dimension
        batch_x_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in batch_x], 
                                                    batch_first = True, 
                                                    padding_value = 0.)
        batch_x_tensor = batch_x_tensor.permute(0,2,1)
        batch_x_tensor = batch_x_tensor.float()
        batch_y_tensor = torch.tensor(batch_y)
        chain_lengths = torch.tensor(chain_lengths)
        #chain_protocols = torch.tensor(chain_protocols)
    
        return batch_x_tensor, batch_y_tensor, chain_lengths#, chain_protocols


def create_windows(times, features,
                    window_width = 5,
                    window_count = 11,
                    window_overlap = 3,
                    include_all_window = False,
                    adjust_times = False,
                ):
    """
    Slice a sample's full stream into time-based windows.

    Parameters
    ----------
    times : ndarray
    features : ndarray
    window_width : int
    window_count : int
    window_overlap : int

    Returns
    -------
    list
        A list of stream windows as torch tensors.
    """
    window_features = []

    if include_all_window:
        window_count -= 1

    if window_count > 0:
        window_step = max(window_width - window_overlap, 1)

        # Create overlapping windows
        for start in np.arange(0, stop = window_count * window_step, 
                                  step = window_step):

            end = start + window_width

            window_idx = np.where(np.logical_and(times >= start, times < end))[0]
            window = features[window_idx]
            ## Find the indices for the current window range [start, end)
            #left_idx = np.searchsorted(times.squeeze(), start, side='left')
            #right_idx = np.searchsorted(times.squeeze(), end, side='left')
            ## Slice out the timestamps for this window
            #window = features[left_idx:right_idx]
            
            if adjust_times:
                window[:,0] -= start
            window_features.append(window)

    # add full stream as window
    if include_all_window:
        window_features.append(features)

    return window_features

def process(x, zero_time=False):
    """
    Simple example function to use when processing 
    """
    timestamps   = x[0]
    packet_sizes = x[1]
    directions   = x[2]

    sorted_indices = np.argsort(timestamps)

    timestamps = timestamps[sorted_indices]
    packet_sizes =  packet_sizes[sorted_indices]
    directions = directions[sorted_indices]

    #iats = np.diff(timestamps)
    #iats = np.concatenate(([0], iats))

    if zero_time:
        t_min = np.amin(timestamps)
    else:
        t_min = 0
    output = [(t-t_min, d*s) for t,d,s in zip(timestamps, directions, packet_sizes)]

    output = np.array(sorted(output, key=lambda x: x[0]))

    return output


def load_dataset(filepath, idx_selector=None, zero_time=False):
    """
    Load the metadata for samples collected in our SSID data. 

    Parameters
    ----------
    filepath : str
        The path to the data dictionary saved as a pickle file.
    idx_selector : ndarray
        A 1D numpy array that identifies which samples from the file to load.
        Samples are sorted before selection is applied to insure consistency of loading.

    Returns
    -------
    list
        A nested list of processed streams
        The outer list contains lists of correlated processed streams, while the inner lists contain individual instances
        (with all instances within the list being streams produced by hosts within the same multi-hop tunnel)
    """

    with open(filepath, "rb") as fi:
        all_data = pickle.load(fi)

    if all_data.get('obfs', False):
        return all_data['chains']   # data is already organized into the expected nested list format

    IP_info = all_data['IPs']   # extra src. & dst. IP info available for each stream
    data = all_data['data']     # stream metadata organized by sample and hosts (per sample)
    #data_protocols = all_data['proto']

    # list of all sample idx
    sample_IDs = sorted(list(data.keys()))   # sorted so that it is reliably ordered
    if idx_selector is not None:
        sample_IDs = np.array(sample_IDs, dtype=object)[idx_selector].tolist()  # slice out requested idx

    # fill with lists of correlated samples
    all_streams = []
    all_protocols = []

    # each 'sample' contains a variable number of hosts (between 3 and 6 I believe)
    for s_idx in sample_IDs:
        sample = data[s_idx]
        host_IDs = sorted(list(sample.keys()))
        #protocols = data_protocols[s_idx]
        if len(host_IDs) > 5:
            continue

        # first and last hosts represent the attacker's machine and target endpoint of the chain respectively
        # these hosts should contain only one SSH stream in their sample
        attacker_ID = 1
        target_ID   = len(host_IDs)

        # the stepping stone hosts are everything in-between
        # these hosts should each contain two streams
        steppingstone_IDs = list(filter(lambda x: x not in [attacker_ID, target_ID], host_IDs))

        # loop through each host, process stream metadata into vectors, and add to list
        correlated_streams = []
        stream_protocols = []
        for h_idx in host_IDs:
            #correlated_streams.extend([torch.tensor(x).T for x in sample[h_idx]])
            correlated_streams.extend([process(x, zero_time=zero_time) for x in sample[h_idx]])
            #stream_protocols.extend([PROTO_MAP[proto] for proto in protocols[h_idx]])

        # add group of correlated streams for the sample into the data list
        all_streams.append(correlated_streams)
        #all_protocols.append(stream_protocols)

    return all_streams#, all_protocols


    
# Helper function to process a single chain
def process_chain(i, chain, in_idx, inflow_size, outflow_size, processor):
    target = len(chain) - 2
    # check if inner element is list
    if isinstance(chain[in_idx], list):
        # chain contains samples as sequence of traffic windows
        s1 = [processor(chain[in_idx][j]) for j in range(len(chain[in_idx]))]
        s2 = [[processor(chain[j][k]) for k in range(len(chain[j]))] 
                for j in range(in_idx, len(chain))]
        
        # apply padding to feature windows
        for j in range(len(s1)):
            if len(s1[j]) < inflow_size:
                s1[j] = np.pad(s1[j], ((0, inflow_size - len(s1[j])), (0, 0)))
            else:
                s1[j] = s1[j][:inflow_size]
        for j in range(len(s2)):
            for k in range(len(s2[j])):
                if len(s2[j][k]) < outflow_size:
                    s2[j][k] = np.pad(s2[j][k], ((0, outflow_size - len(s2[j][k])), (0, 0)))
                else:
                    s2[j][k] = s2[j][k][:outflow_size]
    else:
        # non-windowized samples
        s1 = processor(chain[in_idx])
        s2 = [processor(chain[j]) for j in range(in_idx, len(chain))]

        # Pad or trim s1 to match inflow_size
        if len(s1) < inflow_size:
            s1 = np.pad(s1, ((0, inflow_size - len(s1)), (0, 0)))
        else:
            s1 = s1[:inflow_size]

        # Pad or trim each element in s2 to match outflow_size
        for j in range(len(s2)):
            if len(s2[j]) < outflow_size:
                s2[j] = np.pad(s2[j], ((0, outflow_size - len(s2[j])), (0, 0)))
            else:
                s2[j] = s2[j][:outflow_size]

    return s1, s2, target

def build_dataset(pklpath, processor, inflow_size, outflow_size, 
                  in_idx=1, n_jobs=8, window_kwargs=None):
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
        tasks.append((i, chain, in_idx, inflow_size, outflow_size, processor))

    # Process chains in parallel using a multiprocessing Pool
    inflow, outflow, targets = [], [], []
    with mpp.Pool(min(cpu_count(), n_jobs)) as pool:
        for s1, s2, t in tqdm(pool.istarmap(process_chain, tasks, chunksize=16), 
                              total = len(tasks)):
            inflow.append(s1)
            outflow.append(s2)
            targets.append(t)

    if window_kwargs is not None:
        # Stack and transpose as needed
        inflow = np.stack(inflow).transpose((0, 1, 3, 2))
        outflow = np.array([np.stack(samples).transpose((0, 1, 3, 2)) for samples in outflow], dtype=object)
    else:
        inflow = np.stack(inflow).transpose((0, 2, 1))
        outflow = np.array([np.stack(samples).transpose((0, 2, 1)) for samples in outflow], dtype=object)
        
    targets = np.array(targets)

    return inflow, outflow, targets


if __name__ == "__main__":

    pklpath = '../processed.pkl'

    # chain-based sample splitting
    te_idx = np.arange(0,1300)
    tr_idx = np.arange(1300,13000)

    # stream window definitions
    window_kwargs = {
                     'window_width': 5, 
                     'window_count': 11, 
                     'window_overlap': 2
                     }

    # multi-channel feature processor
    processor = DataProcessor(('sizes', 'iats', 'time_dirs', 'dirs'))

    # load SSI dataset object
    for idx in (te_idx, tr_idx):
        print(f'Chains: {len(idx)}')

        # build base dataset object
        data = BaseDataset(pklpath, processor,
                            window_kwargs = window_kwargs,
                            sample_idx = idx,
                            stream_ID_range = (0,1))
        print(f'Streams: {len(data)}')
        for windows, chain_label, sample_ID in data:
            pass

        # construct a triplets dataset object, derived from the base dataset object
        # Note: changes to the base dataset propogate to the triplet object once initialized (I think?)
        triplets = TripletDataset(data)
        print(f'Triplets: {len(triplets)}')
        for anc, pos, neg in triplets:
            pass
        triplets.reset_split()
