import torch
from torch.nn import functional as F
import numpy as np



def rate_estimator(iats, sizes):
    """Simple/naive implementation of a running average traffic flow rate estimator
       It is entirely vectorized, so it is fast
    """
    times = np.cumsum(iats, axis=0)
    #indices = torch.arange(1, iats.size(0) + 1)
    sizes = np.cumsum(sizes, axis=0)
    flow_rate = np.where(times != 0, sizes / times, np.ones_like(times))
    return flow_rate


def weighted_rate_estimator(iats, k=0.1):
    """Implementation of a traffic flow rate estimation function with an expoential decay
       follows guidance from: https://stackoverflow.com/questions/23615974/estimating-rate-of-occurrence-of-an-event-with-exponential-smoothing-and-irregul?noredirect=1&lq=1
    """
    times = np.cumsum(iats, axis=0)
    exps1 = np.exp(k * -iats)
    exps2 = np.exp(k * -times)
    rates = [0]
    for i in range(len(iats)-1):
        rate = k + (exps1[i+1] * rates[i])
        rate = rate / (1 - exps2[i+1])
        rate = np.clip(np.nan_to_num(rate, nan=1e4), 1e4)
        rates.append(rate)
    return np.array(rates)


def pad_or_truncate(arr, size):
    """Pad or truncate a numpy array to a specified size."""
    if len(arr) < size:
        return np.pad(arr, (0, size - len(arr)), 'constant')
    else:
        return arr[:size]
    

class DataProcessor2:
    
    def __init__(self, *args, interval_size=0.03, **kwargs):
        self.interval_size = interval_size
        self.process_options = []
        self.input_channels = self.process(np.ones((100,2))).shape[0]

    def process(self, x, window_size=1000):
        packets = x
        
        times, sizes = zip(*packets)
        times = np.array(times)
        sizes = np.array(sizes)
        directions = np.sign(sizes)
        
        upload = directions > 0
        download = ~upload
        iats = np.diff(times, prepend=0)

        num_intervals = int(np.ceil(times.max() / self.interval_size))

        split_points = np.arange(0, num_intervals) * self.interval_size
        split_indices = np.searchsorted(times, split_points)

        interval_dirs_up = np.zeros(num_intervals + 1)
        interval_dirs_down = np.zeros(num_intervals + 1)
        for j, tensor in enumerate(np.split(directions, split_indices)):
            size = len(tensor)
            if size > 0:
                up = (tensor >= 0).sum()
                interval_dirs_up[j] = up
                interval_dirs_down[j] = size - up
    
        interval_size_up = np.zeros(num_intervals + 1)
        interval_size_down = np.zeros(num_intervals + 1)
        for j, tensor in enumerate(np.split(sizes*directions, split_indices)):
            size = np.abs(tensor).sum()
            if size > 0:
                up = (tensor * (tensor > 0)).sum().item()
                interval_size_up[j] = up / 1500
                interval_size_down[j] = (size - up) / 1500

        interval_times = np.zeros(num_intervals + 1)
        for j, tensor in enumerate(np.split(times, split_indices)):
            if len(tensor) > 0:
                interval_times[j] = tensor.mean()
            elif j > 0:
                interval_times[j] = interval_times[j - 1]

        interval_times_norm = interval_times - interval_times.mean()
        if np.abs(interval_times_norm).max() != 0:
            interval_times_norm /= np.abs(interval_times_norm).max()

        interval_iats = np.zeros(num_intervals + 1)
        for j, tensor in enumerate(np.split(iats, split_indices)):
            if len(tensor) > 0:
                interval_iats[j] = tensor.mean()
            elif j > 0:
                interval_iats[j] = interval_iats[j - 1] + self.interval_size

        download_iats = np.diff(times[download], prepend=0)
        upload_iats = np.diff(times[upload], prepend=0)
        flow_iats = np.zeros_like(times)
        flow_iats[upload] = upload_iats
        flow_iats[download] = download_iats
        inv_iat_logs = np.log(np.nan_to_num(1 / flow_iats + 1, nan=1e4, posinf=1e4))
        interval_inv_iat_logs = np.zeros(num_intervals + 1)
        for j, tensor in enumerate(np.split(inv_iat_logs, split_indices)):
            if len(tensor) > 0:
                interval_inv_iat_logs[j] = tensor.mean()

        size_dirs = sizes * directions
        cumul = np.cumsum(size_dirs)
        interval_cumul = np.zeros(num_intervals + 1)
        for j, tensor in enumerate(np.split(cumul, split_indices)):
            if len(tensor) > 0:
                interval_cumul[j] = tensor.mean()
            elif j > 0:
                interval_cumul[j] = interval_cumul[j - 1]

        interval_cumul_norm = interval_cumul - interval_cumul.mean()
        if np.abs(interval_cumul_norm).max() != 0:
            interval_cumul_norm /= np.abs(interval_cumul_norm).max()

        interval_dirs_sum = interval_dirs_up + interval_dirs_down
        interval_dirs_sub = interval_dirs_up - interval_dirs_down
    
        interval_size_sum = interval_size_up + interval_size_down
        interval_size_sub = interval_size_up - interval_size_down

        features = np.stack([
            #pad_or_truncate(sizes, size=window_size),
            #pad_or_truncate(times, size=window_size),
            #pad_or_truncate(directions, size=window_size),
            pad_or_truncate(interval_dirs_up, size=window_size),
            pad_or_truncate(interval_dirs_down, size=window_size),
            pad_or_truncate(interval_dirs_sum, size=window_size),
            pad_or_truncate(interval_dirs_sub, size=window_size),
            #pad_or_truncate(interval_size_up, size=window_size),
            #pad_or_truncate(interval_size_down, size=window_size),
            #pad_or_truncate(interval_size_sum, size=window_size),
            #pad_or_truncate(interval_size_sub, size=window_size),
            pad_or_truncate(interval_iats, size=window_size),
            pad_or_truncate(interval_inv_iat_logs, size=window_size),
            pad_or_truncate(interval_cumul_norm, size=window_size),
            pad_or_truncate(interval_times_norm, size=window_size)
        ])
    
        return features
    
    def __call__(self, x):
        return self.process(x)

class DataProcessor2:
    
    def __init__(self, *args, interval_size=0.03, **kwargs):
        self.input_channels = 12
        self.interval_size = interval_size
        self.process_options = []
        
    def process(self, x):
        """Convert a single file to a PyTorch tensor of features."""
        # Extract times and sizes from the file
        
        sorted_indices = np.argsort(x.T[0])
        
        times = 0.5*x.T[0][sorted_indices]
        sizes = x.T[1][sorted_indices]
        directions = x.T[2][sorted_indices]

        upload = directions > 0
        download = ~upload
        iats = torch.diff(times, prepend=torch.tensor([0.0]))

        num_intervals = int(torch.ceil(times.max() / self.interval_size).item())

        split_points = torch.arange(0, num_intervals) * self.interval_size
        split_indices = torch.searchsorted(times.contiguous(), split_points.contiguous()).cpu()

        interval_dirs_up = torch.zeros(num_intervals + 1)
        interval_dirs_down = torch.zeros(num_intervals + 1)
        for j, tensor in enumerate(torch.tensor_split(directions, split_indices)):
            size = tensor.size(0)
            if size > 0:
                up = (tensor >= 0).sum().item()
                interval_dirs_up[j] = up
                interval_dirs_down[j] = size - up
                
        interval_size_up = torch.zeros(num_intervals + 1)
        interval_size_down = torch.zeros(num_intervals + 1)
        for j, tensor in enumerate(torch.tensor_split(sizes*directions, split_indices)):
            size = tensor.abs().sum()
            if size > 0:
                up = torch.relu(tensor).sum().item()
                interval_size_up[j] = up / 1500
                interval_size_down[j] = (size - up) / 1500

        interval_times = torch.zeros(num_intervals + 1)
        for j, tensor in enumerate(torch.tensor_split(times, split_indices)):
            if tensor.size(0) > 0:
                interval_times[j] = tensor.mean().item()
            elif j > 0:
                interval_times[j] = interval_times[j - 1]

        interval_times_norm = interval_times - interval_times.mean()
        if torch.abs(interval_times_norm).max().item() != 0:
            interval_times_norm /= torch.abs(interval_times_norm).max()

        interval_iats = torch.zeros(num_intervals + 1)
        for j, tensor in enumerate(torch.tensor_split(iats, split_indices)):
            if tensor.size(0) > 0:
                interval_iats[j] = tensor.mean().item()
            elif j > 0:
                interval_iats[j] = interval_iats[j - 1] + self.interval_size

        download_iats = torch.diff(times[download], prepend=torch.tensor([0.0]))
        upload_iats = torch.diff(times[upload], prepend=torch.tensor([0.0]))
        flow_iats = torch.zeros_like(times)
        flow_iats[upload] = upload_iats
        flow_iats[download] = download_iats
        inv_iat_logs = torch.log(torch.nan_to_num((1 / flow_iats) + 1, nan=1e4, posinf=1e4))
        interval_inv_iat_logs = torch.zeros(num_intervals + 1)
        for j, tensor in enumerate(torch.tensor_split(inv_iat_logs, split_indices)):
            if tensor.size(0) > 0:
                interval_inv_iat_logs[j] = tensor.mean().item()

        size_dirs = sizes * directions
        cumul = torch.cumsum(size_dirs, dim=0)
        interval_cumul = torch.zeros(num_intervals + 1)
        for j, tensor in enumerate(torch.tensor_split(cumul, split_indices)):
            if tensor.size(0) > 0:
                interval_cumul[j] = tensor.mean().item()
            elif j > 0:
                interval_cumul[j] = interval_cumul[j - 1]

        interval_cumul_norm = interval_cumul - interval_cumul.mean()
        if torch.abs(interval_cumul_norm).max().item() != 0:
            interval_cumul_norm /= torch.abs(interval_cumul_norm).max()

        interval_dirs_sum = interval_dirs_up + interval_dirs_down
        interval_dirs_sub = interval_dirs_up - interval_dirs_down
        interval_size_sum = interval_size_up + interval_size_down
        interval_size_sub = interval_size_up - interval_size_down

        features = torch.stack([
            interval_dirs_up,
            interval_dirs_down,
            interval_dirs_sum,
            interval_dirs_sub,
            interval_size_up,
            interval_size_down,
            interval_size_sum,
            interval_size_sub,
            interval_iats,
            interval_inv_iat_logs,
            interval_cumul_norm,
            interval_times_norm,
        ], dim=-1)

        return features
    
    def __call__(self, x):
        return self.process(x)
        

class DataProcessor:
    """ Initialize with desired list of features
        Apply to map a raw traffic sample into it's feature representation
    """

    # list of valid processor options and any opts that depend on them
    DEPENS = {
                'times': [],        # times & dirs are always available so no need to list depens
                'sizes': [],
                'dirs': [], 
                'time_dirs': [],    # tik-tok style representation
                'size_dirs': ['cumul', 
                              'dcf'],

                'burst_edges': [],  # a sparse representation that identifies whenever traffic changes direction

                'cumul': ['cumul_norm', 
                          'interval_cumul'],  # per-packet cumulative sum (based on cumul)
                'cumul_norm': [],         # normalize cumul features into [-1,+1] range, centered around the mean
                'times_norm': [],         # normalize timestamp features into [-1,+1] range, centered around the mean

                'iats': ['iat_dirs', 
                         'interval_iats', 
                         'running_rates', 
                         'dcf'],     # difference between consequtive timestamps (e.g., inter-arrival-times
                'iat_dirs': [],           # iats with direction encoded into the representation

                'running_rates': ['running_rates_diff'],   # running average of flow rate (ignores direction)
                'running_rates_diff': [],                  # instantaneous change of flow rate
                'running_rates_decayed': ['up_rates_decayed', 
                                          'down_rates_decayed'],       # running average with exponential decay on old rates (expensive to compute)
                'up_rates_decayed': [],       # up-direction only (non-aligned)
                'down_rates_decayed': [],     # down-direction only (non-aligned)
                
                # iats computed on upload-only pkt sequence (is not packet aligned with time/dir seq)
                'up_iats': ['up_iats_sparse', 
                            'up_rates', 
                            'flow_iats'],         
                # iats computed on download-only pkt sequence (is not packet aligned with time/dir seq)
                'down_iats': ['down_iats_sparse', 
                              'down_rates', 
                              'flow_iats'],   
                'up_iats_sparse': [],                  # sparsified sequence (e.g., download pkts have value of zero)
                'down_iats_sparse': [],                # sparsified sequence (e.g., upload pkts have value of zero)
                'up_rates': ['up_rates_sparse'],       # simple rate estimator applied to up_iats
                'down_rates': ['down_rates_sparse'],   # simple rate estimator applied to down_iats
                'up_rates_sparse': [],        # sparsified, but pck-aligned sequence
                'down_rates_sparse': [],      # sparsified, but pck-aligned sequence
                # up & down iats merged into one sequence (aligned with time/dir seqs)
                'flow_iats': ['burst_filtered_times', 
                              'inv_iat_logs'], 
                'burst_filtered_times': ['burst_filtered_time_dirs'],  # filtered sequence with large gaps removed (non-aligned)
                'burst_filtered_time_dirs': [],                        # with direction encoded (non-aligned)
                # log applied to the inverse of flow iats (adjust with +1 to avoid negative logs)
                'inv_iat_logs': ['inv_iat_log_dirs', 
                                 'interval_inv_iat_logs'],  
                'inv_iat_log_dirs': [],                  # with pkt direction encoded

                'interval_dirs_up': ['interval_dirs_sum', 
                                     'interval_dirs_sub'], 
                'interval_dirs_down': ['interval_dirs_sum', 
                                       'interval_dirs_sub'], 
                'interval_dirs_sum': [], 
                'interval_dirs_sub': [], 
                'interval_size_up': ['interval_size_sum', 
                                     'interval_size_sub'], 
                'interval_size_down': ['interval_size_sum', 
                                       'interval_size_sub'], 
                'interval_size_sum': [], 
                'interval_size_sub': [], 
                'interval_times': ['interval_times_norm'], 
                'interval_times_norm': [], 
                'interval_iats': [], 
                'interval_inv_iat_logs': [], 
                'interval_cumul': ['interval_cumul_norm'], 
                'interval_cumul_norm': [], 
                'interval_rates': [],

                'dcf': [],
             }

    def __init__(self, process_options = ('dirs',), interval_size=0.03, **kwargs):
        self.process_options = process_options if process_options else {}
        self.input_channels = len(self.process_options)
        self.cache = dict()
        self.interval_size = interval_size

        assert len(self.process_options) > 0
        assert all(opt in self.DEPENS.keys() for opt in self.process_options)

    def _resolve_depens(self, opt):
        """get list of options that depend on opt"""
        depens = []
        for depen in self.DEPENS[opt]:
            depens.append(depen)
            depens.extend(self._resolve_depens(depen))
        return depens

    def _is_enabled(self, *opts):
        """if opt or any of its dependencies are in self.process_options, then func returns true"""
        res = self.cache.get('-'.join(opts), None)
        if res is not None:
            return res
        else:
            required = list(opts)
            for opt in opts:
                required.extend(self._resolve_depens(opt))
            res = any(opt in self.process_options for opt in required)
            self.cache['-'.join(opt)] = res
            return res

    def process(self, x):
        """Map raw metadata to processed packet representations using NumPy."""

        if x.shape[0] <= 0:
            return np.empty((0, self.input_channels))

        def fix_size(z, size):
            if z.shape[0] < size:
                z = np.pad(z, (0, size - z.shape[0]), mode='constant')
            elif z.shape[0] > size:
                z = z[:size]
            return z

        feature_dict = {}

        times = x[:, 0]
        feature_dict['times'] = times
        sizes = np.abs(x[:, 1])
        feature_dict['sizes'] = sizes
        dirs = np.sign(x[:, 1])
        feature_dict['dirs'] = dirs

        upload = dirs > 0
        download = ~upload

        if self._is_enabled("time_dirs"):
            feature_dict['time_dirs'] = times * dirs

        if self._is_enabled("size_dirs"):
            size_dirs = sizes * dirs
            feature_dict['size_dirs'] = size_dirs

        if self._is_enabled("times_norm"):
            times_norm = times.copy()
            times_norm -= np.mean(times_norm)
            times_norm /= np.max(np.abs(times_norm))
            feature_dict['times_norm'] = times_norm

        if self._is_enabled("iats"):
            iats = np.diff(times, prepend=0)
            feature_dict['iats'] = iats

        if self._is_enabled("dcf"):
            feature_dict['dcf'] = np.concatenate((feature_dict['iats'] * feature_dict['dirs'] * 1000.,
                                                  feature_dict['size_dirs'] / 1000.))

        if self._is_enabled("cumul"):
            cumul = np.cumsum(size_dirs)
            feature_dict['cumul'] = cumul

        if self._is_enabled("cumul_norm"):
            cumul_norm = cumul.copy()
            cumul_norm -= np.mean(cumul_norm)
            cumul_norm /= np.max(np.abs(cumul_norm))
            feature_dict['cumul_norm'] = cumul_norm

        if self._is_enabled("burst_edges"):
            burst_edges = np.diff(dirs, prepend=0)
            feature_dict['burst_edges'] = burst_edges

        if self._is_enabled("iat_dirs"):
            iat_dirs = (1. + iats) * dirs
            feature_dict['iat_dirs'] = iat_dirs

        if self._is_enabled('running_rates'):
            running_rates = rate_estimator(iats, sizes)
            feature_dict['running_rates'] = running_rates

        if self._is_enabled('running_rates_diff'):
            running_rate_diff = np.diff(running_rates, prepend = [0])
            feature_dict['running_rates_diff'] = running_rate_diff

        if self._is_enabled('running_rates_decayed'):
            running_rates_decay = weighted_rate_estimator(iats)
            feature_dict['running_rates_decayed'] = running_rates_decay

        if self._is_enabled('up_iats'):
            upload_iats = np.diff(times[upload], prepend=[0])
            feature_dict['up_iats'] = upload_iats

        if self._is_enabled('down_iats'):
            download_iats = np.diff(times[download], prepend=[0])
            feature_dict['down_iats'] = download_iats

        if self._is_enabled('up_rates'):
            up_rates = rate_estimator(upload_iats, sizes[upload])
            feature_dict['up_rates'] = up_rates

        if self._is_enabled('up_rates_sparse'):
            sparse_up_rate = np.zeros_like(times)
            sparse_up_rate[upload] = up_rates
            feature_dict['up_rates_sparse'] = sparse_up_rate

        if self._is_enabled('up_rates_decayed'):
            up_rates_decay = weighted_rate_estimator(upload_iats)
            feature_dict['up_rates_decayed'] = up_rates_decay

        if self._is_enabled('down_rates'):
            down_rates = rate_estimator(download_iats, sizes[download])
            feature_dict['down_rates'] = down_rates

        if self._is_enabled('down_rates_sparse'):
            sparse_down_rate = np.zeros_like(times)
            sparse_down_rate[download] = down_rates
            feature_dict['down_rates_sparse'] = sparse_down_rate

        if self._is_enabled('down_rates_decayed'):
            down_rates_decay = weighted_rate_estimator(download_iats)
            feature_dict['down_rates_decayed'] = down_rates_decay

        ## recombine calculated iats into chronological flow
        if self._is_enabled('flow_iats'):
            flow_iats = np.zeros_like(times)
            flow_iats[upload] = upload_iats
            flow_iats[download] = download_iats
            feature_dict['flow_iats'] = flow_iats

        ## filter times by bursts (returns sparse vector)
        if self._is_enabled('burst_filtered_times'):
            delta_times = flow_iats < 0.01
            feature_dict['burst_filtered_times'] = times[delta_times]

        if self._is_enabled('burst_filtered_time_dirs'):
            feature_dict['burst_filtered_time_dirs'] = times[delta_times] * dirs[delta_times]

        if self._is_enabled('inv_iat_logs'):
            # inverse log of iats (adjusted from original to keep logs positive)
            inv_iat_logs = np.log(np.nan_to_num((1 / (flow_iats+1e-6))+1, nan=1e4, posinf=1e4))
            feature_dict['inv_iat_logs'] = inv_iat_logs

        if self._is_enabled('inv_iat_log_dirs'):
            feature_dict['inv_iat_log_dirs'] = inv_iat_logs * dirs

        # Interval-based features
        if self._is_enabled('interval_dirs_up', 'interval_dirs_down', 
                            'interval_size_up', 'interval_size_down',
                            'interval_times', 'interval_times_norm', 
                            'interval_iats', 'interval_inv_iat_logs',
                            'interval_cumul', 'interval_cumul_norm', 
                            'interval_rates'):

            num_intervals = int(np.ceil(np.max(times) / self.interval_size))

            split_points = np.searchsorted(times, np.arange(0, num_intervals) * self.interval_size)

            if self._is_enabled('interval_dirs_up', 'interval_dirs_down', 
                                'interval_size_up', 'interval_size_down'):
                dirs_subs = np.array_split(dirs, split_points)
                size_subs = np.array_split(sizes, split_points)

                interval_dirs_up = np.zeros(num_intervals + 1)
                interval_dirs_down = np.zeros(num_intervals + 1)
                for i, tensor in enumerate(dirs_subs):
                    size = tensor.size
                    if size > 0:
                        up = np.sum(tensor >= 0)
                        interval_dirs_up[i] = up
                        interval_dirs_down[i] = size - up

                if self._is_enabled('interval_dirs_up'):
                    feature_dict['interval_dirs_up'] = interval_dirs_up

                if self._is_enabled('interval_dirs_down'):
                    feature_dict['interval_dirs_down'] = interval_dirs_down

                if self._is_enabled('interval_dirs_sum'):
                    feature_dict['interval_dirs_sum'] = interval_dirs_up + interval_dirs_down

                if self._is_enabled('interval_dirs_sub'):
                    feature_dict['interval_dirs_sub'] = interval_dirs_up - interval_dirs_down

                if self._is_enabled('interval_size_up', 'interval_size_down'):
                    size_subs = np.array_split(sizes * dirs, split_points)

                    interval_size_up = np.zeros(num_intervals + 1)
                    interval_size_down = np.zeros(num_intervals + 1)
                    for i, tensor in enumerate(size_subs):
                        size = tensor.size
                        if size > 0:
                            up = np.sum(tensor >= 0)
                            interval_size_up[i] = up
                            down = np.sum(tensor <= 0)
                            interval_size_down[i] = down

                    if self._is_enabled('interval_size_up'):
                        feature_dict['interval_size_up'] = interval_size_up

                    if self._is_enabled('interval_size_down'):
                        feature_dict['interval_size_down'] = interval_size_down

                    if self._is_enabled('interval_size_sum'):
                        feature_dict['interval_size_sum'] = interval_size_up + interval_size_down

                    if self._is_enabled('interval_size_sub'):
                        feature_dict['interval_size_sub'] = interval_size_up - interval_size_down

            if self._is_enabled('interval_times'):
                times_subs = np.array_split(times, split_points)
                interval_times = np.zeros(num_intervals + 1)
                for i, tensor in enumerate(times_subs):
                    if tensor.size > 0:
                        interval_times[i] = np.mean(tensor)
                    elif i > 0:
                        interval_times[i] = interval_times[i - 1]
                feature_dict['interval_times'] = interval_times

            if self._is_enabled('interval_times_norm'):
                interval_times_norm = interval_times.copy()
                interval_times_norm -= np.mean(interval_times_norm)
                interval_times_norm /= np.amax(np.abs(interval_times_norm))
                feature_dict['interval_times_norm'] = interval_times_norm

            if self._is_enabled('interval_iats'):
                iats_subs = np.array_split(iats, split_points)
                interval_iats = np.zeros(num_intervals+1)
                for i,tensor in enumerate(iats_subs):
                    if tensor.size > 0:
                        interval_iats[i] = tensor.mean()
                    elif i > 0:
                        interval_iats[i] = interval_iats[i-1] + self.interval_size
                feature_dict['interval_iats'] = interval_iats

            if self._is_enabled('interval_inv_iat_logs'):
                inv_iat_logs_subs = np.array_split(inv_iat_logs, split_points)
                interval_inv_iat_logs = np.zeros(num_intervals+1)
                for i,tensor in enumerate(inv_iat_logs_subs):
                    if tensor.size > 0:
                        interval_inv_iat_logs[i] = tensor.mean()
                feature_dict['interval_inv_iat_logs'] = interval_inv_iat_logs

            if self._is_enabled('interval_cumul'):
                cumul_subs = np.array_split(cumul, split_points)
                interval_cumul = np.zeros(num_intervals+1)
                for i,tensor in enumerate(cumul_subs):
                    if tensor.size > 0:
                        interval_cumul[i] = tensor.mean()
                    elif i > 0:
                        interval_cumul[i] = interval_cumul[i-1]
                feature_dict['interval_cumul'] = interval_cumul

            if self._is_enabled('interval_cumul_norm'):
                interval_cumul_norm = interval_cumul.copy()
                interval_cumul_norm -= np.mean(interval_cumul_norm)
                interval_cumul_norm /= np.amax(np.abs(interval_cumul_norm))
                feature_dict['interval_cumul_norm'] = interval_cumul_norm
                
            if self._is_enabled('interval_rates'):
                interval_rates = rate_estimator(interval_iats, np.abs(interval_cumul))
                feature_dict['interval_rates'] = interval_rates

        # Adjust feature vectors sizes to match traffic sequence length and stack
        if len(self.process_options) > 1:
            target_size = max([feature_dict[opt].size for opt in self.process_options])
            feature_stack = [fix_size(feature_dict[opt], target_size) for opt in self.process_options]
        else:
            feature_stack = [feature_dict[self.process_options[0]]]

        features = np.stack(feature_stack, axis=-1)
        features = np.nan_to_num(features)

        return features

    def __call__(self, x):
        return self.process(x)
