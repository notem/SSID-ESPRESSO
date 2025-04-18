import numpy as np
import torch 
from torch.distributions.exponential import Exponential


class NumpyHisto:
    
    def __init__(self, data):
        self.data = data
        self.hist, self.bin_edges = np.histogram(data, bins=100)

    def sample(self, size):
        # Generate random indices based on the histogram
        indices = np.random.choice(len(self.hist), size=size, p=self.hist/np.sum(self.hist))
        
        # Use the bin edges to get the corresponding values
        sampled_values = np.random.uniform(self.bin_edges[indices], self.bin_edges[indices + 1])
        
        return sampled_values

def insert_dummy_packets_torch(sizes, times, directions, 
                               num_dummy_packets = 0,
                               size_distributions = None):
    if num_dummy_packets == 0:
        return sizes, times, directions
    device = sizes.device

    num_dummy_packets = np.random.randint(0, num_dummy_packets)

    # Assume packets with direction 0 are non-existent (padding)
    real_packet_mask = directions != 0

    # Filter out non-existent packets
    real_sizes = sizes[real_packet_mask]
    real_times = times[real_packet_mask]
    real_directions = directions[real_packet_mask]

    # Generate dummy packets
    dummy_directions = torch.randint(0, 2, (num_dummy_packets,), device=device) * 2 - 1  # Converts {0, 1} to {-1, 1}
    dummy_times = torch.linspace(0, 5, num_dummy_packets, device=device)

    # Generate sizes for dummy packets based on their directions
    dummy_sizes = torch.zeros(num_dummy_packets, device=device)
    download_mask = dummy_directions == -1
    upload_mask = dummy_directions == 1
    download_amount = download_mask.sum().item()
    upload_amount = upload_amount.sum().item()
    if not size_distributions:
        dummy_sizes[download_mask] = torch.randint(100, 1001, (download_amount,), device=device).float()
        dummy_sizes[upload_mask] = torch.randint(50, 201, (upload_amount,), device=device).float()
    else:
        dummy_sizes[download_mask] = torch.tensor(size_distributions[0].sample(download_amount), device=device).float()
        dummy_sizes[upload_mask] = torch.tensor(size_distributions[1].sample(upload_amount), device=device).float()

    # Combine filtered real packets with dummy packets
    combined_sizes = torch.cat((real_sizes, dummy_sizes))
    combined_times = torch.cat((real_times, dummy_times))
    combined_directions = torch.cat((real_directions, dummy_directions))

    # Sort the combined packets based on times
    sorted_indices = torch.argsort(combined_times)
    sorted_sizes = combined_sizes[sorted_indices]
    sorted_times = combined_times[sorted_indices]
    sorted_directions = combined_directions[sorted_indices]

    return sorted_sizes, sorted_times, sorted_directions


def calculate_inter_packet_times(times):
    # Ensure 'prepend' has the same dimensionality as 'times', except for the last dimension
    batch_size, seq_length = times.shape
    prepend_tensor = torch.zeros((batch_size, 1), device=times.device)  # Match the batch dimension, add a single column for prepend
    
    non_padded_diff = torch.diff(times, dim=1, prepend=prepend_tensor)
    # Since you're computing the difference along the last dimension (time sequence), no need to pad after diff
    
    return torch.abs(non_padded_diff)


def calculate_times_with_directions(times, directions):
    return times * directions


def calculate_cumulative_traffic(sizes, times):
    # Assuming 'sizes' and 'times' are PyTorch tensors
    # This method might need adjustments based on the exact representation of 'times'
    cumulative_traffic = torch.div(torch.cumsum(sizes, dim=1), 1000)
    return cumulative_traffic


def delay_and_sort_traffic_packets(sizes, times, directions, 
                                   max_delay=1.0, delay_probability=0.2):
    device = times.device

    # Generate a random mask to select packets for delaying
    delay_mask = torch.rand(times.size(), device=device) < delay_probability

    # Generate random delays for each packet
    delays = torch.zeros_like(times)
    delays[delay_mask] = torch.rand(sum(delay_mask), device=device) * max_delay

    # Apply the delays to the timestamps
    adjusted_times = times + delays

    # Sort the packets by the new timestamps to ensure chronological order
    sorted_indices = torch.argsort(adjusted_times)

    # Use the sorted indices to reorder sizes, times, and directions
    adjusted_sizes = sizes[sorted_indices]
    adjusted_times = adjusted_times[sorted_indices]
    adjusted_directions = directions[sorted_indices]

    return adjusted_sizes, adjusted_times, adjusted_directions

def insert_dummy_packets_torch_exponential(sizes, times, directions, 
                                           num_dummy_packets=0, total_duration=5.0,
                                           size_distributions=None):
    if num_dummy_packets == 0:
        return sizes, times, directions
    
    device = sizes.device

    num_dummy_packets = np.random.randint(0, num_dummy_packets)

    # Assume packets with direction 0 are non-existent (padding)
    real_packet_mask = directions != 0

    # Filter out non-existent packets
    real_sizes = sizes[real_packet_mask]
    real_times = times[real_packet_mask]
    real_directions = directions[real_packet_mask]

    # Generate dummy packets' directions
    dummy_directions = torch.randint(0, 2, (num_dummy_packets,), device=device) * 2 - 1  # Converts {0, 1} to {-1, 1}

    dummy_sizes = torch.zeros(num_dummy_packets, device=device)
    download_mask = dummy_directions == -1
    upload_mask = dummy_directions == 1
    download_amount = download_mask.sum().item()
    upload_amount = upload_amount.sum().item()
    if not size_distributions:
        dummy_sizes[download_mask] = torch.randint(100, 1001, (download_amount,), device=device).float()
        dummy_sizes[upload_mask] = torch.randint(50, 201, (upload_amount,), device=device).float()
    else:
        dummy_sizes[download_mask] = torch.tensor(size_distributions[0].sample(download_amount), device=device).float()
        dummy_sizes[upload_mask] = torch.tensor(size_distributions[1].sample(upload_amount), device=device).float()

    # Calculate the parameter for the exponential distribution based on the desired number of dummy packets and total duration
    if num_dummy_packets > 0:
        lambda_param = num_dummy_packets / total_duration
    else:
        lambda_param = 1.0  # Default value to avoid division by zero

    # Generate dummy packets' times using the exponential distribution
    exp_dist = Exponential(rate=lambda_param)
    dummy_times = torch.cumsum(exp_dist.sample((num_dummy_packets,)).to(device), dim=0)

    # Combine filtered real packets with dummy packets
    combined_sizes = torch.cat((real_sizes, dummy_sizes))
    combined_times = torch.cat((real_times, dummy_times))
    combined_directions = torch.cat((real_directions, dummy_directions))

    # Sort the combined packets based on times
    sorted_indices = torch.argsort(combined_times)
    sorted_sizes = combined_sizes[sorted_indices]
    sorted_times = combined_times[sorted_indices]
    sorted_directions = combined_directions[sorted_indices]

    return sorted_sizes, sorted_times, sorted_directions

def calculate_cumulative_traffic_torch(packet_sizes, packet_times):
    batch_size, num_packets = packet_sizes.shape
    device = packet_sizes.device

    # Initialize the output tensor for 51.2 seconds with 0.1 second intervals, for each batch item
    cumulative_traffic = torch.zeros(batch_size, 512, device=device)

    # Convert packet_times to indices within the [0, 512) range
    indices = (packet_times / 0.1).long()

    # Ensure indices are within bounds
    valid_mask = (indices >= 0) & (indices < 512)

    # Iterate over the batch
    for i in range(batch_size):
        # Apply valid_mask for the current batch item
        valid_indices = indices[i][valid_mask[i]]
        valid_sizes = packet_sizes[i][valid_mask[i]]

        # Create an empty tensor to accumulate sizes for the current batch item
        temp_traffic = torch.zeros(512, device=device)
        
        # Accumulate sizes for valid indices
        temp_traffic.index_add_(0, valid_indices, valid_sizes)
        
        # Assign the temp cumulative sum to the corresponding batch item
        cumulative_traffic[i] = temp_traffic

    # Compute the cumulative sum along the second dimension
    cumulative_traffic = torch.cumsum(cumulative_traffic, dim=1)

    # Pad the tensor to a length of 1000 with zeros for each batch item
    padded_cumulative_traffic = torch.nn.functional.pad(cumulative_traffic, (0, 1000 - 512), "constant", 0)

    return torch.log(padded_cumulative_traffic+1)


def calculate_inter_packet_times_torch(times):
    """
    Calculate the inter-packet times for a batch of packet times,
    ensuring the output shape matches the input batch shape [256, 1000].

    Parameters:
    - times: A torch tensor of shape [256, 1000] with packet times.

    Returns:
    - A torch tensor of shape [256, 1000] with the reciprocal of inter-packet times.
    """
    batch_size, seq_length = times.shape
    device = times.device
    dtype = times.dtype

    # Initialize an output tensor to store the reciprocal of IPTs for each sequence
    inter_packet_times_inv_batch = torch.zeros(batch_size, seq_length, device=device, dtype=dtype)

    # Process each sequence individually
    for i in range(batch_size):
        # Extract the current sequence
        sequence = times[i, :]

        # Filter out zeros and calculate differences
        non_zero_times = sequence[sequence != 0]
        non_zero_times_prepended = torch.cat((torch.tensor([0], device=device, dtype=dtype), non_zero_times))
        non_padded_diff = torch.diff(non_zero_times_prepended)

        # Pad the differences to match the original sequence length
        padded_diff = torch.nn.functional.pad(non_padded_diff, (0, seq_length - len(non_padded_diff)), "constant", 0)

        # Calculate the reciprocal, replacing zeros in padded_diff with ones for division
        inter_packet_times_inv = 1.0 / torch.where(padded_diff == 0, torch.ones_like(padded_diff), padded_diff)

        # Store the result in the output tensor
        inter_packet_times_inv_batch[i, :] = inter_packet_times_inv

    return inter_packet_times_inv_batch


def reciprocal_with_zeros(tensor):
    # Create a mask for non-zero elements
    non_zero_mask = tensor != 0.0

    # Initialize a tensor of the same shape filled with zeros
    result = torch.zeros_like(tensor)

    # Apply reciprocal only to non-zero elements
    result[non_zero_mask] = 1.0 / tensor[non_zero_mask]

    return result
