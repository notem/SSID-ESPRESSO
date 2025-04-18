import os
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import argparse
import torch.nn.functional as F
import math

import torch
from torch import nn
from torch.nn import functional as F
from utils.processor import DataProcessor
from utils.data import *

class DFModel(nn.Module):
    def __init__(self, input_shape=(5, 1000), emb_size=64):
        super(DFModel, self).__init__()
        
        self.block1_conv1 = nn.Conv1d(input_shape[0], 32, 8, padding='same')
        self.block1_adv_act1 = nn.ELU()
        self.block1_conv2 = nn.Conv1d(32, 32, 8, padding='same')
        self.block1_adv_act2 = nn.ELU()
        self.block1_pool = nn.MaxPool1d(8, stride=3, padding=2)
        self.block1_dropout = nn.Dropout(0.1)

        self.block2_conv1 = nn.Conv1d(32, 64, 8, padding='same')
        self.block2_act1 = nn.ReLU()
        self.block2_conv2 = nn.Conv1d(64, 64, 8, padding='same')
        self.block2_act2 = nn.ReLU()
        self.block2_pool = nn.MaxPool1d(8, stride=3, padding=2)
        self.block2_dropout = nn.Dropout(0.1)

        self.block3_conv1 = nn.Conv1d(64, 128, 8, padding='same')
        self.block3_act1 = nn.ReLU()
        self.block3_conv2 = nn.Conv1d(128, 128, 8, padding='same')
        self.block3_act2 = nn.ReLU()
        self.block3_pool = nn.MaxPool1d(8, stride=3, padding=2)
        self.block3_dropout = nn.Dropout(0.1)

        self.block4_conv1 = nn.Conv1d(128, 256, 8, padding='same')
        self.block4_act1 = nn.ReLU()
        self.block4_conv2 = nn.Conv1d(256, 256, 8, padding='same')
        self.block4_act2 = nn.ReLU()
        self.block4_pool = nn.MaxPool1d(8, stride=3, padding=2)

        self.flatten = nn.Flatten()
        self.input_shape = input_shape
        
        flat_size = self._get_flattened_size()
        self.dense = nn.Linear(flat_size, emb_size)

    def _get_flattened_size(self):
        x = torch.zeros(1, *self.input_shape)
        x = self.block1_pool(self.block1_conv2(self.block1_conv1(x)))
        x = self.block2_pool(self.block2_conv2(self.block2_conv1(x)))
        x = self.block3_pool(self.block3_conv2(self.block3_conv1(x)))
        x = self.block4_pool(self.block4_conv2(self.block4_conv1(x)))
        x = self.flatten(x)
        return x.size(1)

    def forward(self, x):
        x = self.block1_pool(self.block1_adv_act2(self.block1_conv2(self.block1_adv_act1(self.block1_conv1(x)))))
        x = self.block1_dropout(x)
        x = self.block2_pool(self.block2_act2(self.block2_conv2(self.block2_act1(self.block2_conv1(x)))))
        x = self.block2_dropout(x)
        x = self.block3_pool(self.block3_act2(self.block3_conv2(self.block3_act1(self.block3_conv1(x)))))
        x = self.block3_dropout(x)
        x = self.block4_pool(self.block4_act2(self.block4_conv2(self.block4_act1(self.block4_conv1(x)))))

        x = self.flatten(x)
        x = self.dense(x)
        return x
 
def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model for inflow and outflow data.")
    #parser.add_argument('--train_dir', type=str, default='data/', help="Directory containing the training .npy files.")
    #parser.add_argument('--val_dir', type=str, default='data/', help="Directory containing the validation .npy files.")
    parser.add_argument('--data', type=str, help="Directory containing the data files.")
    parser.add_argument('--bs', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--margin', type=float, default=0.1, help="Margin for the triplet loss function.")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Initial learning rate.")
    parser.add_argument('--num_epochs', type=int, default=1000, help="Number of epochs to train.")
    parser.add_argument('--model_name', type=str, default='final.pth', help="Name of the saved model file.")
    parser.add_argument('--ckpt_dir',
                        default = './checkpoint',
                        type = str,
                        help = "Set directory for model checkpoints.")
    parser.add_argument('--temporal_alignment', 
                        action = 'store_true', default=False,
                        help = 'Use temporal alignment loss'
                    )
    parser.add_argument('--batched_windows', 
                        action='store_true', default=False,
                        help = "Include all windows of samples within batches.")
    parser.add_argument('--results_dir', 
                        default = './results',type = str,
                        help = "Set directory for result logs."
                    )
    parser.add_argument('--single_fen', 
                        default = False, action = 'store_true',
                        help = 'Use the same FEN for in and out flows.')
    parser.add_argument('--multi_feats', 
                        default = False, action = 'store_true',
                        help = 'Use multiple packet-based feature representations.')
    parser.add_argument('--interval_feats', 
                        default = False, action = 'store_true',
                        help = 'Use time interval-based feature representations.')
    return parser.parse_args()
 
# Define the learning rate schedule function
def lr_schedule(epoch, num_epochs, initial_lr):
    if epoch < 100:
        # Warmup phase
        return initial_lr * (epoch / 100)
    else:
        # Cosine annealing
        return initial_lr * 0.5 * (1 + math.cos(math.pi * (epoch - 100) / (num_epochs - 100)))
 
class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin
        self.cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
 
    def forward(self, anchor, positive, negative):
        pos_sim = self.cosine_sim(anchor, positive)
        neg_sim = self.cosine_sim(anchor, negative)
        loss = F.relu(neg_sim - pos_sim + self.margin)
        return torch.mean(loss,dim=-1)
 
def main():
    args = parse_args()
 
    window_kwargs = {
        "window_count": 11,
        "window_width": 8,
        "window_overlap": 4,
        "include_all_window": False,
        "adjust_times": True,
    }
    features = ['dcf']
    input_size = 1000
    
    if args.multi_feats:
        features = ('inv_iat_logs', 
                    'iats', 
                    'sizes', 
                    'burst_edges', 
                    'running_rates', 
                    'cumul_norm', 
                    'times_norm')
    
    if args.interval_feats:
        features = (
            "interval_dirs_up",
            "interval_dirs_down",
            "interval_dirs_sum",
            "interval_dirs_sub",
            "interval_size_up",
            "interval_size_down",
            "interval_size_sum",
            "interval_size_sub",
            "interval_iats",
            "interval_inv_iat_logs",
            "interval_cumul_norm",
            "interval_times_norm"
            )
        input_size = 200
        window_kwargs["adjust_times"] = False
        
    processor = DataProcessor(features)

    inflow, outflow, _ = build_dataset(args.data, processor, 
                                             input_size, input_size, in_idx=1, 
                                             window_kwargs = window_kwargs,
                                             )
    #out_idx = -1
    #outflow = np.array([x[out_idx] for x in outflow])
    
    te_idx, va_idx, tr_idx = get_split_idx(len(inflow))
    
    val_inflows = inflow[va_idx]
    val_outflows = outflow[va_idx]
    train_inflows = inflow[tr_idx]
    train_outflows = outflow[tr_idx]
    print(val_inflows.shape, val_outflows.shape)
 
    # Define the datasets
    train_dataset = TripletDataset(train_inflows, train_outflows, 
                                   as_window = not args.batched_windows)
    val_dataset = TripletDataset(val_inflows, val_outflows, 
                                 as_window = not args.batched_windows)
 
    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.bs, 
                              )
    val_loader = DataLoader(val_dataset, batch_size=args.bs, 
                            )
 
    # Instantiate the models
    inflow_model = DFModel(input_shape=(len(features),input_size), emb_size=64)
    params = list(inflow_model.parameters())
    if not args.single_fen:
        outflow_model = DFModel(input_shape=(len(features),input_size), emb_size=64)
        params +=  list(outflow_model.parameters())
    else:
        outflow_model = inflow_model
 
    # Move models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inflow_model.to(device)
    outflow_model.to(device)
 
    # Define the loss function and the optimizer
    criterion = CosineTripletLoss(margin=args.margin)
    optimizer = optim.Adam(params, lr=args.learning_rate)
 
    # Training loop
    
    try:
        best_val_loss = float("inf")
        history = {}
        for epoch in range(args.num_epochs):
            metrics = {}
            lr = lr_schedule(epoch, args.num_epochs, args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            train_dataset.reset_split()
            val_dataset.reset_split()
 
            # Training
            inflow_model.train()
            outflow_model.train()
 
            running_loss = 0.0
            for anchor, positive, negative in train_loader:
                anchor = anchor.float().to(device)
                positive = positive.float().to(device)
                negative = negative.float().to(device)
                N = anchor.size(0)

                if args.batched_windows:
                    anchor = anchor.view(anchor.size(0)*anchor.size(1), 
                                            anchor.size(2), anchor.size(3))
                    positive = positive.view(positive.size(0)*positive.size(1), 
                                            positive.size(2), positive.size(3))
                    negative = negative.view(negative.size(0)*negative.size(1), 
                                            negative.size(2), negative.size(3))
 
                anchor_embeddings = inflow_model(anchor)
                positive_embeddings = outflow_model(positive)
                negative_embeddings = outflow_model(negative)

                if args.batched_windows:
                    anchor_embeddings = anchor_embeddings.view(N, -1, anchor_embeddings.size(-1))
                    positive_embeddings = positive_embeddings.view(N, -1, positive_embeddings.size(-1))
                    negative_embeddings = negative_embeddings.view(N, -1, negative_embeddings.size(-1))
 
                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                
                if args.temporal_alignment and args.batched_windows:
                    t_loss = criterion(anchor_embeddings.permute(0,2,1),
                                        positive_embeddings.permute(0,2,1),
                                        negative_embeddings.permute(0,2,1))
                    a_loss = ((loss - t_loss)**2)
                    loss = (loss + t_loss) / 2 + a_loss
                loss = loss.mean()
 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
 
                running_loss += loss.item()
 
            train_loss = running_loss / len(train_loader)
            metrics['train_loss'] = train_loss
 
            # Validation
            inflow_model.eval()
            outflow_model.eval()
 
            running_loss = 0.0
            with torch.no_grad():
                for anchor, positive, negative in val_loader:
                    anchor = anchor.float().to(device)
                    positive = positive.float().to(device)
                    negative = negative.float().to(device)
                    N = anchor.size(0)

                    if args.batched_windows:
                        anchor = anchor.view(anchor.size(0)*anchor.size(1), 
                                                anchor.size(2), anchor.size(3))
                        positive = positive.view(positive.size(0)*positive.size(1), 
                                                positive.size(2), positive.size(3))
                        negative = negative.view(negative.size(0)*negative.size(1), 
                                                negative.size(2), negative.size(3))
 
                    anchor_embeddings = inflow_model(anchor)
                    positive_embeddings = outflow_model(positive)
                    negative_embeddings = outflow_model(negative)

                    if args.batched_windows:
                        anchor_embeddings = anchor_embeddings.view(N, -1, anchor_embeddings.size(-1))
                        positive_embeddings = positive_embeddings.view(N, -1, positive_embeddings.size(-1))
                        negative_embeddings = negative_embeddings.view(N, -1, negative_embeddings.size(-1))
 
                    loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                    
                    if args.temporal_alignment and args.batched_windows:
                        t_loss = criterion(anchor_embeddings.permute(0,2,1),
                                            positive_embeddings.permute(0,2,1),
                                            negative_embeddings.permute(0,2,1))
                        a_loss = ((loss - t_loss)**2)
                        loss = (loss + t_loss) / 2 + a_loss
                    loss = loss.mean()
 
                    running_loss += loss.item()
 
            val_loss = running_loss / len(val_loader)
            metrics['val_loss'] = val_loss
 
            print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')
 
            # Save the model if it's the best one so far
            #if val_loss < best_val_loss:
            #    print("Best model so far!")
            #    best_val_loss = val_loss
            os.makedirs(args.ckpt_dir, exist_ok=True)
            save_path = os.path.join(args.ckpt_dir, args.model_name)
            torch.save({
                'epoch': epoch,
                'config': {
                            'name': 'DCF', 
                            'features': features,
                            'window_kwargs': window_kwargs,
                            'feature_dim': 64,
                            'inflow_size': input_size,
                            'outflow_size': input_size,
                        },
                'inflow_fen': inflow_model.state_dict(),
                'outflow_fen': outflow_model.state_dict(),
                'opt': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, save_path)
            
            history['epochs'] = metrics
            
    except KeyboardInterrupt:
        pass

    finally:
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
            
        results_fp = f'{args.results_dir}/log.txt'
        with open(results_fp, 'w') as fi:
            json.dump(history, fi, indent='\t')
 
if __name__ == "__main__":
    main()
 
