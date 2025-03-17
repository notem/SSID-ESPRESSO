import torch
import numpy as np
import torch.nn as nn
from sklearn import metrics
from tqdm import tqdm
import argparse
import os
import pickle

# enable if NaN or other odd behavior appears
#torch.autograd.set_detect_anomaly(True)
# disable any unnecessary logging / debugging
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# auto-optimize cudnn ops
torch.backends.cudnn.benchmark = True


class Predictor(nn.Module):
    """
    Simple MLP for binary prediction
    """
    def __init__(self, dim, 
                 drop=0.5, 
                 #ratio=4, 
                 layers=3):
        super(Predictor, self).__init__()
        modules = []
        #hid_dim = int(dim*ratio)
        hid_dim = 128
        for i in range(layers):
            fc = nn.Sequential(
                    nn.Linear(dim, hid_dim) if i == 0 else nn.Linear(hid_dim, hid_dim),
                    nn.GELU(),
                    #nn.BatchNorm1d(hid_dim),
                    )
            modules.append(fc)

        self.fc_modules = nn.ModuleList(modules)
        self.pred = nn.Linear(hid_dim, 1)
        self.dropout = nn.Dropout(drop)
        self.mlp_dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(x)
        for i,module in enumerate(self.fc_modules):
            x = module(x)
            x = self.mlp_dropout(x)
        x = self.pred(x)
        return x.flatten()


def parse_args():
    parser = argparse.ArgumentParser(
                        prog = 'benchmark-mlp.py',
                        description = 'Evaluate FEN correlation performance using an MLP classifier.',
                        #epilog = 'Text at the bottom of help'
                        )

    # experiment configuration options
    parser.add_argument('--sims_file', 
                        default = './path/to/sims.pkl', 
                        type = str,
                        help = "Load the pickle filepath containing the pre-calculated similarity matrix.", 
                        required=True)
    parser.add_argument('--results_file', 
                        default = '/data/path2', 
                        type = str,
                        help = "Path where to save the results (as a pickle file).", 
                        required=True)
    parser.add_argument('--dropout',
                        default = 0.0,
                        type=float,
                        help="Dropout percentage during training. \
                            This dropout rate is applied to all layer (including the input layer).",
                  )
    parser.add_argument('--ckpt',
                        default = None,
                        type=str,
                        help="Save/resume from checkpoint path.",
                  )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    dirpath = os.path.dirname(args.results_file)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    with open(args.sims_file, 'rb') as fi:
        data = pickle.load(fi)

    # Create PyTorch datasets
    #va_sims = data['va_sims']
    #va_labels = data['va_labels']
    te_sims = data['te_sims']
    te_labels = data['te_chain_labels']
    te_hosts = data['te_host_labels']
    print(te_sims.shape, te_labels.shape, te_hosts.shape)
    print(len(np.unique(te_labels)))
    
    # Create PyTorch dataloaders
    te_batch_size = 2048*16
    num_epochs = 50
    
    print(f"Resuming from checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt)
    
    # Instantiate the model and move it to GPU if available
    model = Predictor(dim = checkpoint['dim'], 
                      drop = args.dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Run predictions on the full test set
    y_pred = []
    labels = []
    cur_batch_x = []
    cur_batch_labels = []
    for i in range(te_sims.shape[0]):
        for j in range(te_sims.shape[1]):
            if i == j: continue   # skip self-similarity
            
            sample = torch.tensor(te_sims[i,j]).to(device)
            correlated = te_labels[i] == te_labels[j]
            cur_batch_labels.append([correlated, 
                                     te_labels[i], te_labels[j], 
                                     te_hosts[i], te_hosts[j]])
            cur_batch_x.append(sample)
            if len(cur_batch_x) >= te_batch_size:
                batch = torch.stack(cur_batch_x, dim=0).to(torch.float32)
                batch_probs = torch.sigmoid(model(batch))
                y_pred.append(batch_probs.cpu().detach().numpy())
                labels.append(np.array(cur_batch_labels))
                cur_batch_x = []
                cur_batch_labels = []
    if len(cur_batch_x) > 0:
        batch = torch.stack(cur_batch_x, dim=0).to(torch.float32)
        batch_probs = torch.sigmoid(model(batch))
        y_pred.append(batch_probs.cpu().detach().numpy())
        labels.append(np.array(cur_batch_labels))
    
    # process into numpy array
    y_pred = np.concatenate(y_pred)
    labels = np.concatenate(labels)
    
    # Save the results for posterity
    results = {
        'y_pred': y_pred,
        'labels': labels,
    }
    with open(args.results_file, 'wb') as fo:
        pickle.dump(results, fo)
    print(f"Results saved to {args.results_file}")
    
    # Evaluate the results
    fpr, tpr, thresholds = metrics.roc_curve(labels[:,0], y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print(f"Overall ROC AUC: {roc_auc}")
    
    # Interpolate to get the corresponding thresholds
    num_thresholds = 50
    sampled_thresholds = np.linspace(0, 1, num_thresholds)
    #sampled_thresholds = np.interp(sampled_fpr, fpr[::-1], thresholds[::-1])

    # Now evaluate your custom metrics at these thresholds
    custom_metrics = {}
    for threshold in sampled_thresholds:
    
        # Evaluate results by same host
        results = {'TP': dict(),'FP': dict(),'TN': dict(),'FN': dict()}
        for i in range(len(y_pred)):
            if labels[i][0]:
                same_host = labels[i,3] == labels[i,4]
                if not same_host:
                    continue
                if y_pred[i] > threshold:
                    results['TP'][labels[i,1]] = 1 + results['TP'].get(labels[i,1], 0)
                else:
                    results['FN'][labels[i,1]] = 1 + results['FN'].get(labels[i,1], 0)
            else:
                if y_pred[i] > threshold:
                    results['FP'][labels[i,1]] = 1 + results['FP'].get(labels[i,1], 0)
                else:
                    results['TN'][labels[i,1]] = 1 + results['TN'].get(labels[i,1], 0)
        # Calculate metrics on an average per-chain basis
        per_chain_results = {}
        for chain in np.unique(labels[:,1]):
            TP = results['TP'].get(chain, 0)
            FP = results['FP'].get(chain, 0)
            TN = results['TN'].get(chain, 0)
            FN = results['FN'].get(chain, 0)
            
            precision = TP / (TP + FP + 1e-6)
            recall = TP / (TP + FN + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            accuracy = (TP + TN) / (TP + FP + TN + FN + 1e-6)
            fpr = FP / (FP + TN + 1e-6)
            per_chain_results[chain] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fpr': fpr,
                'accuracy': accuracy,
            }
        # Calculate the average of the per-chain results
        avg_precision = np.mean([x['precision'] for x in per_chain_results.values()])
        stdev_precision = np.std([x['precision'] for x in per_chain_results.values()])
        avg_recall = np.mean([x['recall'] for x in per_chain_results.values()])
        stdev_recall = np.std([x['recall'] for x in per_chain_results.values()])
        avg_f1 = np.mean([x['f1'] for x in per_chain_results.values()])
        stdev_f1 = np.std([x['f1'] for x in per_chain_results.values()])
        avg_fpr = np.mean([x['fpr'] for x in per_chain_results.values()])
        stdev_fpr = np.std([x['fpr'] for x in per_chain_results.values()])
    
        # Calculate percent of chains with all positive pairs identified (e.g., TP == sum(labels==i))
        percent_chains = 0
        for chain in per_chain_results:
            TP = results['TP'].get(chain, 0)
            total_pairs = np.sum((labels[:,1] == chain) \
                & (labels[:,1] == labels[:,2]) \
                & (labels[:,3] == labels[:,4]))
            if TP == total_pairs:
                percent_chains += 1
        percent_chains /= len(per_chain_results)
        
        custom_metrics[threshold] = {
            'precision': avg_precision,
            'recall': avg_recall,
            'fpr': avg_fpr,
            'f1': avg_f1,
            'chain-accuracy': percent_chains,
            'per-chain': per_chain_results,
        }
        
        print(f"Threshold: {threshold:.2f} " +
              f"| Precision: {avg_precision:.3f}±{stdev_precision:0.2f} " +
              f"| Recall: {avg_recall:.3f}±{stdev_recall:0.2f} " +
              f"| FPR: {avg_fpr:.3f}±{stdev_fpr:0.2f} " +
              f"| F1: {avg_f1:.3f}±{stdev_f1:0.2f} " +
              f"| Chains: {percent_chains:.3f}")