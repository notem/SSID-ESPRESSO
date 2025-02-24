import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn import metrics
from tqdm import tqdm
import argparse
import os
import pickle
import transformers

# enable if NaN or other odd behavior appears
#torch.autograd.set_detect_anomaly(True)
# disable any unnecessary logging / debugging
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# auto-optimize cudnn ops
torch.backends.cudnn.benchmark = True


class MyDataset(Dataset):
    def __init__(self, sims, labels, ratio=25):
        self.inputs = sims
        self.targets = np.array(labels)
        self.indices = np.arange(len(self.targets))

        self.tot_pos = np.sum(self.targets)
        self.tot_neg = len(self.targets) - self.tot_pos
        
        #corr_weight = self.tot_neg / self.tot_pos
        corr_weight = self.tot_neg / self.tot_pos
        corr_weight /= ratio
        self.weights = torch.DoubleTensor([corr_weight if label else 1 for label in self.targets])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (torch.tensor(self.inputs[idx], dtype=torch.float), 
                torch.tensor(self.targets[idx], dtype=torch.float))


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

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    dirpath = os.path.dirname(args.results_file)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    with open(args.sims_file, 'rb') as fi:
        data = pickle.load(fi)

    # Create PyTorch datasets
    va_sims = data['va_sims']
    va_labels = data['va_labels']
    te_sims = data['te_sims']
    te_labels = data['te_labels']
    
    # Load the data
    #val_data = np.load('/home/njm3308/SSID/espresso/data/val_output.npy')
    #test_data = np.load('/home/njm3308/SSID/espresso/data/test_output.npy')

    #cutoff = 108

    ## Split the data into inputs and targets
    #va_sims = val_data[:, :-2]
    #va_labels = val_data[:, -1]
    #te_sims = test_data[:, :-2]
    #te_labels = test_data[:, -1]
    
    tr_dataset = MyDataset(va_sims, va_labels)
    te_dataset = MyDataset(te_sims, te_labels)
    del data
    
    # Create PyTorch dataloaders
    tr_batch_size = 256
    te_batch_size = 2048*16
    num_epochs = 50
    
    # use weighted sampler to balance training dataset
    tr_sampler = WeightedRandomSampler(tr_dataset.weights, len(tr_dataset.weights))
    tr_loader = DataLoader(tr_dataset, 
            batch_size = tr_batch_size, 
            sampler = tr_sampler,
            pin_memory = True,
            )
    # (no sampler) evaluate on all pairwise cases in test set
    te_loader = DataLoader(te_dataset, 
            batch_size = te_batch_size, 
            shuffle = False)
    
    # Instantiate the model and move it to GPU if available
    model = Predictor(dim = tr_dataset.inputs.shape[-1], 
                      drop = args.dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Define the loss function and the optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                                len(tr_loader), 
                                                len(tr_loader)*num_epochs)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_pred = 0
        total_pred = 0
        with tqdm(total=len(tr_loader)) as pbar:
            for i, (inputs, targets) in enumerate(tr_loader):
                # Move tensors to the correct device
                inputs, targets = inputs.to(device), targets.to(device)
    
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                logits = torch.sigmoid(outputs) >= 0.5
                correct_pred += (logits == targets).sum().item()
                total_pred += targets.size(0)
    
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
    
                running_loss += loss.item()
                pbar.set_description(f'Epoch {epoch+1}')
                pbar.set_postfix({'loss': running_loss / (i+1), 
                                  'acc': correct_pred / total_pred})
                pbar.update(1)
                
    # Put the model in evaluation mode
    model.eval()
    
    # Lists to store the model's outputs and the actual targets
    outputs_list = []
    targets_list = []
    
    # Pass the validation data through the model
    with torch.no_grad():
        correct_pred = 0
        total_pred = 0
        with tqdm(total=len(te_loader)) as pbar:
            for i, (inputs, targets) in enumerate(te_loader):
                # Move tensors to the correct device
                inputs, targets = inputs.to(device), targets.to(device)
    
                # Forward pass
                outputs = torch.sigmoid(model(inputs)) 
    
                # Store the outputs and targets
                outputs_list.append(outputs.cpu().numpy())
                targets_list.append(targets.cpu().numpy())

                # metrics
                logits = outputs >= 0.5
                correct_pred += (logits == targets).sum().item()
                total_pred += targets.size(0)
                
                pbar.set_description(f'Testing...')
                pbar.set_postfix({'acc': correct_pred / total_pred})
                pbar.update(1)

        test_acc = correct_pred / total_pred
    
    # Compute the ROC curve
    targets_list = np.concatenate(targets_list)
    outputs_list = np.concatenate(outputs_list)
    fpr, tpr, thresholds = metrics.roc_curve(targets_list, outputs_list,
                                             drop_intermediate = True)
    roc_auc = metrics.auc(fpr, tpr)
    print(f"Test accuracy: {test_acc}, roc: {roc_auc}")

    # store results
    with open(args.results_file, 'wb') as fi:
        pickle.dump({
                     'fpr': fpr, 
                     'tpr': tpr,
                     'thresholds': thresholds,
                     'tot_pos': te_dataset.tot_pos,
                     'tot_neg': te_dataset.tot_neg,
                     }, fi)
    
    import matplotlib.pyplot as plt
    plt.title(f'Receiver Operating Characteristic')
    plt.plot(fpr[1:], tpr[1:], 
             'b-', label = 'AUC = %0.6f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot(np.linspace(0, 1, 100000), 
             np.linspace(0, 1, 100000), 
             'k--')
    plt.xlim([1e-8, 1])
    plt.ylim([-0.03, 1.])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xscale('log')
    plt.savefig(os.path.join(dirpath, 'roc.pdf'))