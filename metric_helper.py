import pickle
import numpy as np
from sklearn.metrics import auc


def partial_roc_auc_normalized(fpr, tpr, fpr_min, fpr_max):
    """
    Calculate the normalized partial AUC of an ROC curve between the specified FPR range [fpr_min, fpr_max].
    
    The partial AUC is normalized to the range [0, 1], where 1 represents the maximum possible AUC in the given FPR range.
    
    Parameters:
    - fpr: Array of False Positive Rates (FPR)
    - tpr: Array of True Positive Rates (TPR) corresponding to the FPR values
    - fpr_min: Lower bound of FPR for the partial AUC calculation
    - fpr_max: Upper bound of FPR for the partial AUC calculation

    Returns:
    - partial_auc_normalized: The normalized partial AUC between fpr_min and fpr_max
    """
    # Ensure fpr and tpr are numpy arrays for easier slicing and manipulation
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    
    # Select the indices where fpr is within the desired range
    mask = (fpr >= fpr_min) & (fpr <= fpr_max)
    
    # Add the boundary points (fpr_min and fpr_max) if not already present
    if not any(fpr == fpr_min):
        tpr_min = np.interp(fpr_min, fpr, tpr)
        fpr = np.append(fpr, fpr_min)
        tpr = np.append(tpr, tpr_min)
        
    if not any(fpr == fpr_max):
        tpr_max = np.interp(fpr_max, fpr, tpr)
        fpr = np.append(fpr, fpr_max)
        tpr = np.append(tpr, tpr_max)
    
    # Sort by FPR after appending
    sorted_indices = np.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]
    
    # Mask again after adding boundaries
    mask = (fpr >= fpr_min) & (fpr <= fpr_max)
    
    # Calculate partial AUC using the trapezoidal rule within the masked region
    partial_auc = auc(fpr[mask], tpr[mask])
    
    # Normalize the partial AUC by dividing by the maximum possible AUC in this range (fpr_max - fpr_min)
    max_possible_auc = fpr_max - fpr_min
    partial_auc_normalized = partial_auc / max_possible_auc
    
    return partial_auc_normalized


results_pickles = [
    #"res/Espresso_ssh/res.pkl",
    #"res/Espresso_socat/res.pkl",
    #"res/Espresso_icmp/res.pkl",
    #"res/Espresso_dns/res.pkl",
    #"res/Espresso_mixed3/res.pkl",
    #"res/Espresso_ssh_host/res.pkl",
    "res/Espresso_socat_host/res.pkl",
    #"res/Espresso_icmp_host/res.pkl",
    #"res/Espresso_dns_host/res.pkl",
    #"res/Espresso_mixed_host/res.pkl",
        ]
display_names = [
    #"ESPRESSO-SSH",
    "ESPRESSO-SOCAT",
    #"ESPRESSO-ICMP",
    #"ESPRESSO-DNS",
    #"ESPRESSO-MIXED",
        ]

for name,fpath in zip(display_names, results_pickles):
    print(name)
    with open(fpath, 'rb') as fi:
        data = pickle.load(fi)
    fpr, tpr, = data['fpr'], data['tpr']
    
    full_auc = auc(fpr, tpr)

    # Define your FPR threshold (e.g., 10e-4)
    fpr_threshold = 1e-5

    # Filter the FPR and TPR values to include only the desired range
    fpr_filtered = fpr[fpr <= fpr_threshold]
    tpr_filtered = tpr[:len(fpr_filtered)]

    # Compute the partial AUC using the trapezoidal rule
    partial_auc = partial_roc_auc_normalized(fpr, tpr, np.min(fpr_filtered), fpr_threshold) 

    # Display the result
    print(f"AUC-ROC (full curve): {full_auc}")
    print(f"Max TPR for FPR <= {fpr_threshold}: {np.amax(tpr_filtered)}")
    print(f"pAUC-ROC for FPR <= {fpr_threshold}: {partial_auc}")
    print('')
