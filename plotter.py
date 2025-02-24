import pickle
import numpy as np
from sklearn import metrics
import os

results_pickles = [
            #'/home/njm3308/SSID/tuned_deepcoffea/results_dcf/ssh-net/metrics.pkl',
            '/home/njm3308/SSID/tuned_deepcoffea/results_dcf/ssh-host/metrics.pkl',
            #'/home/njm3308/SSID/tuned_deepcoffea/results_dcf/mixed-net/metrics.pkl',
            #'res/Espresso_ssh_host/res.pkl',
            #'res/Espresso_socat_host/res.pkl',
            #'res/Espresso_icmp_host/res.pkl',
            #'res/Espresso_dns_host/res.pkl',
            #'res/Espresso_mixed_host/res.pkl',
            'res/Espresso_ssh/res.pkl',
            #'res/Espresso_socat/res.pkl',
            #'res/Espresso_icmp/res.pkl',
            #'res/Espresso_dns/res.pkl',
            #'res/Espresso_mixed3/res.pkl',
        ]
display_names = [
            #'SSH-only',
            #'SOCAT-only',
            #'ICMP-only',
            #'DNS-only',
            #'Mixed',
            'DCF',
            'ESPRESSO',
        ]


fpr_range = (3e-6, 5e-1)
    
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

for name,fpath in zip(display_names, results_pickles):
    print(name)
    with open(fpath, 'rb') as fi:
        data = pickle.load(fi)
    fpr, tpr, = data['fpr'], data['tpr']
    
    if 'tot_pos' in data.keys():
        tot_pos, tot_neg = data['tot_pos'], data['tot_neg']
        acc = ((tot_pos*tpr[0]) + (tot_neg*(1-fpr[0]))) / (tot_pos + tot_neg)
    else:
        acc = -1.
    tpr = np.concatenate(([0.], tpr, [1.]))
    fpr = np.concatenate(([0.], fpr, [1.]))
    roc_auc = metrics.auc(fpr, tpr)
    print(f"Test accuracy: {acc}, roc: {roc_auc}")
    
    # filter out non-useful thresholds (creates a monotonic ROC)
    #cur_max = -np.inf
    #keep_idx = []
    #for i in range(len(fpr)):
    #    if tpr[i] > cur_max:
    #        keep_idx.append(i)
    #        cur_max = tpr[i]
    #fpr = fpr[keep_idx]
    #tpr = tpr[keep_idx]
    
    # plot ROC curve
    ax.plot(fpr, tpr, label = name)
    plt.title(f'Receiver Operating Characteristic')
    plt.legend(loc = 'lower right')
    plt.xlim([fpr_range[0], fpr_range[1]])
    plt.ylim([-0.03, 1.03])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xscale('log')
    #ax.yaxis.grid(True, which="minor")
    plt.minorticks_on()
    plt.savefig('result-imgs/roc_ssh_net_comparison2.png', dpi=500)