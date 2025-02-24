import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin
        self.cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, anchor, positive, negative):
        pos_sim = self.cosine_sim(anchor, positive)
        neg_sim = self.cosine_sim(anchor, negative)
        loss = F.relu(neg_sim - pos_sim + self.margin)
        return loss.mean()

#class OnlineCosineTripletLoss(nn.Module):
#    def __init__(self, margin=0.1):
#        super(OnlineCosineTripletLoss, self).__init__()
#        self.margin = margin
#
#    def _get_triplet_mask(self, labels):
#        """Return a 3D mask where mask[a, p, n] is True if the triplet (a, p, n) is valid.
#
#        A triplet (i, j, k) is valid if:
#            - i, j, k are distinct
#            - labels[i] == labels[j] and labels[i] != labels[k]
#
#        Args:
#            labels: torch.Tensor of dtype torch.int32 with shape [batch_size]
#        """
#        # Check that i, j, and k are distinct
#        indices_equal = torch.eye(labels.size(0)).bool().to(labels.device)
#        indices_not_equal = ~indices_equal
#        i_not_equal_j = indices_not_equal.unsqueeze(2)
#        i_not_equal_k = indices_not_equal.unsqueeze(1)
#        j_not_equal_k = indices_not_equal.unsqueeze(0)
#
#        distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k
#
#        # Check if labels[i] == labels[j] and labels[i] != labels[k]
#        labels = labels.unsqueeze(0)
#        label_equal = labels == labels.transpose(0, 1)
#        i_equal_j = label_equal.unsqueeze(2)
#        i_equal_k = label_equal.unsqueeze(1)
#
#        valid_labels = i_equal_j & (~i_equal_k)
#
#        # Combine the two masks
#        mask = distinct_indices & valid_labels
#
#        return mask
#
#    def forward(self, embeddings, labels,
#                ignore_zero = False):
#        # Normalize each vector (element) to have unit norm
#        norms = torch.norm(embeddings, p=2, dim=-1, keepdim=True)  # Compute L2 norms
#        embeddings = embeddings / norms  # Divide by norms to normalize
#        
#        # Compute pairwise cosine similarity
#        if embeddings.dim() == 2:
#            all_sim = torch.mm(embeddings, embeddings.t())
#        elif embeddings.dim() == 3:
#            all_sim = torch.matmul(embeddings.permute(1,0,2), embeddings.permute(1,2,0))
#            all_sim = all_sim.mean(0)
#
#        # mask of valid triplets
#        mask = self._get_triplet_mask(labels).float()
#
#        # expand dims for pairwise comparison
#        anc_pos_sim = all_sim.unsqueeze(-1)
#        anc_neg_sim = all_sim.unsqueeze(-2)
#
#        loss = F.relu(anc_neg_sim - anc_pos_sim + self.margin) * mask
#
#        # calculate average loss (disregarding invalid & easy triplets)
#        nonzero = torch.count_nonzero(loss)
#        if ignore_zero and nonzero > 0:
#            loss = torch.sum(loss) / nonzero
#        else:
#            loss = loss.mean()
#        return loss
#
#
#class OnlineHardCosineTripletLoss(nn.Module):
#    def __init__(self, margin=0.1):
#        super(OnlineHardCosineTripletLoss, self).__init__()
#        self.margin = margin
#
#    def _get_anc_pos_triplet_mask(self, labels):
#        """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have the same label.
#
#        Args:
#            labels: torch.Tensor of dtype torch.int32 with shape [batch_size]
#
#        Returns:
#            mask: torch.Tensor of dtype torch.bool with shape [batch_size, batch_size]
#        """
#        # Check that i and j are distinct
#        indices_equal = torch.eye(labels.size(0)).to(labels.device).bool()
#        indices_not_equal = ~indices_equal
#
#        # Check if labels[i] == labels[j]
#        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
#
#        # Combine the two masks
#        mask = indices_not_equal & labels_equal
#
#        return mask
#
#    def _get_anc_neg_triplet_mask(self, labels):
#        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
#
#        Args:
#            labels: torch.Tensor of dtype torch.int32 with shape [batch_size]
#
#        Returns:
#            mask: torch.Tensor of dtype torch.bool with shape [batch_size, batch_size]
#        """
#        return labels.unsqueeze(0) != labels.unsqueeze(1)
#
#    def forward(self, embeddings, labels, 
#                ignore_zero = True,
#                use_hard_negative_loss = False):
#        """
#        Args:
#            embeddings: torch.Tensor -- batch of feature embeddings with shape [batch_size, features] or [batch_size, windows, features]
#            labels: torch.Tensor of dtype torch.int32 with shape [batch_size]
#            use_ciq_mean: bool -- use the mean of the interquartile range (e.g., exclude high and low quartiles from mean)
#            use_hard_negative_loss: bool -- when enabled, the positive loss component is ignored when the hardest pos is too easy (e.g. pos_sim < neg_sim)
#        """
#        # Normalize each vector (element) to have unit norm
#        norms = torch.norm(embeddings, p=2, dim=-1, keepdim=True)  # Compute L2 norms
#        embeddings = embeddings / norms  # Divide by norms to normalize
#        
#        # Compute pairwise cosine similarity
#        if embeddings.dim() == 2:
#            all_sim = torch.mm(embeddings, embeddings.t())
#
#        elif embeddings.dim() == 3:
#            all_sim = torch.matmul(embeddings.permute(1,0,2), embeddings.permute(1,2,0))
#            all_sim = all_sim.mean(0)
#
#        # find hardest positive pairs (when positive has low sim)
#        # mask of all valid positives
#        mask_anc_pos = self._get_anc_pos_triplet_mask(labels)
#        # prevent invalid pos by increasing sim
#        anc_pos_sim = all_sim + (~mask_anc_pos * 999).float()
#        # select minimum sim positives
#        hardest_pos_sim = anc_pos_sim.min(dim=1, keepdim=True)[0]
#
#        # find hardest negative triplets (when negative has high sim)
#        # mask of all valid negatives
#        mask_anc_neg = self._get_anc_neg_triplet_mask(labels).float()
#        # set invalid negatives to 0
#        anc_neg_sim = all_sim * mask_anc_neg
#        # select maximum sim negatives
#        hardest_neg_sim = anc_neg_sim.max(dim=1, keepdim=True)[0]
#
#        if use_hard_negative_loss:
#            # selective contrastive loss 
#            selective_idx = hardest_neg_sim > hardest_pos_sim
#            hardest_pos_sim[selective_idx] = 0.
#
#        loss = F.relu(hardest_neg_sim - hardest_pos_sim + self.margin)
#
#        # calculate average loss (disregarding invalid & easy triplets)
#        nonzero = torch.count_nonzero(loss)
#        if ignore_zero and nonzero > 0:
#            loss = torch.sum(loss) / nonzero
#        else:
#            loss = loss.mean()
#        return loss



def compute_sim(in_emb, out_emb, mean=True):
    """
    """
    # Normalize each vector (element) to have unit norm
    norms = torch.norm(in_emb, p=2, dim=-1, keepdim=True)  # Compute L2 norms
    in_emb = in_emb / norms  # Divide by norms to normalize
    
    norms = torch.norm(out_emb, p=2, dim=-1, keepdim=True)  # Compute L2 norms
    out_emb = out_emb / norms  # Divide by norms to normalize
    
    # Compute pairwise cosine similarity
    if in_emb.dim() == 2:       # DCF-style output
        all_sim = torch.mm(in_emb, out_emb.t())
    elif in_emb.dim() == 3:     # ESPRESSO-style output
        #all_sim = torch.matmul(in_emb.permute(1,0,2), out_emb.permute(1,2,0))
        all_sim = torch.bmm(in_emb.permute(1,0,2), out_emb.permute(1,2,0))
        if mean:
            all_sim = all_sim.mean(0)  # mean across the window dim (otherwise hard-mining performs very poorly)

    return all_sim


class OnlineCosineTripletLoss(nn.Module):
    """
    """
    def __init__(self, margin = 0.1,
                 semihard = True):
        super(OnlineCosineTripletLoss, self).__init__()
        self.margin = margin
        self.semihard = semihard

    def _get_triplet_mask(self, labels):
        """Return a 3D mask where mask[a, p, n] is True if the triplet (a, p, n) is valid.

        A triplet (i, j, k) is valid if:
            - labels[i] == labels[j] and labels[i] != labels[k]

        Args:
            labels: torch.Tensor of dtype torch.int32 with shape [batch_size]
        """
        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        labels = labels.unsqueeze(0)
        label_equal = labels == labels.transpose(0, 1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)

        valid_labels = i_equal_j & (~i_equal_k)

        return valid_labels

    def forward(self, in_embeddings, out_embeddings):
        """
        """
        all_sim = compute_sim(in_embeddings, out_embeddings)

        labels = torch.arange(in_embeddings.size(0)).to(in_embeddings.get_device())

        # mask of valid triplets
        mask = self._get_triplet_mask(labels).float()
        if all_sim.dim() == 3:
            mask = mask.unsqueeze(0)

        # expand dims for pairwise comparison
        anc_pos_sim = all_sim.unsqueeze(-1)
        anc_neg_sim = all_sim.unsqueeze(-2)

        loss = F.relu(anc_neg_sim - anc_pos_sim + self.margin) * mask
        return loss

        # calculate average loss (disregarding invalid & easy triplets)
        nonzero = torch.count_nonzero(loss)
        if self.semihard and nonzero > 0:
            loss = torch.sum(loss) / nonzero
        else:
            loss = torch.sum(loss) / torch.sum(mask)
        return loss


class OnlineHardCosineTripletLoss(nn.Module):
    """
    """
    def __init__(self, margin = 0.1, 
                 semihard = True, 
                 hard_negative_loss = False):
        super(OnlineHardCosineTripletLoss, self).__init__()
        self.margin = margin
        self.semihard = semihard
        self.hard_negative_loss = hard_negative_loss

    def forward(self, in_embeddings, out_embeddings):
        """
        Args:
        """
        all_sim = compute_sim(in_embeddings, out_embeddings)

        # find hardest positive pairs (when positive has low sim)
        # mask of all valid positives
        mask_anc_pos = torch.eye(in_embeddings.size(0)).to(in_embeddings.get_device()).bool()
        if all_sim.dim() == 3:
            mask_anc_pos = mask_anc_pos.unsqueeze(0)
        # prevent invalid pos by increasing sim
        anc_pos_sim = all_sim + (~mask_anc_pos * 999).float()
        # select minimum sim positives
        hardest_pos_sim = anc_pos_sim.min(dim=-1, keepdim=True)[0]

        # find hardest negative triplets (when negative has high sim)
        # set invalid negatives to 0
        anc_neg_sim = all_sim * ~mask_anc_pos
        # select maximum sim negatives
        hardest_neg_sim = anc_neg_sim.max(dim=-1, keepdim=True)[0]

        if self.hard_negative_loss:
            # selective contrastive loss 
            selective_idx = hardest_neg_sim > hardest_pos_sim
            hardest_pos_sim[selective_idx] = 0.
        loss = F.relu(hardest_neg_sim - hardest_pos_sim + self.margin)
        return loss

        # calculate average loss (disregarding invalid & easy triplets)
        nonzero = torch.count_nonzero(loss)
        if self.semihard and nonzero > 0:
            loss = torch.sum(loss) / nonzero
        else:
            loss = loss.mean()
        return loss