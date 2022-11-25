import torch

def r2_score(preds, labels):
    mean_label = torch.mean(labels)
    return 1 - (((labels - preds)**2).sum() / ((labels - mean_label)**2).sum())