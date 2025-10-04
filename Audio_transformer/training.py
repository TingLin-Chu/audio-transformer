# packages
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# local modules
from config import *

# 2025/5/20: complete smooth()


def sparsity(arr, lamda2):
    # arr: (B, C, 1) --> (B, C)
    arr = torch.squeeze(arr)
    # arr: (B, C) --> (B,)
    arr = torch.norm(arr, dim=1)
    loss = torch.mean(arr)
    return lamda2*loss


def smooth(arr, lamda1):
    # arr: (B, C, 1) --> (B, C)
    arr = torch.squeeze(arr)

    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    # arr_diff: (B, C) --> (B,)
    arr_diff = arr2-arr
    arr_diff = torch.square(arr_diff)
    arr_diff = torch.sum(arr_diff, dim=1)

    loss = torch.mean(arr_diff)
    return lamda1*loss


# =============================================== Training BCE =================================================
def train(dataloader, model, criterion, optimizer, device, epoch):
    total_loss = 0

    model.train()
    for batch in tqdm(dataloader, desc=f"Training Epoch: {epoch}/{EPOCHS}"):
        inputs, labels = batch['input'], batch['label'].float()
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss/len(dataloader)

# =============================================== Testing BCE =================================================


def test(dataloader, model, device, epoch):
    y_trues = []
    y_preds = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validating Epoch: {epoch}/{EPOCHS}"):
            # Assume having batch_size = 1
            inputs, labels = batch['input'], batch['label'].float()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = torch.stack([(oi >= 0.5)
                                  for oi in outputs], dim=0).float()
            y_trues.extend(labels)
            y_preds.extend(outputs)

    y_preds = torch.tensor(y_preds).cpu()
    y_trues = torch.tensor(y_trues).cpu()
    return roc_auc_score(y_trues, y_preds)

# =============================================== Training RTFM =================================================


def train_RTFM(normal_dataloader, abnormal_dataloader, model, criterion, optimizer, device, epoch):
    total_loss = 0
    epoch_len = max(len(normal_dataloader), len(abnormal_dataloader))

    model.train()
    for i in tqdm(range(1, epoch_len+1), desc=f"Training Epoch: {epoch}/{EPOCHS}"):
        if (i-1) % len(normal_dataloader) == 0:
            normal_dataloader_iter = iter(normal_dataloader)
        if (i-1) % len(abnormal_dataloader) == 0:
            abnormal_dataloader_iter = iter(abnormal_dataloader)

        batch_normal = next(normal_dataloader_iter)
        batch_abnormal = next(abnormal_dataloader_iter)

        batch_len = min(len(batch_normal['label']), len(batch_abnormal['label']))

        inputs_normal, labels_normal = batch_normal['input'][:batch_len], batch_normal['label'][:batch_len].float()
        inputs_normal, labels_normal = inputs_normal[:batch_len].to(device), labels_normal[:batch_len].to(device)

        inputs_abnormal, labels_abnormal = batch_abnormal['input'][:
                                                                   batch_len], batch_abnormal['label'][:batch_len].float()
        inputs_abnormal, labels_abnormal = inputs_abnormal[:batch_len].to(
            device), labels_abnormal[:batch_len].to(device)

        optimizer.zero_grad()

        # score_abnormal, score_normal: (B, 1)
        # feat_select_abn, feat_select_normal: (B, k_abn, E)
        # abn_scores: (B, C, 1)
        score_abnormal, score_normal, feat_select_abn, feat_select_normal, abn_scores = model.forward_training(
            inputs_normal, inputs_abnormal)
        loss_sparse = sparsity(abn_scores, 8e-3)
        loss_smooth = smooth(abn_scores, 8e-4)

        cost = criterion(score_normal, score_abnormal, labels_normal, labels_abnormal,
                         feat_select_normal, feat_select_abn) + loss_smooth + loss_sparse
        cost.backward()
        optimizer.step()

        total_loss += cost.item()
    return total_loss/epoch_len
# =============================================== Testing RTFM =================================================


def testing_RTFM(dataloader, model, device, epoch):
    epoch_len = len(dataloader)
    y_trues = []
    y_preds = []

    model.eval()
    with torch.no_grad():
        for batch_in in tqdm(dataloader, desc=f"Validating Epoch: {epoch}/{EPOCHS}"):
            x_in, labels_in = batch_in['input'], batch_in['label'].float()
            x_in, labels_in = x_in.to(device), labels_in.to(device)

            # scores: (B,)
            scores = model.forward_testing(x_in)

            # scores = torch.squeeze(scores, dim=1)

            # Original: Mean of all scores in a bag
            # New: Mean of top-k scores in a bag
            outputs = torch.stack([(oi >= 0.5)for oi in scores], dim=0).float()

            # Extend results
            y_trues.extend(labels_in)
            y_preds.extend(outputs)

    y_preds = torch.tensor(y_preds).cpu()
    y_trues = torch.tensor(y_trues).cpu()
    return roc_auc_score(y_trues, y_preds)
