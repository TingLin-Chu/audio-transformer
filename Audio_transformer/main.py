# packages
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from copy import deepcopy

# local modules
from config import *
from dataset import *
from model import *
from training import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
MODE = "RTFM" if USE_RTFM else "BCE"
"""
if USE_RTFM:
    raise NotImplementedError("RTFM is not implemented yet")
"""
# Check paths
if not os.path.exists(CHECK_POINTS_DIR):
    os.makedirs(CHECK_POINTS_DIR)
if not os.path.exists(SCORES_DIR):
    os.makedirs(SCORES_DIR)
if not os.path.exists(RECORDS_DIR):
    os.makedirs(RECORDS_DIR)

# Load previous result: latest/best model, best score
# Set paths
latest_model_path = os.path.join(CHECK_POINTS_DIR, f"{MODE}_latest.pt")
best_model_path = os.path.join(CHECK_POINTS_DIR, f"{MODE}_best.pt")
latest_score_path = os.path.join(SCORES_DIR, f"{MODE}_latest.txt")
best_score_path = os.path.join(SCORES_DIR, f"{MODE}_best.txt")
records_path = os.path.join(RECORDS_DIR, f"{MODE}_records.csv")


def load_results(model):
    best_model_state = None
    cur_best_score = 0
    cur_best_epoch = 0
    latest_score = 0
    latest_epoch = 0
    records = []
    if LOAD_PREVIOUS_RESULT:
        err = None
        # Check if the latest model exists
        if not os.path.exists(latest_model_path):
            err = "Latest model state doesn't exist"
        elif not os.path.exists(best_model_path):
            err = "Best model state doesn't exist"
        elif not os.path.exists(best_score_path):
            err = "Best score record doesn't exist"
        elif not os.path.exists(latest_score_path):
            err = "Best score record doesn't exist"
        else:
            # Load latest model
            best_model_state = torch.load(best_model_path, weights_only=True)
            model.load_state_dict(torch.load(
                latest_model_path, weights_only=True))
            # Load best score/epoch
            with open(best_score_path, 'r') as f:
                cur_best_score = float(f.readline())
                cur_best_epoch = int(f.readline())
            # Load latest score/epoch
            with open(latest_score_path, 'r') as f:
                latest_score = float(f.readline())
                latest_epoch = int(f.readline())
        if err is not None:
            print(f"Failed to load previous results: {err}")
            print("Going to train from the beginning")
        else:
            print("Successfully loaded previous result")

    return model, best_model_state, cur_best_score, cur_best_epoch, latest_score, latest_epoch, records


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataset
    print("Preparing datasets......")
    if USE_RTFM:
        training_dataloader_normal, training_dataloader_abnormal = load_training_dataset_RTFM()
    else:
        training_dataloader = load_training_dataset()
    testing_dataloader = load_testing_dataset()

    # Prepare model
    print("Preparing model......")
    model = AudioTransformer_RTFM(model_dim=MODEL_DIM, output_dim=OUTPUT_DIM, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
                                  num_layers=NUM_LAYERS, device=device) if USE_RTFM else AudioTransformer(model_dim=MODEL_DIM, output_dim=OUTPUT_DIM, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
                                                                                                          num_layers=NUM_LAYERS)
    criterion = RTFM_loss() if USE_RTFM else nn.BCELoss()
    # criterion = nn.BCELoss() if USE_RTFM else
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model, best_model_state, cur_best_score, cur_best_epoch, latest_score, latest_epoch, records = load_results(model)

    print("======================================= Start Training=============================================")
    model = model.to(device)
    auc_roc_scores = []
    for epoch in range(1, EPOCHS+1):
        # Traing
        avg_loss = train_RTFM(training_dataloader_normal, training_dataloader_abnormal, model, criterion, optimizer,
                              device, epoch) if USE_RTFM else train(training_dataloader, model, criterion, optimizer, device, epoch)
        # Validation
        auc_roc_score = testing_RTFM(testing_dataloader, model, device, epoch) if USE_RTFM else test(
            testing_dataloader, model, device, epoch)

        # Rounding loss/score
        avg_loss = round(avg_loss, 4)
        auc_roc_score = round(auc_roc_score, 4)

        # Showing results of the current epoch
        latest_epoch += 1
        auc_roc_scores.append(auc_roc_score)
        print(
            f"Epoch {epoch}/{EPOCHS}, Average Loss: {avg_loss}, AUC_ROC: {auc_roc_score}")
        records.append(
            {"Epoch": latest_epoch, "Average_Loss": avg_loss, "AUC_ROC": auc_roc_score})

        if auc_roc_score > cur_best_score:
            cur_best_score = auc_roc_score
            cur_best_epoch = latest_epoch
            best_model_state = deepcopy(model.state_dict())
            # torch.save(model.state_dict(), best_model_path)
        latest_score = auc_roc_score

    print("======================================= End Training=============================================")

    # Showing results
    print("======================================= Result =============================================")
    print(
        f"Max score: {max(auc_roc_scores)}, Average score: {sum(auc_roc_scores)/len(auc_roc_scores)}")

    # Saving results
    # Save models
    print("Saving best model...")
    torch.save(best_model_state, best_model_path)
    print("Saving latest model...")
    torch.save(model.state_dict(), latest_model_path)

    # Save scores
    with open(best_score_path, 'w') as f:
        f.write(str(cur_best_score) + '\n')
        f.write(str(cur_best_epoch))
    with open(latest_score_path, 'w') as f:
        f.write(str(latest_score) + '\n')
        f.write(str(latest_epoch))

    # Save records
    print("Saving records...")
    if len(records) > 0:
        records_df = pd.DataFrame(records)
        records_mode = 'a' if LOAD_PREVIOUS_RESULT else 'w'
        records_df.to_csv(records_path, mode=records_mode,
                          header=not os.path.exists(records_path), index=False)


if __name__ == "__main__":
    main()
