import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import *


class Audio_clip_dataset(Dataset):
    def __init__(self, root_dir, is_train=False, use_zero_padding=USE_ZERO_PADDING):
        dir_name = "Training" if is_train else "Testing"
        dir_name = dir_name + "_zero" if use_zero_padding else dir_name

        self.root_dir = root_dir
        self.is_train = is_train
        self.data_dir = os.path.join(root_dir, dir_name)
        self.normal_flag = '_label_A'
        self.data = []

        for embeddings_file in os.listdir(self.data_dir):
            video_name = embeddings_file.split('.')[0]
            label = 0.0 if self.normal_flag in video_name else 1.0

            embeddings_path = os.path.join(self.data_dir, embeddings_file)
            embeddings_video = torch.from_numpy(np.load(embeddings_path))

            # embeddings_video.shape = (C, F, E) --> (F, C, E)
            embeddings_video = torch.permute(embeddings_video, (1, 0, 2))
            # embeddings_video.shape = (F, C, E) --> (F, C*E)
            embeddings_video = torch.flatten(embeddings_video, start_dim=1)

            data = {"input": embeddings_video, "label": label}
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Audio_clip_dataset_RTFM(Dataset):
    def __init__(self, root_dir, is_train=True, use_zero_padding=USE_ZERO_PADDING, is_normal=False):
        dir_name = "Training" if is_train else "Testing"
        dir_name = dir_name + "_zero" if use_zero_padding else dir_name

        self.root_dir = root_dir
        self.is_train = is_train
        self.is_normal = is_normal
        self.data_dir = os.path.join(root_dir, dir_name)
        self.normal_flag = '_label_A'
        self.data = []

        for embeddings_file in os.listdir(self.data_dir):
            video_name = embeddings_file.split('.')[0]
            if (is_train) and (is_normal != (self.normal_flag in video_name)):
                continue
            label = 0.0 if self.normal_flag in video_name else 1.0

            embeddings_path = os.path.join(self.data_dir, embeddings_file)
            embeddings_video = torch.from_numpy(np.load(embeddings_path))

            # embeddings_video.shape = (C, F, E) --> (F, C, E)
            embeddings_video = torch.permute(embeddings_video, (1, 0, 2))
            # embeddings_video.shape = (F, C, E) --> (F, C*E)
            embeddings_video = torch.flatten(embeddings_video, start_dim=1)

            data = {"input": embeddings_video, "label": label}
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    ret = {}

    # Split inputs and labels
    inputs = [bi["input"] for bi in batch]
    labels = [bi["label"] for bi in batch]
    # Find maximum seq_len(F)
    max_seq_len = max(xi.size(0) for xi in inputs)

    # Padding data to same seq_len
    padded_inputs = [nn.functional.pad(
        xi, (0, 0, 0, max_seq_len-xi.size(0))) for xi in inputs]

    # Stack together
    ret["input"] = torch.stack(padded_inputs, dim=0)
    ret["label"] = torch.tensor(labels)
    # print(f"inputs: {ret['input'].size()}, labels: {ret['label'].size()}")

    return ret


def load_training_dataset():
    # Training dataset
    training_dataset = Audio_clip_dataset(
        root_dir=DATASET_ROOT_DIR, is_train=True, use_zero_padding=USE_ZERO_PADDING)
    training_dataloader = DataLoader(
        training_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    return training_dataloader


def load_training_dataset_RTFM():
    training_dataset_RTFM_normal = Audio_clip_dataset_RTFM(
        root_dir=DATASET_ROOT_DIR, is_train=True, use_zero_padding=USE_ZERO_PADDING, is_normal=True)
    training_dataset_RTFM_abnormal = Audio_clip_dataset_RTFM(
        root_dir=DATASET_ROOT_DIR, is_train=True, use_zero_padding=USE_ZERO_PADDING, is_normal=False)

    training_dataloader_RTFM_normal = DataLoader(
        training_dataset_RTFM_normal, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    training_dataloader_RTFM_abnormal = DataLoader(
        training_dataset_RTFM_abnormal, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    return (training_dataloader_RTFM_normal, training_dataloader_RTFM_abnormal)


def load_testing_dataset():
    # Testing dataset
    testing_dataset = Audio_clip_dataset(
        root_dir=DATASET_ROOT_DIR, is_train=False, use_zero_padding=USE_ZERO_PADDING)
    testing_dataloader = DataLoader(
        testing_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    return testing_dataloader
