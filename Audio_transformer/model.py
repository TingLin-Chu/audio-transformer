import torch
import torch.nn as nn
import math
from config import *

# 2025/05/23 15:22 ======> Editting forward_testing()
C = CLIP_LEN
E = EMBEDDING_DIM


class PositionalEncoding(nn.Module):
    # Ref:
    # [1] https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
    # [2] https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class AudioTransformer(nn.Module):
    # If the model_dim (C*E) is too high, making lack of memory, we may need to lower its dimention and restore before MLP
    def __init__(self, model_dim, output_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.positional_encoding = PositionalEncoding(model_dim)

        self.class_token = nn.Parameter(torch.rand(1, model_dim))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape = (B, F, C*E) --> (F, B, C*E)
        x = torch.permute(x, (1, 0, 2))

        # Positional encoding
        x = self.positional_encoding(x)

        # Form the tokens for Transformer encoder
        # x.shape = (F, B, C*E) --> (F+1, B, C*E)
        x = torch.permute(x, (1, 0, 2))
        x = torch.stack([torch.vstack((self.class_token, xi)) for xi in x])
        x = torch.permute(x, (1, 0, 2))

        # Transformer encoder
        # x.shape = (F+1, B, C*E) --> (B, C*E)
        x = self.transformer_encoder(x)[0]

        # MLP
        # x.shape = (B, C*E) --> (B, 1)
        print(f"Before MLP: {x.shape}")
        x = self.fc1(x)
        x = x + self.fc2(x)  # residual learning
        x = self.fc3(x)
        print(f"After MLP: {x.shape}")

        # Sigmoid
        x = self.sigmoid(x)

        # Squeeze dim_1 and return
        # x.shape = (B, 1) --> (B,)
        x = torch.squeeze(x, 1)
        print(f"After squeeze: {x.shape}")
        return x


class AudioTransformer_RTFM(nn.Module):
    def __init__(self, model_dim, output_dim, hidden_dim, num_heads, num_layers, device, k_abn=K_ABN, k_nor=K_NOR):
        # output_dim should be equal to CLIP_LEN (# of clips in a video)
        self.device = device
        self.k_abn = k_abn
        self.k_nor = k_nor
        super().__init__()
        self.positional_encoding = PositionalEncoding(model_dim)

        self.class_token = nn.Parameter(torch.rand(1, model_dim))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Sequential(
            nn.Linear(E, hidden_dim),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(DROP_OUT)

    def forward_training(self, x_normal, x_abnormal):
        B, F, _ = x_normal.size()
        # x.shape = (B, F, C*E) --> (F, B, C*E)
        x_normal = torch.permute(x_normal, (1, 0, 2))
        x_abnormal = torch.permute(x_abnormal, (1, 0, 2))

        # Positional encoding
        x_normal = self.positional_encoding(x_normal)
        x_abnormal = self.positional_encoding(x_abnormal)

        # Form the tokens for Transformer encoder
        # x.shape = (F, B, C*E) --> (F+1, B, C*E)
        x_normal = torch.permute(x_normal, (1, 0, 2))
        x_normal = torch.stack([torch.vstack((self.class_token, xi)) for xi in x_normal])
        x_normal = torch.permute(x_normal, (1, 0, 2))
        x_abnormal = torch.permute(x_abnormal, (1, 0, 2))
        x_abnormal = torch.stack([torch.vstack((self.class_token, xi)) for xi in x_abnormal])
        x_abnormal = torch.permute(x_abnormal, (1, 0, 2))

        # Transformer encoder
        # x.shape = (F+1, B, C*E) --> (B, C*E)
        x_normal = self.transformer_encoder(x_normal)[0]
        x_abnormal = self.transformer_encoder(x_abnormal)[0]
        normal_features = x_normal
        abnormal_features = x_abnormal

        # MLP
        # x.shape = (B, C*E) --> (B, C, E) --> (B, C, 1)
        x_normal = x_normal.view(-1, C, E)
        x_abnormal = x_abnormal.view(-1, C, E)
        x_normal = self.fc1(x_normal)
        x_normal = x_normal + self.fc2(x_normal)  # residual learning
        normal_scores = self.fc3(x_normal)
        normal_scores = self.sigmoid(normal_scores)

        x_abnormal = self.fc1(x_abnormal)
        x_abnormal = x_abnormal + self.fc2(x_abnormal)  # residual learning
        abnormal_scores = self.fc3(x_abnormal)
        abnormal_scores = self.sigmoid(abnormal_scores)

        # features.shape: (B, C*E) --> (B, C, E)
        normal_features = normal_features.view(-1, C, E)
        abnormal_features = abnormal_features.view(-1, C, E)

        # Take embeddings as features
        # feat_magnitudes.shape: (B, C, E) --> (B, C)
        nfea_magnitudes = torch.norm(normal_features, p=2, dim=2)  # normal feature magnitudes
        # nfea_magnitudes = nfea_magnitudes.view(B, C, -1).mean(1)
        afea_magnitudes = torch.norm(abnormal_features, p=2, dim=2)  # abnormal feature magnitudes
        # feat_magnitudes = feat_magnitudes.view(B, C, -1).mean(1)
        # n_size = nfea_magnitudes.shape[0]

        select_idx = torch.ones_like(afea_magnitudes)
        # select_idx = self.dropout(select_idx)
        select_idx_normal = torch.ones_like(nfea_magnitudes)
        # select_idx_normal = self.dropout(select_idx_normal)

        #######  process abnormal videos -> select top3 feature magnitude  #######
        # afea_magnitudes_drop: (B, C)
        # idx_abn: (B, k_abn)
        # idx_abn_feat: (B, k_abn, E)
        afea_magnitudes_drop = afea_magnitudes * select_idx
        idx_abn = torch.topk(afea_magnitudes_drop, self.k_abn, dim=1)[1]
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, self.k_nor, dim=1)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        # total_select_abn_feature = torch.zeros(0, device=self.device)
        # feat_select_abn: (B, k_abn, E)
        feat_select_abn = torch.gather(abnormal_features, dim=1, index=idx_abn_feat)
        feat_select_normal = torch.gather(normal_features, dim=1, index=idx_normal_feat)
        """
        for abnormal_feature in abnormal_features:
            # top 3 features magnitude in abnormal bag
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))
        """

        # idx_abn_score: (B, k_abn, 1)
        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        # top 3 scores in abnormal bag based on the top-3 magnitude
        # (B, C, 1) --> (B, k_abn, 1) --> (B, 1)
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)  # top 3 scores in normal bag

        ####### process normal videos -> select top3 feature magnitude #######
        """
        for nor_fea in normal_features:
            # top 3 features magnitude in normal bag (hard negative)
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))
        """

        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, abnormal_scores

    def forward_testing(self, x_in):
        B, F, _ = x_in.size()
        C = CLIP_LEN
        E = EMBEDDING_DIM
        # x.shape = (B, F, C*E) --> (F, B, C*E)
        x_in = torch.permute(x_in, (1, 0, 2))

        # Positional encoding
        x_in = self.positional_encoding(x_in)

        # Form the tokens for Transformer encoder
        # x.shape = (F, B, C*E) --> (F+1, B, C*E)
        x_in = torch.permute(x_in, (1, 0, 2))
        x_in = torch.stack([torch.vstack((self.class_token, xi)) for xi in x_in])
        x_in = torch.permute(x_in, (1, 0, 2))

        # Transformer encoder
        # x.shape = (F+1, B, C*E) --> (B, C*E)
        x_in = self.transformer_encoder(x_in)[0]

        # MLP
        # x.shape = (B, C*E) --> (B, C, E) --> (B, C, 1)
        # scores: (B, C, 1) --> (B, C)
        x_in = x_in.view(B, C, E)
        x_features = x_in
        x_in = self.fc1(x_in)
        x_in = x_in + self.fc2(x_in)  # residual learning
        scores = self.fc3(x_in)
        scores = torch.squeeze(scores, dim=2)

        # x_feature_magnitudes: (B, C)
        x_feature_magnitudes = torch.norm(x_features, p=2, dim=2)  # normal feature magnitudes
        x_select = torch.ones_like(x_feature_magnitudes)
        # x_select = self.dropout(x_select)

        # x_idx: (B, k_abn)
        # x_idx_feat: (B, k_abn, E)
        x_feature_magnitudes = x_feature_magnitudes * x_select
        score_idx = torch.topk(x_feature_magnitudes, self.k_abn, dim=1)[1]
        # x_idx_feat = x_idx.unsqueeze(2).expand([-1, -1, x_features.shape[2]])

        # idx_score = idx_normal.unsqueeze(2).expand([-1, -1, scores.shape[2]])
        # scores_topk: (B, k_abn)
        # logit: (B,)
        scores_topk = torch.gather(scores, 1, score_idx)
        logit = torch.mean(scores_topk, dim=1)

        return logit


class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha=0.0001, margin=100):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.criterion = nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal

        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()

        label = label.cuda()

        loss_cls = self.criterion(score, label)  # BCE loss in the score space

        # Average selected feature magnitudes' 2nd-norm
        feat_a = torch.mean(feat_a, dim=1)
        feat_a = torch.norm(feat_a, p=2, dim=1)

        # Batch normalization (Because hard margin isn't reasonable)
        feat_a_max = torch.max(feat_a)
        feat_a_diff = feat_a_max - feat_a
        # feat_a_diff = self.margin - feat_a
        loss_abn = torch.abs(feat_a_diff)
        # print(feat_a_diff)

        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)

        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)

        loss_total = loss_cls + self.alpha * loss_rtfm

        return loss_total
