import os
# Load previous result: best/latest model/score
LOAD_PREVIOUS_RESULT = False
RESULT_ROOT_DIR = "F:\\Audio_transformer\\Results"
CHECK_POINTS_DIR = os.path.join(RESULT_ROOT_DIR, "Check_points/")
SCORES_DIR = os.path.join(RESULT_ROOT_DIR, "Scores/")
RECORDS_DIR = os.path.join(RESULT_ROOT_DIR, "Records/")

# Dataset config
DATASET_ROOT_DIR = "F:\\XD_Violence\\Embeddings"
USE_ZERO_PADDING = True

# Model config
USE_RTFM = True
CLIP_LEN = 32
EMBEDDING_DIM = 128
MODEL_DIM = CLIP_LEN * EMBEDDING_DIM    # C*E
HIDDEN_DIM = EMBEDDING_DIM // 4 if USE_RTFM else MODEL_DIM // 16
OUTPUT_DIM = 1
NUM_HEADS = 4
NUM_LAYERS = 2
DROP_OUT = 0.5

# RTFM
K_ABN = 3
K_NOR = 3

# Hyperparameters
LR = 1e-5
EPOCHS = 10
# To avoid dimension problem on dataloader, we need to keep the BATCH_SIZE to 1
BATCH_SIZE = 16
