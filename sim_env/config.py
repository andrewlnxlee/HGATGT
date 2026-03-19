# config.py

# --- 数据生成配置 ---
DATA_ROOT = "./data"
XISHU = 0.1
NUM_TRAIN_SAMPLES = 2000  # 训练集样本数 (每个样本是一个几十帧的片段)
NUM_VAL_SAMPLES = 200     # 验证集
NUM_TEST_SAMPLES = 100    # 测试集
FRAMES_PER_SAMPLE = 50    # 每个样本包含多少帧 (短片段利于训练)
MAX_GROUPS = 5            # 场景中最大群数量

# --- 模型配置 ---
# INPUT_DIM = 4     # x, y, vx, vy
# EDGE_DIM = 6      # dx, dy, dist, dvx, dvy, cos_sim
INPUT_DIM = 2       # x, y 原先
EDGE_DIM = 3        # dx, dy, dist
HIDDEN_DIM = 64

# --- 训练配置 ---
BATCH_SIZE = 1      # 图网络建议 Batch=1 (指一次处理一个时序图序列) 或使用 PyG Batch
LEARNING_RATE = 0.0005
EPOCHS = 50
MODEL_VERSION = "dual_head_v1"
MODEL_SAVE_PATH = f"sim_env/model/best_model_{MODEL_VERSION}.pth"
MODEL_USE_PATH = MODEL_SAVE_PATH
DEVICE = "cuda"    # 或 "cpu"

LAMBDA_GROUP = 0.75
LAMBDA_POINT = 1.25
LAMBDA_TEMP = 0.2

# --- 点级评估 / 后处理配置 ---
ENABLE_POINT_UNCERTAINTY_GATING = True
ENABLE_POINT_UNCERTAINTY_ABLATION = False
POINT_UNCERTAINTY_GATE_SCALE = 1.0
POINT_CLUSTER_EPS = 35
POINT_CLUSTER_MIN_SAMPLES = 3
POINT_TRACK_STAGE1_THRESHOLD = 14.0
POINT_TRACK_RECOVERY_THRESHOLD = 22.0
POINT_TRACK_MAX_AGE = 20
POINT_MATCH_THRESHOLD = 28.0
GROUP_TO_POINT_ASSOC_THRESH = 28.0
GROUP_TO_CENTROID_THRESH = 20.0
GROUP_POINT_MAX_AGE = 4
ENABLE_MEAS_DIAGNOSTIC = True

# --- 输出配置 ---
OUTPUT_TEST_DIR = "sim_env/output/test"
OUTPUT_GIF_DIR = "sim_env/output/track_result_gif"


# ---------------------- EWAP数据集 ----------------------
# -------------------------------------------------------

# --- 模型配置 ---
EWAP_MODEL_SAVE_PATH = "ewap_env/model/best_model_ewap_v2.pth"
EWAP_MODEL_USE_PATH = "ewap_env/model/best_model_ewap_v1.pth"

# --- 坐标转换 (用于 ETH/Hotel 数据集) ---
COORD_SCALE = 50.0        # 缩放倍数
COORD_OFFSET = [500.0, 500.0]  # 中心偏移 (x, y)

# --- 输出配置 ---
OUTPUT_MP4_DIR = "ewap_env/output/track_result_mp4"
