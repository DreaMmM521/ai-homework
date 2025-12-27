"""实验配置文件"""

class Config:
    # 数据集配置
    DATA_ROOT = './data'
    DATASET_NAME = 'MNIST'
    BATCH_SIZE = 64
    TEST_BATCH_SIZE = 1000
    
    # 模型配置
    INPUT_SIZE = 784  # 28x28
    HIDDEN_SIZES = [128, 64]
    OUTPUT_SIZE = 10
    
    # 训练配置
    EPOCHS = 20
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    # 实验配置
    ACTIVATIONS = {
        'ReLU': 'torch.nn.ReLU',
        'Sigmoid': 'torch.nn.Sigmoid', 
        'Tanh': 'torch.nn.Tanh',
        'LeakyReLU': 'torch.nn.LeakyReLU'
    }
    
    # LeakyReLU参数
    LEAKY_RELU_NEGATIVE_SLOPE = 0.01
    
    # 随机种子
    SEED = 42
    
    # 路径配置
    MODEL_SAVE_DIR = './models'
    RESULT_SAVE_DIR = './results'
    PLOT_SAVE_DIR = './plots'
    
    # 实验重复次数（减少随机性）
    NUM_REPEATS = 3