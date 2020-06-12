def set_global_seeds(seed:int):
    import numpy as np
    import torch
    import gym
    torch.manual_seed(seed)
    np.random.seed(seed)

