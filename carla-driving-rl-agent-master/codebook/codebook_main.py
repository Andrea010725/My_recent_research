# edited by Zhiwen 10.20
from collections import OrderedDict
from typing import Tuple, Dict, Optional, List, Sequence
from typing import TypeVar

import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from gym.spaces.dict import Dict as SpaceDict
from omegaconf import DictConfig

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
    DistributionType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.aux_losses.losses import MultiAuxTaskNegEntropyLoss
from allenact.embodiedai.models.aux_models import AuxiliaryModel
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from allenact.embodiedai.models.fusion_models import Fusion
from allenact.utils.model_utils import FeatureEmbedding
from allenact.utils.system import get_logger


import sys
sys.path.append('C:\Users\19501\Downloads\carla-driving-rl-agent-master\carla-driving-rl-agent-master\codebook')
from SparseMax import Sparsemax


class RunningMeanStd(object):      # 计算偏差值  保留
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count


#   Codebook 初始化
class Codebook():

    # action_space: gym.spaces.Discrete

    def __init__(self ):
       
        # codebook
        self.codebook_type = cfg.model.codebook.type
        self.codebook_size = cfg.model.codebook.size
        self.code_dim = cfg.model.codebook.code_dim
        self.codebook.embeds = "joint_embeds"   # 可修改   "beliefs"

        # 目前只考虑 codebook 是随机初始化的 
        # self.codebook.initialization == "random":
        self.codebook = torch.nn.Parameter(torch.randn(self.codebook_size, self.code_dim))
        self.codebook.requires_grad = True

        # if self.codebook_indexing == "sparsemax":
        # 使用Sparsemax作为激活函数，用于生成稀疏的概率分布
        self.sparsemax = Sparsemax(dim=-1)

        # dropout to prevent codebook collapse
        # self.dropout_prob = cfg.model.codebook.dropout  
        # 设置dropout概率为0.1  可以修改
        self.dropout_prob =  0.1
        self.dropout = nn.Dropout(self.dropout_prob)

        # codebook indexing
        if self.codebook.embeds == "joint_embeds":
            self.linear_codebook_indexer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(1574, self.codebook_size),
            )
            self.linear_upsample = nn.Sequential(
                nn.Linear(self.code_dim, 1574),
            )
        elif self.codebook.embeds == "beliefs":
            self.linear_codebook_indexer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self._hidden_size, self.codebook_size),
            )
            self.linear_upsample = nn.Sequential(
                nn.Linear(self.code_dim, self._hidden_size),
            )

        # running mean and std
        self.rms = RunningMeanStd()



# 实现了 codebook 的嵌入机制。codebook 是一个可训练的参数矩阵，包含多个嵌入向量。这些向量用于将输入的奖励权重映射到一个紧凑的表示空间。
# input_dim 输入目标 设置为3  
class CodebookEmbedder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=20, codebook_dim=30, embedding_dim=48):
        super(CodebookEmbedder, self).__init__()
        
        # The encoder consists of two linear layers separated by a ReLU activation.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, codebook_dim)
        )
        
        # The codebook contains 30 different vectors, each of 48 dimensions.
        self.codebook = nn.Parameter(torch.randn(codebook_dim, embedding_dim))

    def forward(self, x):
        # Transform the input to a distribution over the codebook vectors
        # x shape: W x B x num_objectives
        x_shape = x.shape
        x = x.view(-1, x_shape[-1]).type(torch.float32)
        # 通过编码器将输入的奖励权重转换为概率分布
        logits = self.encoder(x)   
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Use the distribution to get the codebook embeddings
        embeddings = torch.matmul(probs, self.codebook)
        embeddings = embeddings.view(x_shape[0], x_shape[1], -1)
        print("Reward Embedding ：",embeddings)
        return embeddings
