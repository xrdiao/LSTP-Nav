import numpy as np


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.n = 0
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.std = np.ones(shape)
        self.epsilon = epsilon

    def update(self, x):
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        
        delta = batch_mean - self.mean
        total_count = self.n + batch_count
        
        # 使用Welford算法进行更稳定的更新
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.n
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.n * batch_count / total_count
        self.var = M2 / total_count
        self.std = np.sqrt(self.var + self.epsilon)
        self.n = total_count


class Normalization:
    def __init__(self, shape, clip=10.0):
        self.running_ms = RunningMeanStd(shape=shape)
        self.clip = clip
        
    def __call__(self, x, update=True):
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        if self.clip is not None:
            x = np.clip(x, -self.clip, self.clip)
        return x


class RewardScaling:
    def __init__(self, shape, gamma, clip=10.0):
        self.shape = shape
        self.gamma = gamma
        self.clip = clip
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)
        
    def __call__(self, x):
        self.R = self.gamma * self.R + np.array(x)
        self.running_ms.update(self.R)
        # 同时减去均值和除以标准差
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        if self.clip is not None:
            x = np.clip(x, -self.clip, self.clip)
        return x
        
    def reset(self, i=None):
        if i is None:
            self.R = np.zeros(self.shape)
        else:
            self.R[i] = 0
