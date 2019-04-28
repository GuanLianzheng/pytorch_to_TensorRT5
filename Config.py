# -*- coding: utf-8 -*-
# 配置参数
class Config:
    def __init__(self):
        self.BATCH_SIZE = 64
        self.TEST_BATCH_SIZE = 1000
        self.EPOCHS = 3
        self.LEARNING_RATE = 0.001
        self.SGD_MOMENTUM = 0.5
        self.SEED = 1
        self.LOG_INTERVAL = 10