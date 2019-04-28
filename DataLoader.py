# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms
from Config import Config

# 数据加载
class DataLoaders:

    def __init__(self):
        self.config = Config()
        self.kwargs = {'num_workers': 1, 'pin_memory': True}

    def trainDataLoader(self):
        '''
        加载训练集
        :return: 训练集
        '''
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/tmp/mnist/data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            **self.kwargs)
        return train_loader

    def testDataLoader(self):
        '''
        加载测试集
        :return: 测试集
        '''
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/tmp/mnist/data', train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.config.TEST_BATCH_SIZE,
            shuffle=True,
            **self.kwargs)
        return test_loader

if __name__ == '__main__':
    test = DataLoaders()
    print(test.trainDataLoader())
    print(test.testDataLoader())