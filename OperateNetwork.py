# -*- coding: utf-8 -*-
import torch
from Config import Config
from DataLoader import DataLoaders
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from Network import Net

class OperateNetwork:

    def __init__(self):
        self.config = Config()
        self.data = DataLoaders()

    def train(self, model):
        '''
        训练模型
        :param model: 神经网络模型
        '''
        optimizer = optim.SGD(model.parameters(), lr=self.config.LEARNING_RATE, momentum=self.config.SGD_MOMENTUM)
        model.train()
        for batch, (data, target) in enumerate(self.data.trainDataLoader()):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    def test(self, model):
        '''
        测试正确率
        :param model: 神经网络模型
        '''
        test_loader = self.data.testDataLoader()
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
        test_loss /= len(test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss,
                      correct,
                      len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))

    def saveModel(self, model, filepath):
        '''
        保存神经网络权重参数
        :param model: 神经网络模型
        :param filepath: 权重文件路径
        '''
        torch.save(model.state_dict(), filepath)

    def saveONNX(self, model, filepath):
        '''
        保存ONNX模型
        :param model: 神经网络模型
        :param filepath: 文件保存路径
        '''

        # 神经网络输入数据类型
        dummy_input = torch.randn(self.config.BATCH_SIZE, 1, 28, 28, device='cuda')
        torch.onnx.export(model, dummy_input, filepath, verbose=True)

    def loadModel(self, filepath):
        '''
        加载权重参数
        :param filepath:
        :return:
        '''
        model = Net()
        model.cuda()
        model.load_state_dict(torch.load(filepath))
        return model

    def main(self):
        # torch.cuda.manual_seed(self.config.SEED)
        model = Net()
        # 利用GPU训练
        model.cuda()
        # 训练三世代
        for e in range(self.config.EPOCHS):
            self.train(model)
            self.test(model)
        self.saveModel(model, 'lianzheng_mnist.pth')
        self.saveONNX(model, 'lianzheng_mnist.onnx')
        return model

if __name__ == '__main__':
    op = OperateNetwork()
    op.main()
