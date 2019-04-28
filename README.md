# pytorch_to_TensorRT5
通过pytorch搭建卷积神经网络完成手写识别任务，并将训练好的模型以多种方式部署到TensorRT中加速。

## 文件描述：
Config.py：卷积神经网络配置参数
DataLoader.py：读取训练集与测试集
Network.py：卷积神经网络结构
OperateNetwork.py：对卷积神经网络的操作（训练，测试，保存读取权重，保存onnx）
TensorRTNet.py：三种方式创建引擎
main.py：主函数

## 运行方式：
'''
python main.py
'''

## 注意：
(1)对于多输入模型保存onnx的方式：
'''
dummy_input0 = torch.FloatTensor(Batch_size, seg_length).to(torch.device("cuda"))  
dummy_input1 = torch.FloatTensor(Batch_size, seg_length).to(torch.device("cuda"))  
dummy_input2 = torch.FloatTensor(Batch_size, seg_length).to(torch.device("cuda"))  
torch.onnx.export(model. (dummy_input0, dummy_input1, dummy_input2), filepath)  
'''
(2)TensorRT不支持int64，float64,因此模型不应该包含这两种数据类型的运算。
