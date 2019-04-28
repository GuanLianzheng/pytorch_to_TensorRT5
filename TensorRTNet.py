# -*- coding: utf-8 -*-
import tensorrt as trt
import torch
import pycuda.driver as cuda
import pycuda.autoinit # 非常重要
import numpy as np
from OperateNetwork import OperateNetwork
from time import time
from DataLoader import DataLoaders


class TensorRTNet:

    def loadModelStateDict(self, filepath):
        '''
        加载权重文件
        :param filepath: 权重文件路径
        :return: 权重参数
        '''
        op = OperateNetwork()
        model = op.loadModel(filepath)
        return model.state_dict()

    def tensorRTNet(self, network, weights):
        '''
        根据模型构建TensorRT网络结构
        :param network: 初始化网络结构对象
        :param weights: 权重参数
        :return: network
        '''
        data = network.add_input("data", trt.float32, (1, 28, 28))
        assert (data)

        # -------------
        conv1_w = weights['conv1.weight'].cpu().numpy().reshape(-1)
        conv1_b = weights['conv1.bias'].cpu().numpy().reshape(-1)
        conv1 = network.add_convolution(data, 20, (5, 5), conv1_w, conv1_b)
        assert (conv1)
        conv1.stride = (1, 1)

        # -------------
        pool1 = network.add_pooling(conv1.get_output(0), trt.PoolingType.MAX, (2, 2))
        assert (pool1)
        pool1.stride = (2, 2)

        # -------------
        conv2_w = weights['conv2.weight'].cpu().numpy().reshape(-1)
        conv2_b = weights['conv2.bias'].cpu().numpy().reshape(-1)
        conv2 = network.add_convolution(pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b)
        assert (conv2)
        conv2.stride = (1, 1)

        # -------------
        pool2 = network.add_pooling(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
        assert (pool2)
        pool2.stride = (2, 2)

        # -------------
        fc1_w = weights['fc1.weight'].cpu().numpy().reshape(-1)
        fc1_b = weights['fc1.bias'].cpu().numpy().reshape(-1)
        fc1 = network.add_fully_connected(pool2.get_output(0), 500, fc1_w, fc1_b)
        assert (fc1)

        # -------------
        relu1 = network.add_activation(fc1.get_output(0), trt.ActivationType.RELU)
        assert (relu1)

        # -------------
        fc2_w = weights['fc2.weight'].cpu().numpy().reshape(-1)
        fc2_b = weights['fc2.bias'].cpu().numpy().reshape(-1)
        fc2 = network.add_fully_connected(relu1.get_output(0), 10, fc2_w, fc2_b)
        assert (fc2)

        fc2.get_output(0).name = "prob"
        network.mark_output(fc2.get_output(0))

        return network

    def toTensorRT(self, filepath):
        '''
        构建TensorRT运行引擎，并保存计划文件
        :param filepath: 权重参数文件路径
        '''

        # 读取训练好的模型的
        weights = self.loadModelStateDict(filepath)

        # 打印日志
        G_LOGGER = trt.Logger(trt.Logger.WARNING)

        # 创建Builder
        builder = trt.Builder(G_LOGGER)

        # 根据模型创建TensorRT的网络结构
        network = builder.create_network()

        network = self.tensorRTNet(network, weights)

        builder.max_batch_size = 100
        builder.max_workspace_size = 1 << 20

        # 创建引擎
        engine = builder.build_cuda_engine(network)
        del network
        del builder

        runtime = trt.Runtime(G_LOGGER)

        # 读取测试集
        datas = DataLoaders()
        test_loader = datas.testDataLoader()
        img, target = next(iter(test_loader))
        img = img.numpy()
        target = target.numpy()

        img = img.ravel()

        context = engine.create_execution_context()
        output = np.empty((100, 10), dtype=np.float32)

        # 分配内存
        d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
        d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
        bindings = [int(d_input), int(d_output)]

        # pycuda操作缓冲区
        stream = cuda.Stream()
        # 将输入数据放入device
        cuda.memcpy_htod_async(d_input, img, stream)
        # 执行模型
        context.execute_async(100, bindings, stream.handle, None)
        # 将预测结果从从缓冲区取出
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # 线程同步
        stream.synchronize()

        print("Test Case: " + str(target))
        print("Prediction: " + str(np.argmax(output, axis=1)))

        # 保存计划文件
        # with open("lianzheng.engine", "wb") as f:
        #     f.write(engine.serialize())

        del context
        del engine
        del runtime

    def loadEngine2TensorRT(self, filepath):
        '''
        通过加载计划文件，构建TensorRT运行引擎
        :param filepath: 计划文件路径
        '''

        # 打印日志
        G_LOGGER = trt.Logger(trt.Logger.WARNING)

        # 反序列化引擎
        with open("lianzheng.engine", "rb") as f, trt.Runtime(G_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        # 读取测试集
        datas = DataLoaders()
        test_loader = datas.testDataLoader()
        img, target = next(iter(test_loader))
        img = img.numpy()
        target = target.numpy()

        img = img.ravel()

        context = engine.create_execution_context()
        output = np.empty((100, 10), dtype=np.float32)

        # 分配内存
        d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
        d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
        bindings = [int(d_input), int(d_output)]

        # pycuda操作缓冲区
        stream = cuda.Stream()
        # 将输入数据放入device
        cuda.memcpy_htod_async(d_input, img, stream)
        # 执行模型
        context.execute_async(100, bindings, stream.handle, None)
        # 将预测结果从从缓冲区取出
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # 线程同步
        stream.synchronize()

        print("Test Case: " + str(target))
        print("Prediction: " + str(np.argmax(output, axis=1)))

        del context
        del engine

    def ONNX_build_engine(self, onnx_file_path):
        '''
        通过加载onnx文件，构建engine
        :param onnx_file_path: onnx文件路径
        :return: engine
        '''
        # 打印日志
        G_LOGGER = trt.Logger(trt.Logger.WARNING)

        with trt.Builder(G_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, G_LOGGER) as parser:
            builder.max_batch_size = 100
            builder.max_workspace_size = 1 << 20

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')

            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")

            # 保存计划文件
            # with open(engine_file_path, "wb") as f:
            #     f.write(engine.serialize())
            return engine

    def loadONNX2TensorRT(self, filepath):
        '''
        通过onnx文件，构建TensorRT运行引擎
        :param filepath: onnx文件路径
        '''

        engine = self.ONNX_build_engine(filepath)

        # 读取测试集
        datas = DataLoaders()
        test_loader = datas.testDataLoader()
        img, target = next(iter(test_loader))
        img = img.numpy()
        target = target.numpy()

        img = img.ravel()

        context = engine.create_execution_context()
        output = np.empty((100, 10), dtype=np.float32)

        # 分配内存
        d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
        d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
        bindings = [int(d_input), int(d_output)]

        # pycuda操作缓冲区
        stream = cuda.Stream()
        # 将输入数据放入device
        cuda.memcpy_htod_async(d_input, img, stream)
        # 执行模型
        context.execute_async(100, bindings, stream.handle, None)
        # 将预测结果从从缓冲区取出
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # 线程同步
        stream.synchronize()

        print("Test Case: " + str(target))
        print("Prediction 100: " + str(np.argmax(output, axis=1)))

        del context
        del engine

