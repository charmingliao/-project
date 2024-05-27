import socket
import torch
import io
from model import Net, train, test
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import argparse
from loguru import logger
import os
import sys
import dill
CLIENT_DATA_PATH = "C:/Users/22712/OneDrive - sjtu.edu.cn/桌面/工科创4-I/-project/FL_Data/Data_CIFAR10/"


def get_logger(output_file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    while os.path.exists(output_file):
        output_file = output_file.replace('.log', '_1.log')
    if output_file:
        logger.add(output_file, enqueue=True, format=log_format)
    return logger
with open(os.path.join(CLIENT_DATA_PATH, "Test.pkl"), "rb") as f:
    test_dataset_server = dill.load(f)
    test_loader = DataLoader(test_dataset_server, batch_size=32,shuffle=False)
    
    
parser = argparse.ArgumentParser()
parser.add_argument("num_clients", type=int, help="The number of clients.")
parser.add_argument("num_rounds", type=int, help="The number of training rounds.")
parser.add_argument("num_epochs", type=int, help="The number of epochs for each training round.")
parser.add_argument("send_port", type=int, help="The port which client send the models.")
parser.add_argument("receive_port", type=int, help="The port which client receive the models.")
args = parser.parse_args()
# 全局模型
global_model = Net()

# zero the parameter of global model
for param in global_model.parameters():
    param.data = torch.zeros_like(param.data)

# 定义客户端数量
num_clients = args.num_clients
num_rounds = args.num_rounds
num_epochs = args.num_epochs

logger = get_logger(f'./log_file/result_{num_clients}_{num_rounds}_{num_epochs}.log')
logger.info(args)

def handle_client(connection):
    try:
        # 接收数据
        data = b''
        while True:
            packet = connection.recv(4096)
            if not packet: 
                break
            data += packet

        # 从二进制数据中加载模型参数
        buffer = io.BytesIO(data)
        buffer.seek(0)
        params = torch.load(buffer)

        # 载入模型参数
        model = Net()
        model.load_state_dict(params)

        # 聚合模型参数
        for param_global, param_client in zip(global_model.parameters(), model.parameters()):
            param_global.data += param_client.data
    finally:
        # 清理连接
        connection.close()

def receive_models():
    # 创建一个TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # 绑定socket到端口
    server_address = ('localhost', args.receive_port)
    print('starting up on {} port {}'.format(*server_address))
    sock.bind(server_address)
    # 监听连接
    sock.listen(num_clients)
    for _ in range(num_clients):
        # 等待连接
        print('waiting for a connection')
        connection, client_address = sock.accept()
        print('connection from', client_address)

        # 处理连接
        handle_client(connection)
    # 关闭socket
    sock.close()

def send_models(client_id):
    # 创建一个TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # 绑定socket到端口
    server_address = ('localhost', args.send_port)
    print('starting up on {} port {} to send model'.format(*server_address))
    sock.bind(server_address)
    # 监听连接
    sock.listen(num_clients)
    for idx in range(num_clients):
        # 等待连接
        print('waiting for a connection for send global model')
        connection, client_address = sock.accept()
        print('connection from for global model', client_address)

        # 将全局模型参数发送给客户端
        buffer = io.BytesIO()
        model_params = {
            'client_id': client_id[idx], # assign client id
            'model_state_dict': global_model.state_dict()
        }
        torch.save(model_params, buffer)
        buffer.seek(0)
        connection.sendall(buffer.getvalue())

        # 清理连接
        connection.close()
    # 关闭socket
    sock.close()

for idx in range(num_rounds):
    # Receive models
    receive_models()

    # 计算平均模型参数
    for param_global in global_model.parameters():
        param_global.data /= num_clients
    # test the global model
    acc=test(global_model, test_loader)
    logger.info(f'Round {idx+1} acc: {acc:.4f}')

    if num_clients!=20:
        client_id = np.random.choice(20, num_clients, replace=False)
    else:
        client_id = np.arange(20)

    # Send models
    send_models(client_id)
