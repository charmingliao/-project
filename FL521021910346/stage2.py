import dill
import torch
from model import Net
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
import sys
import numpy as np
from tqdm import tqdm

MODEL_PATH = "./models/"
CLIENT_MODEL_PATH = os.path.join(MODEL_PATH, "client_models")
GLOBAL_MODEL_PATH = os.path.join(MODEL_PATH, "server_model.pth")
CLIENT_DATA_PATH = "C:/Users/22712/OneDrive - sjtu.edu.cn/桌面/工科创4-I/-project/FL_Data/Data_CIFAR10/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clients_num = 20
pkl_path = "C:/Users/22712/OneDrive - sjtu.edu.cn/桌面/工科创4-I/-project/FL_Data/Data_CIFAR10/Client1.pkl"

with open(pkl_path, 'rb') as f:
    train_dataset_client_1 = dill.load(f)

l = len(train_dataset_client_1)
print(l)
# 2500

def load_data():
    train_loader_list = []
    for i in range(clients_num):
        pkl_path = os.path.join(CLIENT_DATA_PATH, f"Client{i+1}.pkl")
        with open(pkl_path, 'rb') as f:
            train_dataset_client = dill.load(f)
            dataloader = DataLoader(train_dataset_client,batch_size=32, shuffle=True, drop_last=True)
            train_loader_list.append(dataloader)
    with open(os.path.join(CLIENT_DATA_PATH, "Test.pkl"), "rb") as f:
        test_dataset_server = dill.load(f)
        test_loader = DataLoader(test_dataset_server, batch_size=32,shuffle=False)
        
    return train_loader_list, test_loader
    
    
# 保存日志
def get_logger(output_file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    while os.path.exists(output_file):
        output_file = output_file.replace('.log', '_1.log')
    if output_file:
        logger.add(output_file, enqueue=True, format=log_format)
    return logger
logger = get_logger(f'./log_file/result_stage2.log')

def train(num_rounds, num_epoches, M = 16):
    # load data
    train_loader_list, test_loader = load_data()
    print("mark")
    # initialize global model parameters
    global_model = Net().to(device)
    global_model_path = GLOBAL_MODEL_PATH
    torch.save(global_model.state_dict(), global_model_path)
    
    for round in range(10):
    # train M 个clients
        idx = np.random.choice(len(train_loader_list), M)
        for client in tqdm(idx):
            
            # 将服务端全局模型发给各个客户端
            local_model = Net().to(device)
            checkpoint = torch.load(global_model_path)
            local_model.load_state_dict(checkpoint)
            # 加载客户端的数据集
            dataloader = train_loader_list[client]
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(local_model.parameters(), lr=1e-3)
            for i in range(num_epoches):
                print(f"Client{client+1},begin to train, epoch {i+1}")
                logger.info(f"Client{client+1},begin to train, epoch {i+1}")

                for feature, label in dataloader:
                    feature, label = feature.to(device), label.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(feature)
                    loss = criterion(outputs, label)
                    loss.backward()
                    optimizer.step()
            # 保存客户端本地模型
            save_path = CLIENT_MODEL_PATH + f"/model{client+1}.pth"
            torch.save(local_model.state_dict(), save_path)
            
            # 测一下本客户端的正确率
            accuracy = test(local_model, dataloader)
            print(accuracy)
            logger.info(f"accuracy: {accuracy}")

        
        avg_model = Net().to(device)
        avg_model.load_state_dict(torch.load(global_model_path))
        
        for i in range(M):
            client_model_path = CLIENT_MODEL_PATH + f"/model{i+1}.pth"
            client_model = Net().to(device)
            client_model.load_state_dict(torch.load(client_model_path))
            for avg_param, client_param in zip(avg_model.parameters(), client_model.parameters()):
                avg_param.data += client_param.data
        
        for avg_param in avg_model.parameters():
            avg_param.data /= M
        global_model.load_state_dict(avg_model.state_dict())

        accuracy = test(global_model, test_loader)
        print(accuracy)
        
        logger.info(f"Round: {round+1}, Accuracy: {accuracy}")

        torch.save(global_model.state_dict(),global_model_path)
        
    
def test(model, test_loader):
    model.eval()
    test_loss = 0
    n_total = 0
    n_correct = 0
    
    with torch.no_grad():
        for feature, labels in test_loader:
            feature, labels = feature.to(device), labels.to(device)
            output = model(feature)
            
            for i, out in enumerate(output):
                #print(torch.argmax(out))
                #print(labels[i])
                if torch.argmax(out) == labels[i]:
                    n_correct += 1
                    #print("suc")
                n_total += 1
    return n_correct / n_total
    
def main():

    train(1,20)


if __name__ == "__main__":
    main()