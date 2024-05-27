import socket
import torch
import argparse
from model import Net,train  # Replace with your model and functions
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import io
import time
import os 
import dill

MODEL_PATH = "./models/"
CLIENT_MODEL_PATH = os.path.join(MODEL_PATH, "client_models")
GLOBAL_MODEL_PATH = os.path.join(MODEL_PATH, "server_model.pth")
CLIENT_DATA_PATH = "C:/Users/22712/OneDrive - sjtu.edu.cn/桌面/工科创4-I/-project/FL_Data/Data_CIFAR10/"
def loaddata():
    train_loader_list = []
    for i in range(20):
        pkl_path = os.path.join(CLIENT_DATA_PATH, f"Client{i+1}.pkl")
        with open(pkl_path, 'rb') as f:
            train_dataset_client = dill.load(f)
            dataloader = DataLoader(train_dataset_client,batch_size=32, shuffle=True, drop_last=True)
            train_loader_list.append(dataloader)
    
    return train_loader_list
    

def federated_train_and_send(train_loader_list, client_id, num_rounds, num_epochs, lr, server_ip, receive_port, send_port):
    # Initialize the model
    model = Net()
    print("Client {} initialized the model.".format(client_id+1))
    for r in range(num_rounds+1):
        # Load the global model parameters
        global_params = None
        if r > 0:
            client_id, global_params = receive_global_params(server_ip, receive_port)
            model.load_state_dict(global_params)
            print("Client {} received global model parameters.".format(client_id+1))

            if r == num_rounds:
                break
        
        
        dataloader = train_loader_list[client_id]

        # Train the model for num_epochs
        train(model, dataloader, num_epochs, lr)
        print("Client {} finished training round {}.".format(client_id+1, r+1))
        # Send the model parameters to the server
        send_params_to_server(model, server_ip, send_port)

def send_params_to_server(model, server_ip, server_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            s.connect((server_ip, server_port))
            break
        except ConnectionRefusedError:
            print("Connection refused. Retrying...")
            time.sleep(1)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    s.sendall(buffer.getvalue())
    s.close()

def receive_global_params(server_ip, server_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            s.connect((server_ip, server_port))
            break
        except ConnectionRefusedError:
            print("Connection refused. Retrying...")
            time.sleep(1)
    params_bytes = b''
    while True:
        packet = s.recv(4096)
        if not packet:
            break
        params_bytes += packet
    buffer = io.BytesIO(params_bytes)
    buffer.seek(0)
    model_params = torch.load(buffer)
    s.close()
    client_id = model_params['client_id']
    params = model_params['model_state_dict']
    return client_id, params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("client_id", help="The ID of the client.")
    parser.add_argument("num_rounds", help="The number of training rounds.")
    parser.add_argument("num_epochs", help="The number of epochs for each training round.")
    parser.add_argument("lr", help="Learning rate for SGD optimizer.")
    parser.add_argument("receive_port", help="The port of the server for receiving global model.")
    parser.add_argument("send_port", help="The port of the server for sending local model.")
    args = parser.parse_args()

    # set seed for reproducibility
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    # Load the dataset
    batch_size = 32
    train_loader_list = loaddata()

    server_ip = "localhost"  # Replace with your server IP

    federated_train_and_send(train_loader_list, int(args.client_id), int(args.num_rounds), int(args.num_epochs), float(args.lr), server_ip, int(args.receive_port), int(args.send_port))