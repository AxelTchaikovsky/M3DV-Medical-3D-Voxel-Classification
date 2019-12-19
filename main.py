import json
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import multiprocessing

from model import dataloader_v2 as dataloader
from model import Resnet
from model import VoxNet
from model.func import save_model, eval_model_new_thread, eval_model

if __name__ == "__main__":
    time_start = time.time()

    config = json.load(open("config.json"))
    DEVICE = t.device(config["DEVICE"])
    LR = config['lr']
    EPOCH = config['epoch']

    DataSet = dataloader.MyDataSet()
    train_data, test_data = DataSet.test_train_split()

    train_loader = DataLoader.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, 
        num_workers=config["num_workers"])
    test_loader = DataLoader.DataLoader(
        test_data, batch_size=1, shuffle=False, 
        num_workers=config["num_workers"])

    #model = Resnet.ResNet18().to(DEVICE)
    model = VoxNet.MVVoxNet(2).to(DEVICE)

    #optemizer to use adam, SGD
    optimizer = t.optim.SGD(model.parameters(),lr=LR)

    #criterian to use CrossEntropy
    criterian = t.nn.CrossEntropyLoss().to(DEVICE)

    #STARTING TRAIN AND TEST
    for epoch in range(EPOCH):
        ##################################################
        #PYTORCH start training mode
        ##################################################
        model = model.train()
        train_loss = 0
        correct = 0
        for batch_idx, [data, label] in enumerate(train_loader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            out = model(data).squeeze()
            loss = criterian(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            pred = out.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
        train_loss /= len(train_loader.dataset)
        print('\nEpoch: {}, Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
                                                                                                 train_loss, correct, len(
                                                                                                     train_loader.dataset),
                                                                                                 100. * correct / len(train_loader.dataset)))

        ##################################################
        #PYTORCH start evaluation mode
        ##################################################
        model = model.eval()
        test_loss = 0                                                                        
        correct = 0
        for batch_idx, [data, label] in enumerate(test_loader):
            with t.no_grad():
                data, label = data.to(DEVICE), label.to(DEVICE)
                out = model(data)
                loss = criterian(out, label)
                test_loss += loss
                pred = out.max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
                                                                                                 test_loss, correct, len(
                                                                                                     test_loader.dataset),
                                                                                                 100. * correct / len(test_loader.dataset)))

        save_model(model, epoch)
        eval_model_new_thread(epoch, 1)
        # LZX pls using the following code instead
        # multiprocessing.Process(target=eval_model(epoch, '0'), args=(multiprocess_idx,))
        # multiprocess_idx += 1
    time_end = time.time()
    print('time cost', time_end-time_start)
