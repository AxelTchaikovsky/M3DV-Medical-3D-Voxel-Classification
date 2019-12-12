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
    # os.environ["CUDA_VISIBLE_DEVICES"] = config["GPU"]
    DEVICE = t.device(config["DEVICE"])
    LR = config['lr']
    EPOCH = config['epoch']

    DataSet = dataloader.MyDataSet()
    train_data, test_data = DataSet.test_train_split()

    train_loader = DataLoader.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    test_loader = DataLoader.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    #model = Resnet.ResNet18().to(DEVICE)
    model = VoxNet.MVVoxNet(2).to(DEVICE)

    # Multi GPU setting
    # model = t.nn.DataParallel(model,device_ids=[0,1])

    optimizer = t.optim.Adam(model.parameters())

    criterian = t.nn.CrossEntropyLoss().to(DEVICE)

    # Test the train_loader
    
    
    for epoch in range(EPOCH):
        model = model.train()
        multiprocess_idx = 2
        train_loss = 0
        correct = 0
        for batch_idx, [data, label] in enumerate(train_loader):
            #data = data.unsqueeze(1)
            #print(data.shape)
            data, label = data.to(DEVICE), label.to(DEVICE)
            out = model(data).squeeze()
            loss = criterian(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            pred = out.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(label.view_as(pred)).sum().item()
        train_loss /= len(train_loader.dataset)
        print('\nEpoch: {}, Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
                                                                                                 train_loss, correct, len(
                                                                                                     train_loader.dataset),
                                                                                                 100. * correct / len(train_loader.dataset)))


        model = model.eval()
        test_loss = 0                                                                        
        correct = 0
        for batch_idx, [data, label] in enumerate(test_loader):
            with t.no_grad():
                data, label = data.to(DEVICE), label.to(DEVICE)
                out = model(data)
                loss = criterian(out, label)
                test_loss += loss
                pred = out.max(1, keepdim=True)[1]  # 找到概率最大的下标
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
