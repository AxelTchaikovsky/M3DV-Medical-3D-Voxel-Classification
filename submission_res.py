import json
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import multiprocessing

from model.dataloader_v3 import *
from model import Resnet
from model import VoxNet_try as VoxNet
from model.func import save_model, eval_model_new_thread, eval_model, load_model
import argparse
#from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold
import pandas as pd
if __name__ == "__main__":
    time_start = time.time()

    config = json.load(open("config.json"))
    DEVICE = t.device(config["DEVICE"])

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default=config["GPU"], type=str, help="choose which DEVICE U want to use")
    parser.add_argument("--epoch", default=28, type=int,
                        help="The epoch to be tested")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    test_set = In_the_wild_set()
    test_set.sort()
    test_loader = DataLoader.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    criterian = t.nn.CrossEntropyLoss()

    model1 = Resnet.resnet18().to(DEVICE)
    # Test the train_loader
    model1.load_state_dict(t.load("saved_model/74.pkl"))
              #t.load("saved_model/41.pkl"))
    model1.eval()

    with t.no_grad():
        # Test the test_loader
        test_loss = 0
        correct = 0
        idx = []
        Name = []
        Score = []
        for batch_idx, [data, name] in enumerate(test_loader):
            data = data.to(DEVICE)
            out1 = t.nn.functional.softmax(model1(data))
            out = out1
            out = out.squeeze()
            Name.append(name[0])
            Score.append(out[1].item())
        test_dict = {'Id': Name, 'Predicted': Score}
        test_dict_df = pd.DataFrame(test_dict)
        print(test_dict_df)
        path = 'result'
        if not os.path.exists(path):
            os.makedirs(path)
        test_dict_df.to_csv('result/Submission_74_mixup_res_adam.csv', index=False)