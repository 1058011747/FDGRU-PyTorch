import os
import time
import datetime
import argparse

import torch
import torch.utils.data
import torchvision.transforms as transforms
from dataset import CWRUdata
from logger import Logger
from model_CNN_LSTM import RNN
from train import train, val
import numpy as np
# import seaborn as sn
# import pandas as pd
import matplotlib.pyplot as plt
def main():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    logger = Logger(arg.saveDir + '/logs_{}'.format(now))

    if arg.loadModel != 'none':
        model = torch.load(arg.loadModel).cuda()
    else:
        # input_size, hid_size, seq, layers, class
        model = RNN(1024,1024,32,1,10).cuda() # parameters of model

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.RMSprop(model.parameters(),
        lr=arg.LR,
        alpha=0.99,
        eps=1e-08,
        weight_decay=0, momentum=0, centered=False)

    train_data = CWRUdata(arg.DATA_DIR, split = 'trian', transform = transforms.ToTensor())
    test_data = CWRUdata(arg.TEST_DATA_DIR, split ='trian', transform = transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=arg.trainBatch,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=arg.valBatch,
        shuffle=False)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    best_accuracy = 0.0
    for epoch in range(1, arg.nEpochs+1):
        timestart = time.time()
        loss_train, acc_train = train(epoch, train_loader, model, criterion, optimizer)
        logger.scalar_summary('loss_train', loss_train, epoch)
        logger.scalar_summary('acc_train', acc_train, epoch)
        logger.scalar_summary('time', time.time()-timestart, epoch)
        if epoch % arg.valIntervals == 0:
            loss_val, acc_val = val(epoch, test_loader, model, criterion)
            logger.scalar_summary('loss_val', loss_val, epoch)
            logger.scalar_summary('acc_val', acc_val, epoch)
            folder = os.path.exists(arg.savemodel)
            if not folder:
                os.makedirs(arg.savemodel)
            if acc_val >= best_accuracy:
                best_accuracy = acc_val
                torch.save(model, arg.savemodel + '/best_model_{}.pth'.format(i+1))
            logger.write('{:8f} {:8f} {:8f} {:8f} \n'.format(loss_train, acc_train, loss_val, acc_val))
            print('Epoch:{} | Loss:{:8f} | Accuracy:{:8f} | Loss_Val:{:8f} | Acc_Val:{:8f} | Time: {:5f} \n'.format(epoch, loss_train, acc_train, loss_val, acc_val, time.time()-timestart))
        else:
            logger.write('{:8f} {:8f} \n'.format(loss_train, acc_train))
            print('Epoch:{} | Loss:{:8f} | Accuracy:{:8f} | Time: {:5f} \n'.format(epoch, loss_train, acc_train, time.time()-timestart))
        # adjust_learning_rate(optimizer, epoch, arg.dropLR, arg.LR)
        if epoch in arg.dropLR:
            LR = arg.LR * (0.1 ** (arg.dropLR.index(epoch) + 1))
            print(LR)
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR

    torch.save(model, arg.savemodel + '/model_{}.pth'.format(i + 1))
    logger.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--loadModel', default = 'none', help = 'Provide full path to a previously trained model')
    parser.add_argument('--DATA_DIR', default = '***', help = 'Path (folder) to data')  # path of the train dataset
    parser.add_argument('--TEST_DATA_DIR', default='***', help='Path (folder) to data')  # path of the test dataset
    parser.add_argument('--LR', type = float, default = 2e-4, help = 'Learning Rate')
    parser.add_argument('--dropLR',  default = [25,50], help = 'drop LR')
    parser.add_argument('--nEpochs', type = int, default = 60, help = '#training epochs')
    parser.add_argument('--trainBatch', type = int, default = 32, help = 'Mini-batch size')
    parser.add_argument('--valBatch', type = int, default = 32, help = 'Mini-batch size')
    parser.add_argument('--valIntervals', type = int, default = 1, help = '#valid intervel')
    parser.add_argument('--saveDir', default = '***', help = 'save log and model')  # save path of the logfile
    parser.add_argument('--savemodel', default = '***', help = 'save log and model')  # save path of the model

    arg = parser.parse_args()

    for i in range(10):
        main()
    # np.savetxt(arg.saveDir+'/best_accuracy.npy', best, delimiter="\\n")