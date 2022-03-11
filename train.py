from __future__ import division
import torch
import torch.utils.data
import numpy as np

def step(split, epoch, dataLoader, model, criterion, optimizer = None):
    if split == 'train':
        model.train()
    else:
        model.eval()

    total_loss = 0
    accuracy = 0
    total = 0
    correct = 0
    for step, (img, target) in enumerate(dataLoader):
        img, target = img.cuda(), target.cuda()
        model = model.cuda()
        output = model(img)
        loss = criterion(output, target)
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += torch.sum(predicted == target)
    accuracy = 100 * float(correct) / total

    return total_loss, accuracy

def train(epoch, train_loader, model, criterion, optimizer):
    return step('train', epoch, train_loader, model, criterion, optimizer)

def val(epoch, val_loader, model, criterion):
    return step('val', epoch, val_loader, model, criterion)