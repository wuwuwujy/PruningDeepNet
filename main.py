import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torch.optim as optim
from torchvision.transforms import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from vgg_prune import vgg16_bn_prune
from vgg import vgg16_bn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--batch_size', type=int, default=256)
# parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--pruning_constant', type=float, default=0.3)

args = parser.parse_args()


def get_data():
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        
    #     transforms.Resize(32),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    batch_size=128

    trainset = torchvision.datasets.CIFAR100(root='./cifar10', train=True, download=False, transform=transform_train)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR100(root='./cifar10', train=False, download=False, transform=transform_test)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader

def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
    return correct / total

#CITE from https://github.com/wanglouis49/pytorch-weights_pruning/blob/master/pruning/utils.py
def prune_rate(model, verbose=True):

    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0

    for parameter in model.parameters():

        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = \
                np.count_nonzero(parameter.cpu().data.numpy()==0)
            nb_zero_param += zero_param_this_layer

            if verbose:
                print("Layer {} | {} layer | {:.2f}% parameters pruned" \
                    .format(
                        layer_id,
                        'Conv' if len(parameter.data.size()) == 4 \
                            else 'Linear',
                        100.*zero_param_this_layer/param_this_layer,
                        ))
    pruning_perc = 100.*nb_zero_param/total_nb_param
    if verbose:
        print("Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc

def train(model, prune, train_dataloader, test_dataloader, model_name, lr, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    training_loss = 0
    start_time = time.time()
    train_acc=[]
    test_acc=[]
    if prune==False:
        for epoch in range(num_epochs):
            correct = 0
            total = 0
            if epoch > 50:
                lr = 0.00001
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            iteration = 0
            training_loss=0
            start_time = time.time()

            for batch_idx, (X_train_batch,Y_train_batch) in enumerate(train_dataloader):
                X_train_batch,Y_train_batch = Variable(X_train_batch).cuda(),Variable(Y_train_batch).cuda()
                outputs = model(X_train_batch)
                loss = criterion(outputs, Y_train_batch)
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)#output.data.max(1)[1]
                total += Y_train_batch.size(0)
                correct += (predicted == Y_train_batch.data).sum()
                training_loss += loss.item()
                iteration += 1
            train_acc.append(float(correct) / float(total) * 100.)
            print('\nEpoch ', epoch,' training accuracy =', float(correct) / float(total) * 100., " running time =", time.time() - start_time, "s")
            if epoch%10==0 or epoch == args.num_epochs-1:
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    test_accuracy = []
                    for batch_idx, (X_test_batch,Y_test_batch) in enumerate(test_dataloader):
                        X_test_batch,Y_test_batch = Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()
                        outputs = model(X_test_batch)
                        _, predicted = torch.max(outputs.data, 1)
                        total += Y_test_batch.size(0)
                        correct += (predicted == Y_test_batch.data).sum()
                test_acc.append(float(correct) / float(total) * 100.)
                torch.save(model,'temp.model')
                with open(model_name+'train_acc.txt', 'w') as f:
                    for item in train_acc:
                        f.write("%s, " % item)
                f.close()
                    
                with open(model_name+'test_acc.txt', 'w') as f:
                    for item in test_acc:
                        f.write("%s, " % item)
                f.close()
        torch.save(model,model_name)
    else:
        for epoch in range(num_epochs):
            correct = 0
            total = 0
            if epoch > 20:
                learning_rate = lr/2
            if epoch >50:
                learning_rate = lr/5
            if epoch<4:
                model.set_flags(False)
            elif (epoch)%5==0:
                if epoch == 9:
                    train_acc[8] = 5
                    train_acc[7] = 11
                if trend[epoch]+trend[epoch-1]<=-2 or train_acc[epoch-1]-train_acc[epoch-2]<-4:
                    model.set_weight_back()
                else:
                    model.set_flags(True)
                    model.update_masks(c_rate)  
                    prune_rate(model)
                    if c_rate > 0.6:
                        c_rate -= 0.2
                    if c_rate > 0.2:
                        c_rate -= 0.1
            else:
                model.set_flags(False)
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            iteration = 0
            training_loss=0
            start_time = time.time()
            for batch_idx, (X_train_batch,Y_train_batch) in enumerate(train_dataloader):
                X_train_batch,Y_train_batch = Variable(X_train_batch).cuda(),Variable(Y_train_batch).cuda()
                outputs = model(X_train_batch)
                loss = criterion(outputs, Y_train_batch)
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total += Y_train_batch.size(0)
                correct += (predicted == Y_train_batch.data).sum()
                training_loss += loss.item()
                iteration += 1
            train_acc.append(float(correct) / float(total) * 100.)
            trend.append(np.sign(float(correct) / float(total) * 100.-trend[epoch]))
            print('\nEpoch ', epoch,' training accuracy =', float(correct) / float(total) * 100., " running time =", time.time() - start_time, "s")

            if epoch%10==0 or epoch == args.num_epochs-1:
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    test_accuracy = []
                    for batch_idx, (X_test_batch,Y_test_batch) in enumerate(test_dataloader):
                        X_test_batch,Y_test_batch = Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()
                        outputs = model(X_test_batch)
                        _, predicted = torch.max(outputs.data, 1)
                        total += Y_test_batch.size(0)
                        correct += (predicted == Y_test_batch.data).sum()
                test_acc.append(float(correct) / float(total) * 100.)
                torch.save(model,'temp.model')
                with open(model_name+'train_acc.txt', 'w') as f:
                    for item in train_acc:
                        f.write("%s, " % item)
                f.close()
                    
                with open(model_name+'test_acc.txt', 'w') as f:
                    for item in test_acc:
                        f.write("%s, " % item)
                f.close()
        torch.save(model,model_name)
        return model

if __name__ == '__main__':
    train_loader, test_loader = get_data()
    print('========Data Load Successful=========')
    train(vgg16_bn(), prune=False, train_dataloader=rain_loader, test_dataloader=test_loader, model_name='vgg16', lr=0.001, num_epochs=100)
    train(vgg16_bn_prune(),prune=False, train_dataloader=rain_loader, test_dataloader=test_loader, model_name='vgg16', lr=0.001, num_epochs=100)

    










