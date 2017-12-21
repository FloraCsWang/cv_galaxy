from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import load_data
import realtime_augmentation as ra
from preprocess import dataLoader

import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

### Data Initialization and Loading

train_ids = load_data.train_ids
num_train = len(train_ids)
num_valid = num_train // 10 # integer division
num_train -= num_valid
train_indices = np.arange(num_train)
valid_indices = np.arange(num_train, num_train + num_valid)


from model import Net, weight_init




### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
model = Net()
model.apply(weight_init)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
'''
for parameter in model.parameters():
    print(parameter)
'''


def train(epoch, train_loader):
    model.train()
    train_data_loss_in = []
    train_num = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        train_num += 1
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        #print(target)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                                                           epoch, batch_idx * len(data), train_num,
                                                                           100. * batch_idx / train_num, loss.data[0]))
        train_data_loss_in.append(loss.data[0])
    return train_data_loss_in

def validation(val_loader):
    model.eval()
    validation_loss = 0
    correct = 0
    val_num = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        val_num += 1
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # validation_loss += nn.MSELoss(output, target, size_average=False).data[0] # sum up batch loss
        validation_loss += F.mse_loss(output, target).data[0]
    # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    # correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= val_num
    print('\nValidation set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
                                                                                 validation_loss, val_num))

# val_data_loss = []
# val_data_acc = []
train_data_loss = []
for epoch in range(1, args.epochs + 1):
    print("\nepoch: %d" % epoch)
    
    
    # need to set train_loader and val_loader again at each epoch
    train_loader = dataLoader( data_indices = train_indices, batch_size=40, shuffle=True)
    val_loader = dataLoader( data_indices = valid_indices, batch_size=5, shuffle=False)
    
    
    
    '''
    for batch_idx, (data, target) in enumerate(train_loader):
        print("batch: %d" % batch_idx)
        print(data.size(), target.size())
        train_num += 1
    
    print("validation batches:")
    for batch_idx, (data, target) in enumerate(val_loader):
        print("batch: %d" % batch_idx)
        print(data.size(), target.size())
        val_num += 1
    '''
    train_loss = train(epoch, train_loader)
    validation(val_loader)
    # val_data_loss.append(val_loss)
    # val_data_acc.append(val_acc)
    train_data_loss = train_data_loss + train_loss
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')


print(train_data_loss)


# plt.plot(range(1, args.epochs + 1), val_data_loss)
# plt.xlabel('epchos')
# plt.ylabel('val_loss')
# plt.show()

# plt.plot(range(1, args.epochs + 1), val_data_acc)
# plt.xlabel('epchos')
# plt.ylabel('val_acc')
# plt.show()

#plt.plot(range(1, len(train_data_loss) + 1), train_data_loss)
#plt.title('train loss vs stpes')
#plt.xlabel('epchos')
#plt.ylabel('val_loss')
#plt.show()
