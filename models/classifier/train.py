#################
### Load data ###
#################

from models.classifier.dataset import prepare_data

data_dir = "/content/drive/MyDrive/Colab Notebooks/COMP4471/Project/facemask" 
train_loader, val_loader, test_loader = prepare_data(data_dir=data_dir, verbose=True)


#############################
### Load model checkpoint ###
#############################

USE_GPU = True
dtype = torch.float32 # We will use float torch

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
from models.classifier.model import EfficientNet

model = EfficientNet.from_name('efficientnet-b0', num_classes=3)

model_dir = "/content/drive/MyDrive/Colab Notebooks/COMP4471/Project/efficientnet-b0-ckpt4.pth"
model.load_state_dict(torch.load(model_dir))


#############
### Utils ###
#############

from sklearn.metrics import classification_report

def check_accuracy(loader, model):  
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.squeeze(1).to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc

def check_performance(loader, model):
    y_true = []
    y_pred = []
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.squeeze(1).to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            y_true += y.tolist()
            y_pred += preds.tolist()

    print(classification_report(y_true, y_pred, digits=4))
    
def print_num_params(model, display_all_modules=False):
    total_num_params = 0
    for n, p in model.named_parameters():
        num_params = 1
        for s in p.shape:
            num_params *= s
        if display_all_modules: print("{}: {}".format(n, num_params))
        total_num_params += num_params
    print("-" * 50)
    print("Total number of parameters: ", total_num_params)

def train(model, optimizer, epochs=1):
    """    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: history
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    loss_train = []
    val_acc = []

    for e in range(epochs):
        print("Epoch: ", e)
        for t, (x, y) in enumerate(train_loader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.squeeze(1).to(device=device, dtype=torch.long)

            scores = model(x)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                acc = check_accuracy(val_loader, model)
                print()
                loss_train.append(loss.item())
                val_acc.append(acc)

            if t == 500:  # for demo purpose
                break
                
        # if log per epoch
        #loss_train.append(loss.item())
        #val_acc.append(acc)           

    return loss_train, val_acc


###########
### Run ###
###########

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

# control how frequently to print train loss.
print_every = 50
print('using device:', device)

import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay= 5e-4)
history = train(model, optimizer, epochs=1)


##########################
### Performance & Save ###
##########################

check_performance(val_loader, model)
model_dir = "/content/drive/MyDrive/Colab Notebooks/COMP4471/Project/efficientnet-b0-ckpt5.pth"
torch.save(model.state_dict(), model_dir)