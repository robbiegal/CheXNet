# encoding: utf-8

"""
The main CheXNet model trainig implementation.
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
import torch.optim as optim



CKPT_PATH = 'model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './ChestX-ray14/images'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list_factor_13.txt'
TRAIN_IMAGE_LIST = './ChestX-ray14/labels/train_list.txt'
BATCH_SIZE = 2

LOADER_WORKERS=0

LR=0.001
MOMENTUM=0.9


def main():

    cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES).cuda()
    model = torch.nn.DataParallel(model).cuda()

    #if os.path.isfile(CKPT_PATH):
    if 0:
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")


    #define loss and optimizer
    criterion=nn.CrossEntropyLoss()
    optimiser=optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TRAIN_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=LOADER_WORKERS, pin_memory=True)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    running_loss = 0.0
    # switch to train mode
    model.train()
    print("starting loop")
    try:
        for epoch in range(2):
            for i, (inp, label) in enumerate(train_loader):
                label = label.cuda()
                gt = torch.cat((gt, label), 0)
                bs, n_crops, c, h, w = inp.size()
                input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda())
                #zero parameter gradients
                optimiser.zero_grad()
                #fw + back + optimise
                output = model(input_var)  #output dim should be: minibatch, classnum,
                loss=criterion(output, label)
                loss.backward()
                optimiser.step()
                #output_mean = output.view(bs, n_crops, -1).mean(1)
                #pred = torch.cat((pred, output_mean.data), 0)
                running_loss += loss.item()
                #print statistics
                if not i%100:
                   print('[%d, %5d] loss=%.3f' %(epoch, i, running_loss))
    except:
        print('error in iteration: '+str(i))
        raise()


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == '__main__':
    main()
