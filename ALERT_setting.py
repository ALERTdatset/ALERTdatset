import os
import sys
import pandas as pd
import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import pickle
import torchsummary  
import gc
import copy
import random
from tqdm import tqdm
import time
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity
from fvcore.nn import FlopCountAnalysis
from torchvision.models.densenet import _DenseBlock, _DenseLayer, _Transition

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Pre-settings
learning_algorithms = ['GoogLeNet', 'ResNet', 'DenseNet', 'MobileNet', 'CNN+RNN', 'DeiT', 'ViT', 'ISAViT']
cropping_methods = ['cropO', 'cropX', 'CA'] 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
subject_list = ['dk', 'jj', 'ks', 'sh', 'wh', 'yj', 'ys', 'yh', 'jp']
tester = 'jp'
label_dict = {'relax' : 0, 'sw': 1, 'nod':2, 'drink': 3, 'phone': 4, 'smoke': 5, 'panel': 6} #sw is steeringwheel
label_list = ['Relax', 'Drive', 'Nod', 'Drink', 'Phone', 'Smoke', 'Panel']
num_classes = len(label_dict)
feature_dict_original = {'0' : [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': []}
seed = 42
pretrained = True
 