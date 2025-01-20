import os
import sys
import pandas as pd
import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import random_projection
from torchvision.models.inception import InceptionOutputs
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
import ALERT_setting as setting
import timm
from timm.models.layers import to_2tuple,trunc_normal_


class ConcatModel(nn.Module):
    def __init__(self, base_model_t, base_model_f, classifier, lorv):
        super(ConcatModel, self).__init__()
        self.base_model_t = base_model_t
        self.base_model_f = base_model_f
        self.classifier = classifier
        self.lorv = lorv
        if self.lorv == 'RNN':
            self.lstm = LSTM()
            self.hidden_size = self.lstm.hidden_size
            self.layer_size = self.lstm.layer_size
            self.features_size = None
        elif self.lorv == 'DeiT':
            self.deit = VisionTransformer_deit()
        elif self.lorv == 'ViT':
            self.vit = VisionTransformer()


    def forward(self, t, f):
        features_t = self.base_model_t(t)
        features_f = self.base_model_f(f)
        
        if self.lorv == 'RNN':
            #features = features_f # For MVL
            features = torch.cat((features_t, features_f), dim=2) #[Batch_size, seq_length, input_size]
            self.features_size = features.size(2) # is input_size
            output = self.lstm(features.float()).to(setting.device)
            output = self.classifier(output, None)
        elif self.lorv == 'DeiT':
            output = self.deit(features_t, features_f) 
            output = self.classifier(output, None)
        elif self.lorv == 'ViT':
            output = self.vit(features_t, features_f) 
            output = self.classifier(output, None)
        elif self.lorv == 'ISAViT':
            #features_f = features_f * 0.5 #for beta learning
            #features = torch.cat((features_t, features_f), dim=1)
            output = self.classifier(features_t, features_f) #general case
            #output = self.classifier(features_f, None) #single case
        else:
            #output = self.classifier(features_f, None) # For MVL
            output = self.classifier(features_t, features_f)
     
        return output

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()         
        self.num_classes = num_classes
        self.initialized = False
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.fc1 = nn.Linear(3, 2)
        self.fc2 = nn.Linear(2, 1)
        self.fc3 = nn.Linear(1, 1)
        self.output_size = None

    def forward(self, t, f):
        # Concatenate the features from both models
        
        if f != None:
            #beta = torch.sigmoid(self.beta)
            #alpha = torch.sigmoid(self.alpha)
            #t = alpha * t
            #f = self.beta * f
            #print(self.beta)
            output = torch.cat((t, f), dim=1)
        else:
            output = t
        feature = output
        self.output_size = output.shape[1]
        
        if not self.initialized:
            self.fc1 = nn.Linear(self.output_size, self.output_size//2).to(setting.device)
            self.fc2 = nn.Linear(self.output_size//2, self.output_size//4).to(setting.device)
            self.fc3 = nn.Linear(self.output_size//4, self.num_classes).to(setting.device)
            self.initialized = True
        output = torch.relu(self.fc1(output))
        output = torch.relu(self.fc2(output))
        output = self.fc3(output)
        
        return feature, output


class GoogLeNet3(nn.Module):
    def __init__(self):
        super(GoogLeNet3, self).__init__()
        if setting.pretrained == True:
            self.googlenet3 = models.inception_v3(pretrained=True, aux_logits=True)
            #self.googlenet3.Conv2d_1a_3x3.conv = torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
            self.googlenet3.aux_logits = False
            self.googlenet3.AuxLogits = None
            
        else:
            self.googlenet3 = models.inception_v3(pretrained=False, aux_logits=False)
            self.googlenet3.Conv2d_1a_3x3.conv = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.googlenet3.fc = torch.nn.Identity()
            
    def forward(self, x):
        if x.shape[2] < 80:
            x = F.pad(x, (0, 0, int((80-x.shape[2])/2), int((80-x.shape[2])/2))) # For making proper input size
        if setting.pretrained == True:
            x = x.repeat(1, 3, 1, 1)
        x = self.googlenet3(x)      
        if isinstance(x, InceptionOutputs):
            x = x.logits
        x = x.view(x.size(0), -1)
        return x

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        # Load a standard pre-trained ResNet50 model
        if setting.pretrained == True:
            self.resnet = models.resnet50(pretrained=True)
        #self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias=False)
        
        else:
            self.resnet = models.resnet50(pretrained=False)
            self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=0, bias=False)
        # Remove the fully connected layer (classifier)
        self.resnet.fc = torch.nn.Identity()
        
    def forward(self, x):
        if setting.pretrained == True:
            x = x.repeat(1, 3, 1, 1)
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        return x


class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        if setting.pretrained == True:
            self.densenet = timm.create_model('densenet121', pretrained=True)#models.resnet18(pretrained=True)
        else:
            self.features = nn.Sequential(
                nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            
            # Each dense block
            num_features = num_init_features
            for i, num_layers in enumerate(block_config):
                block = _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                )
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                
                if i != len(block_config) - 1:  # do not add transition layer after the last block 
                    trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2) # transition layer reduces the channel for calculation efficiency
                    self.features.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2
            
            # Final batch norm
            self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        self.densenet.classifier = torch.nn.Identity()
        
    def forward(self, x):
        if setting.pretrained == True:
            x = x.repeat(1, 3, 1, 1)
            x = self.densenet(x)
            x = x.view(x.size(0), -1)
        else:
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
        return x

class MobileNet3(nn.Module):
    def __init__(self):
        super(MobileNet3, self).__init__()
        if setting.pretrained == True:
            self.mobilenet3 = models.mobilenet_v3_large(pretrained=True)
        else:
            self.mobilenet3 = models.mobilenet_v3_large(pretrained=False)
            self.mobilenet3.features[0][0] = torch.nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding = (1, 1), bias=False)
        
        self.mobilenet3.classifier = torch.nn.Identity()
        
    def forward(self, x):
        if setting.pretrained == True:
            x = x.repeat(1, 3, 1, 1)
        x = self.mobilenet3(x)
        x = x.view(x.size(0), -1)
        return x

# Not used
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.hidden_size = 1024 #1024 in CNNRNN
        self.layer_size = 1 # 1in CNNRNN
        self.initialized = False
        self.input_size = 51 #don't care number
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.layer_size, batch_first=True, bidirectional=True) #True in CNNRNN
        #self.lstm = nn.RNN(self.input_size, self.hidden_size, batch_first=True)
        
    def forward(self, x):
        if not self.initialized:
            input_size = x.size(2)
            #print(input_size) #140
            self.lstm = nn.LSTM(input_size, self.hidden_size, self.layer_size, batch_first=True, bidirectional=True).to(setting.device)
            #self.lstm = nn.RNN(input_size, self.hidden_size).to(setting.device)
            self.initialized = True            
        h0 = torch.zeros(self.layer_size*2, x.size(0), self.hidden_size).to(setting.device)
        c0 = torch.zeros(self.layer_size*2, x.size(0), self.hidden_size).to(setting.device)
        out, _ = self.lstm(x, (h0,c0)) #[sequence length, batch_size, hidden size(output_size)]
        #out, _ = self.lstm(x, h0) #[sequence length, batch_size, hidden size(output_size)]
        
        out = out[:, -1, :]
        #print(out.shape)
        return out


class CNN_RNN(nn.Module):
    def __init__(self):
        super(CNN_RNN, self).__init__()
        self.mini_ResNet = ResNet50()
        self.feature_list = []

    def forward(self, x):
        feature_list = []
        snippet = int(x.size(3) / 10)
        for i in range(10):
            snippet_feature = self.mini_ResNet(x[:,:,:,snippet*i:snippet*(i+1)])
            feature_list.append(snippet_feature)
        #feature_list=torch.tensor(np.array(feature_list))
        #feature_list = feature_list.transpose(0,1)
        feature_list = torch.stack(feature_list, dim=1)
        return feature_list #[batch, snippet, input_size]


# Simple Vision Transformer Architecture
class pre_Transformer(nn.Module):
    def __init__(self, img_height=None, num_patches=10, emb_size=384):
        super().__init__()
        
    def forward(self, x):
        x = nn.Upsample(size=(224, 224), mode='bilinear')(x)
        return x


class VisionTransformer_deit(nn.Module):
    def __init__(self, img_height=None, num_patches=10, emb_size=384, num_heads=16, depth=16): #num_head:12, depth:12, emb_size:768
        super().__init__()
        if setting.pretrained == True:
            self.model = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=True)
        else:
            self.model = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=False)
        
        #self.model.patch_embed.proj = torch.nn.Conv2d(1, 192, kernel_size=(16, 16), stride=(16, 16)) # For MVL
        #self.model.patch_embed.proj = torch.nn.Conv2d(2, 192, kernel_size=(16, 16), stride=(16, 16))
        
        #weight change 
        self.model.head = torch.nn.Identity()#torch.nn.Linear(192, 192)
        self.model.head_dist = torch.nn.Identity()

    def forward(self, x1, x2):
        #x = x2 # For MVL
        #x = x.repeat(1, 3, 1, 1)# For MVL
        x = torch.cat((x1, x2), dim=1) #[B, 2, 244, 244]
        third_channel = x.mean(dim=1, keepdim=True)  # Calculate the mean of the two channels
        x = torch.cat((x, third_channel), dim=1)
        x = self.model(x)
        return x  # Class prediction


class VisionTransformer(nn.Module):
    def __init__(self, img_height=None, num_patches=10, emb_size=384, num_heads=16, depth=16): #num_head:12, depth:12, emb_size:768
        super().__init__()
        if setting.pretrained == True:
            self.model = timm.create_model('vit_large_patch16_224', pretrained=True)
        else:
            print("We don't provide the scratch model.")
            sys.exit()
        #self.model.patch_embed.proj = torch.nn.Conv2d(1, 192, kernel_size=(16, 16), stride=(16, 16)) # For MVL
        #self.model.patch_embed.proj = torch.nn.Conv2d(2, 1024, kernel_size=(16, 16), stride=(16, 16))
        #weight change 
        self.model.head = torch.nn.Identity()#torch.nn.Linear(192, 192)
        #self.model.head_dist = torch.nn.Identity()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1) #[B, 2, 244, 244]
        third_channel = x.mean(dim=1, keepdim=True)  # Calculate the mean of the two channels
        x = torch.cat((x, third_channel), dim=1)
        x = self.model(x)
        return x  # Class prediction        

class ISAViT(nn.Module):
    def __init__(self): 
        super().__init__()
        if setting.pretrained == True:
            self.model = timm.create_model('vit_large_patch16_224', pretrained=True)
        else:
            self.model = timm.create_model('vit_large_patch16_224', pretrained=False)
        
        # conv filter weight resizing
        self.initialized = False
        self.h_num_padding = None
        self.w_num_padding = None
        self.num_patches = None
        self.patch_side_h = 37
        self.patch_side_w = 37
        self.side_h = 504
        self.side_w = 504
        self.original_num_patches = self.model.patch_embed.num_patches
        self.original_hw = int(self.original_num_patches ** 0.5) # 14
        self.original_embedding_dim = self.model.pos_embed.shape[2]
        self.original_patch_size = self.model.patch_embed.proj.weight.shape[2]


        #projection for time initialization
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(10, 10), stride=(10, 10))
        weight_onechannel = torch.sum(self.model.patch_embed.proj.weight, dim=1).unsqueeze(1)
        new_proj.weight = torch.nn.Parameter(self.patch_avg_pool(10, 10, 10, 10, weight_onechannel))
        new_proj.bias = self.model.patch_embed.proj.bias
        self.model.patch_embed.proj = new_proj

        #self.model.patch_embed.strict_img_size = False
        #self.model.patch_embed.dynamic_img_pad = False

        self.model.head = torch.nn.Identity()#torch.nn.Linear(192, 192)
        #self.model.head_dist = torch.nn.Identity()
    
    def patch_avg_pool(self, patch_size_old_h, patch_size_old_w, patch_size_new_h, patch_size_new_w, patch_weight):
        if patch_size_old_h >= patch_size_new_h and patch_size_old_w >= patch_size_new_w:
            k_size_h = patch_size_old_h - patch_size_new_h + 1
            k_size_w = patch_size_old_w - patch_size_new_w + 1
            new_patch_weight = F.avg_pool2d(patch_weight, kernel_size = (k_size_h, k_size_w), stride = 1)
        else: 
            new_patch_weight = F.interpolate(patch_weight, size=(patch_size_new_h, patch_size_new_w), mode='bilinear', align_corners=False)
    
        return new_patch_weight

    def forward(self, x):
        if not self.initialized:
            long_side = max(x.shape[2], x.shape[3])
            self.patch_side_h = int(long_side/self.original_hw) + 1
            self.patch_side_w = self.patch_side_h
            self.side_h = self.patch_side_h * self.original_hw
            self.side_w = self.patch_side_w * self.original_hw
            # make patch_embedding and assign the pre-trained weight
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(self.patch_side_h, self.patch_side_w), stride=(self.patch_side_h, self.patch_side_w))
            weight_onechannel = torch.sum(self.model.patch_embed.proj.weight, dim=1).unsqueeze(1)
            new_proj.weight = torch.nn.Parameter(self.patch_avg_pool(self.original_patch_size, self.original_patch_size, self.patch_side_h, self.patch_side_w, weight_onechannel))
            new_proj.bias = self.model.patch_embed.proj.bias
            self.model.patch_embed.proj = new_proj
            
            self.num_patches = self.original_num_patches
            self.initialized = True
        

        x = F.interpolate(x, size=(self.side_h, self.side_w), mode='bilinear', align_corners=False) #self.side_handw must be save 
        
        B = x.shape[0]
        x = self.model.patch_embed.proj(x)
        #print(x.shape)
        x = x.view(B, self.original_embedding_dim, self.num_patches).transpose(1, 2)
        
        cls_tokens = self.model.cls_token.expand(B, -1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)
        x = self.model.head(x)
        x = x[:, 0]

        return x  # Class prediction


def model_selection(model):
    if model == 'LeNet': # LeNet-5
        base_model_t = LeNet5()
        base_model_f = LeNet5()
        epochs = 30
        lr = 0.3 # SGD:0.01~0.1, adam:0.0001~0.001
        criterion = nn.CrossEntropyLoss() 
    elif model == 'AlexNet':
        base_model_t = AlexNet()
        base_model_f = AlexNet()
        epochs = 30
        lr = 0.5 # SGD 0.001~0.01
        criterion = nn.CrossEntropyLoss()
    elif model == 'VGG':
        base_model_t = VGG19()
        base_model_f = VGG19()
        epochs = 30
        lr = 0.05 # Recommended 0.01 
        criterion = nn.CrossEntropyLoss()
    elif model == 'GoogLeNet':
        base_model_t = GoogLeNet3()
        base_model_f = GoogLeNet3()
        epochs = 30
        lr = 0.0001
        criterion = nn.CrossEntropyLoss()
    elif model == 'ResNet':
        base_model_t = ResNet50()
        base_model_f = ResNet50()
        epochs = 30
        lr = 0.0001
        criterion = nn.CrossEntropyLoss()
    elif model == 'DenseNet':
        base_model_t = DenseNet121()
        base_model_f = DenseNet121()
        epochs = 30
        lr = 0.0001
        criterion = nn.CrossEntropyLoss()
    elif model == 'MobileNet':
        base_model_t = MobileNet3()
        base_model_f = MobileNet3()
        epochs = 30
        lr = 0.001 # SGD:0.005~0.01, Adam:0.001
        criterion = nn.CrossEntropyLoss()
    elif model == 'CNN+RNN':
        base_model_t = CNN_RNN()
        base_model_f = CNN_RNN()
        epochs = 30
        lr = 0.0001
        criterion = nn.CrossEntropyLoss()
    elif model == 'DeiT':
        base_model_t = pre_Transformer()
        base_model_f = pre_Transformer()
        epochs = 300
        lr = 0.00001 #AdamW: 0.0001~0.001, optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
        criterion = nn.CrossEntropyLoss()
    elif model == 'ViT':
        base_model_t = pre_Transformer()
        base_model_f = pre_Transformer()
        epochs = 30
        lr = 0.00001 
        criterion = nn.CrossEntropyLoss()
    elif model == 'ISAViT':
        base_model_t = ISAViT()
        base_model_f = MobileNet3()
        lr = 0.00001 
        #lr = 0.0001 # Beta testing
        criterion = nn.CrossEntropyLoss()
    elif model == 'all':
        for lr_alg in learning_algorithms:
            None
    else:
        pass
    
    classifier = Classifier(setting.num_classes)
    if model == 'RNN' or model == 'CNN+RNN':
        base_model = ConcatModel(base_model_t, base_model_f, classifier, 'RNN').to(setting.device)
    elif model == 'DeiT':
        base_model = ConcatModel(base_model_t, base_model_f, classifier, 'DeiT').to(setting.device)
    elif model == 'ViT':
        base_model = ConcatModel(base_model_t, base_model_f, classifier, 'ViT').to(setting.device)
    elif model == 'ISAViT':        
        base_model = ConcatModel(base_model_t, base_model_f, classifier, 'ISAViT').to(setting.device)
    else:
        base_model = ConcatModel(base_model_t, base_model_f, classifier, None).to(setting.device)
    
    if model == 'GoogLeNet':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)  # Recommended RMSprop
    elif model == 'ResNet':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)
    elif model == 'DenseNet':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)
    elif model == 'MobileNet':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)
    elif model == 'CNN+RNN':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)
    elif model == 'DeiT':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)#optim.AdamW(base_model.parameters(), lr=lr, weight_decay=0.05) #optim.SGD(base_model.parameters(), lr=lr, momentum=0.9)
    elif model == 'ViT':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)#optim.AdamW(base_model.parameters(), lr=lr, weight_decay=0.05) #optim.SGD(base_model.parameters(), lr=lr, momentum=0.9)
    elif model == 'ISAViT':
        optimizer = optim.Adam(base_model.parameters(), lr=lr)
    elif model == 'all':
        pass
    else:
        pass
    
    return base_model, optimizer, criterion, epochs, lr
