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
import ALERT_models
import ALERT_setting as setting
import wandb

def usage_exam():
    print("===============================================================================================================\n")
    print("                                 USAGE EXAMPLE\n")
    print("===============================================================================================================\n")
    print("Please write down the exact command")
    print("\n")
    print("   EX)     python3 ALERT_bench.py models(All/GoogleNet/transferX) cropping(cropO/cropX) sample_size your_memo")
    print("\n")
    print("===============================================================================================================\n")
    sys.exit()

def parsing_argv():
    model = sys.argv[1]
    if model in setting.learning_algorithms or model == 'all':
        pass
    else:
        usage_exam()

    crop = sys.argv[2]
    if crop in setting.cropping_methods:
        pass
    else:
        usage_exam()

    sample_size = int(sys.argv[3])
    N_shot = int(sys.argv[4])
    adapt_epoch = int(sys.argv[5])
    exp_iteration = int(sys.argv[6])
    memo = sys.argv[7]
    
    return model, crop, sample_size, N_shot, adapt_epoch, exp_iteration, memo



def perfromance_eval(cm, acc, base_model, t_shape, f_shape, time, phase):
    print("================================================================")
    # Confusion matrix
    print(f"{phase} confusion matrix: \n{cm}")
    # Accuracy
    print(f"{phase} accuracy: {acc}%")
    
    # Precision, recall, f1_score
    precision = [0 for p in range(setting.num_classes)]
    recall = [0 for r in range(setting.num_classes)]
    f1_score = [0 for f in range(setting.num_classes)]
    eps = 1e-16
    for i in range(setting.num_classes):
        sum_p = 0.0
        sum_r = 0.0
        for j in range(setting.num_classes):
            sum_p += cm[j][i]
            sum_r += cm[i][j]

        precision[i] = cm[i][i] * 100 / (sum_p + eps)
        recall[i] = cm[i][i] * 100 / (sum_r + eps)
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]+eps)
    print(f"{phase} precision: {precision}")
    print(f"{phase} recall: {recall}")
    print(f"{phase} f1_score: {f1_score}")
    
    # Number of parameters
    num_param = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"{phase} num_parameters: {num_param}")

    # Number of FLOPs
    dummy_t = torch.randn(t_shape)
    dummy_f = torch.randn(f_shape)
    flops = FlopCountAnalysis(base_model, (dummy_t.to(setting.device), dummy_f.to(setting.device)))
    print(f"{phase} num_FLOPs (forward path): {flops.total()}")
    
    # Memory usage
    print(f"{phase} current memory allocated: {torch.cuda.memory_allocated() / 1024**2} MB")
    print(f"{phase} max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2} MB")
    
    # Spending time
    print(f"{phase} time duration: {time:.2f} s")

    # Best/worst label
    high = max(f1_score)
    low = min(f1_score)
    high_idx = f1_score.index(high)
    low_idx = f1_score.index(low)
    print(f"{phase} the best label: {next(key for key, value in setting.label_dict.items() if value == high_idx)} ")
    print(f"{phase} the wrost label: {next(key for key, value in setting.label_dict.items() if value == low_idx)} ")       
    print("=======================================================================")


def save_feature_distribution(feature_dict, train_or_test, phase):
    featurelist = []
    classlist = [0 for cls_num in range(setting.num_classes)]
    for feature in feature_dict:
        featurelist += feature_dict[feature]
        classlist[int(feature)] = len(feature_dict[feature])
    featurelist = np.array(featurelist)
    dist_list = []
    for feat in featurelist:
        dist_list.append(feat.flatten())
    featurelist = np.array(dist_list)
    
    class_labels = np.concatenate([np.full(sample_count, class_label) for class_label, sample_count in enumerate(classlist)])

    tsne = TSNE(n_components=2, random_state=42)
    embedded_features = tsne.fit_transform(featurelist)
    scatter = plt.scatter(embedded_features[:, 0], embedded_features[:, 1], c=class_labels, cmap='tab10', s=5)
    legend_labels = setting.label_list
    #plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title='Legend', fontsize=12)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig('./ALERT/feature_distribution/'+phase+'_'+model+'_'+train_or_test+'_'+crop+'_'+str(sample_size)+'_'+str(N_shot)+'_'+str(adapt_epoch)+'_'+str(exp_iteration)+'.png')
    plt.close()


if __name__ == "__main__":
    setting.set_seed(setting.seed) 
    tester = sys.argv[8]
    # Parsing functions
    model, crop, sample_size, N_shot, adapt_epoch, exp_iteration, memo = parsing_argv()

    # Preparing dataset
    time_test = []
    freq_test = []
    labels_test = []
    tmp = []
    time_adapt =[]
    freq_adapt =[]
    labels_adapt =[]
    
    with open('./pickles/'+crop+'_'+tester+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_time_data.pickle', 'rb') as f:
        tmp = pickle.load(f)
    for label_idx in range(len(tmp)):
        time_adapt += tmp[label_idx][:N_shot]
        time_test += tmp[label_idx][50:]
    with open('./pickles/'+crop+'_'+tester+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_freq_data.pickle', 'rb') as f:
        tmp = pickle.load(f)
    for label_idx in range(len(tmp)):
        freq_adapt += tmp[label_idx][:N_shot]
        freq_test += tmp[label_idx][50:]
    with open('./pickles/'+crop+'_'+tester+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_labels_data.pickle', 'rb') as f:
        tmp = pickle.load(f)
    for label_idx in range(len(tmp)):
        labels_adapt += tmp[label_idx][:N_shot]
        labels_test += tmp[label_idx][50:]

    time_adapt = torch.tensor(np.array(time_adapt))
    freq_adapt = torch.tensor(np.array(freq_adapt))
    labels_adapt = torch.tensor(np.array(labels_adapt))
    
    time_test = torch.tensor(np.array(time_test))
    freq_test = torch.tensor(np.array(freq_test))
    labels_test = torch.tensor(np.array(labels_test))   
    
    adapt_dataset = torch.utils.data.TensorDataset(time_adapt, freq_adapt, labels_adapt)
    adapt_dataloader = DataLoader(adapt_dataset, batch_size=5, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(time_test, freq_test, labels_test)
    test_dataloader = DataLoader(test_dataset, batch_size=25, shuffle=False)
    
    
    # Selecting model parameters & choosing model
    base_model, optimizer, criterion, epochs, lr = ALERT_models.model_selection(model)

    
    wandb.init(
        project = 'test', 
        config = {
        "learninig_rate": lr,
        "epochs": epochs,
        "model": model,
        "crop": crop,
        "sample_size": sample_size,
        "N_shot": N_shot,
        "adapt_epoch": adapt_epoch,
        "exp_iteration": exp_iteration
        }
    )
    wandb.run.name = model+'_'+str(exp_iteration)+'_'+memo
    wandb.run.save()
    
    
    #'''
    PHASE = 'Training phase'
    
    # Training phase
    confusion_matrix = [[0 for i in range(setting.num_classes)] for j in range(setting.num_classes)]
    feature_dict_train_trainer = copy.deepcopy(setting.feature_dict_original)
    feature_dict_train_tester = copy.deepcopy(setting.feature_dict_original)
    duration_train = 0
    log_step = 0
    for epoch in tqdm(range(epochs), desc=f"Training progress: ", leave=True):
        running_loss = 0.0
        for subject in setting.subject_list:
            if subject != tester:
                time_train = []
                freq_train = []
                labels_train = []
                if subject == 'jj' or subject == 'ks':
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_time_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        if label_idx != setting.label_dict['nod'] and label_idx != setting.label_dict['drink'] and label_idx != setting.label_dict['panel']:
                            time_train += tmp[label_idx]
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_freq_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        if label_idx != setting.label_dict['nod'] and label_idx != setting.label_dict['drink'] and label_idx != setting.label_dict['panel']:
                            freq_train += tmp[label_idx]
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_labels_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        if label_idx != setting.label_dict['nod'] and label_idx != setting.label_dict['drink'] and label_idx != setting.label_dict['panel']:
                            labels_train += tmp[label_idx]
                elif subject == 'sh' or subject == 'ys' or subject == 'yj':
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_time_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        if label_idx != setting.label_dict['smoke']:
                            time_train += tmp[label_idx]
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_freq_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        if label_idx != setting.label_dict['smoke']:
                            freq_train += tmp[label_idx]
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_labels_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        if label_idx != setting.label_dict['smoke']:
                            labels_train += tmp[label_idx]
                else:
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_time_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        time_train += tmp[label_idx]
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_freq_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        freq_train += tmp[label_idx]
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_labels_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        labels_train += tmp[label_idx]
                
                train_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(time_train)), torch.tensor(np.array(freq_train)), torch.tensor(np.array(labels_train)))
                
                base_model.train()
                        
                train_dataloader = DataLoader(train_dataset, batch_size=25, shuffle=True)
                start_train = time.time()
                #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
                for step, (t_train, f_train, l_train) in enumerate(train_dataloader):
                    t_train, f_train, l_train = t_train.to(setting.device), f_train.to(setting.device), l_train.to(setting.device)

                    t_train = t_train.unsqueeze(1) #[25,1,51,500]
                    f_train = f_train.unsqueeze(1) #[25,1,89,500]
                    
                    optimizer.zero_grad()
                    labels_indices = torch.argmax(l_train, dim=1)
                    features, outputs = base_model(t_train.float(), f_train.float())
                    
                    if epoch == epochs - 1:
                        features = features.tolist()
                        for feature_idx, labels_idx in enumerate(labels_indices):
                            feature_dict_train_trainer[str(int(labels_idx))].append(features[feature_idx])

                    loss = criterion(outputs, labels_indices)                    
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    running_loss += loss.item()
                    log_step += 1
                    if log_step % 100 == 0:
                        wandb.log({"Train loss (per 100 iterations)": loss.item()})
                #scheduler.step()    
                
                end_train = time.time()
                duration_train += end_train - start_train
        
        if epoch % 10 == 0 or epoch == epochs-1:
            base_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for t_test, f_test, l_test in test_dataloader:
                    t_test, f_test, l_test = t_test.to(setting.device), f_test.to(setting.device), l_test.to(setting.device)
                    
                    t_test = t_test.unsqueeze(1)
                    f_test = f_test.unsqueeze(1)
                    
                    features, outputs = base_model(t_test.float(), f_test.float())
                    
                    labels_indice = torch.argmax(l_test, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    total += l_test.size(0)
                    
                    if epoch == epochs-1:
                        features = features.tolist()
                        for feature_idx, labels_idx in enumerate(labels_indice):
                            feature_dict_train_tester[str(int(labels_idx))].append(features[feature_idx])

                        for k in range(len(predicted)):
                            confusion_matrix[labels_indice[k]][predicted[k]] += 1
                    correct += (predicted == labels_indice).sum().item()
            accuracy = 100 * float(correct) / float(total)
            wandb.log({"Train accuracy (per 10 epochs)": accuracy})
    
    perfromance_eval(confusion_matrix, accuracy, base_model, t_test.shape, f_test.shape, duration_train, PHASE)
    if model != 'VGG':
        save_feature_distribution(feature_dict_train_trainer, 'trainer', PHASE)
        save_feature_distribution(feature_dict_train_tester, 'tester', PHASE)            
 
    # Model save
    PATH = './ALERT/checkpoint/'
    if model == 'RNN' or model == 'CNN+RNN':
        torch.save({
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': [group['lr'] for group in optimizer.param_groups],
        'output_size': base_model.classifier.output_size,
        'lstm_input' : base_model.features_size,
        'lstm_hidden': base_model.hidden_size,
        'lstm_layer': base_model.layer_size,
        'mini_resnet_t': base_model.base_model_t.mini_ResNet,
        'mini_resnet_f': base_model.base_model_f.mini_ResNet
        }, PATH + model+'_'+crop+'_'+tester+'_'+str(sample_size)+'_'+str(epochs)+'_'+str(N_shot)+'_'+str(adapt_epoch)+'_'+str(exp_iteration)+'_'+memo+'.pt')
    elif model == 'ISAViT':
        torch.save({
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': [group['lr'] for group in optimizer.param_groups],
        'output_size': base_model.classifier.output_size,
        'isavit_t': base_model.base_model_t.model,
        'num_patches_t': base_model.base_model_t.num_patches,
        'side_h_t' : base_model.base_model_t.side_h,
        'side_w_t' : base_model.base_model_t.side_w,
        'beta' : base_model.classifier.beta,
        'alpha' : base_model.classifier.alpha
        }, PATH + model+'_'+crop+'_'+tester+'_'+str(sample_size)+'_'+str(epochs)+'_'+str(N_shot)+'_'+str(adapt_epoch)+'_'+str(exp_iteration)+'_'+memo+'.pt')
    else:
        torch.save({
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': [group['lr'] for group in optimizer.param_groups],
        'output_size': base_model.classifier.output_size,
        'beta' : base_model.classifier.beta,
        'alpha' : base_model.classifier.alpha
        }, PATH + model+'_'+crop+'_'+tester+'_'+str(sample_size)+'_'+str(epochs)+'_'+str(N_shot)+'_'+str(adapt_epoch)+'_'+str(exp_iteration)+'_'+memo+'.pt')
    #'''


    '''
    # Model load  Resizing for pretrained model
    PATH = './ALERT/checkpoint/'
    checkpoint = torch.load(PATH + model+'_'+crop+'_'+tester+'_'+str(sample_size)+'_'+str(epochs)+'_'+str(N_shot)+'_'+str(adapt_epoch)+'_'+str(exp_iteration)+'_'+memo+'.pt')
    
    output_size = checkpoint['output_size']
    base_model.classifier.fc1 = nn.Linear(output_size, output_size//2).to(setting.device)
    base_model.classifier.fc2 = nn.Linear(output_size//2, output_size//4).to(setting.device)
    base_model.classifier.fc3 = nn.Linear(output_size//4, setting.num_classes).to(setting.device)
    base_model.classifier.initialized = True
    
    if model == 'RNN' or model == 'CNN+RNN':
        input_size = checkpoint['lstm_input']
        hidden_size = checkpoint['lstm_hidden']
        layer_size = checkpoint['lstm_layer']
        base_model.base_model_t.mini_ResNet = checkpoint['mini_resnet_t']
        base_model.base_model_f.mini_ResNet = checkpoint['mini_resnet_f']
        base_model.lstm.lstm = nn.LSTM(input_size, hidden_size, layer_size, batch_first=True, bidirectional=True).to(setting.device)
        base_model.lstm.initialized = True
    elif model == 'ISAViT':
        base_model.base_model_t.model = checkpoint['isavit_t']
        base_model.base_model_t.num_patches = checkpoint['num_patches_t']     
        base_model.base_model_t.side_h = checkpoint['side_h_t']     
        base_model.base_model_t.side_w = checkpoint['side_w_t']  
        base_model.classifier.beta = checkpoint['beta']
        base_model.classifier.alpha = checkpoint['alpha']  
        base_model.base_model_t.initialized = True
        #base_model.base_model_f.initialized = True
    else:
        base_model.classifier.beta = checkpoint['beta']
        base_model.classifier.alpha = checkpoint['alpha'] 
        
    base_model.load_state_dict(checkpoint['model_state_dict'])
    #base_model.classifier.beta = nn.Parameter(torch.tensor(0.5))
    #base_model.classifier.beta = nn.Parameter(torch.tensor(1.0))
    #base_model.classifier.beta = nn.Parameter(torch.tensor(1.0)) # for beta training
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #print(base_model.state_dict())

    PHASE = 'Before adaptation inference'
    # Before adaptation
    
    confusion_matrix = [[0 for i in range(setting.num_classes)] for j in range(setting.num_classes)]
    correct = 0
    total = 0
    duration_ba = 0
    start_ba = time.time()
    
    base_model.eval()
    with torch.no_grad():
        for t_test, f_test, l_test in test_dataloader:
            t_test, f_test, l_test = t_test.to(setting.device), f_test.to(setting.device), l_test.to(setting.device)
            
            t_test = t_test.unsqueeze(1)
            f_test = f_test.unsqueeze(1)
            
            features, outputs = base_model(t_test.float(), f_test.float())
            labels_indice = torch.argmax(l_test, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += l_test.size(0)
            
            for k in range(len(predicted)):
                confusion_matrix[labels_indice[k]][predicted[k]] += 1
            
            correct += (predicted == labels_indice).sum().item()
            end_ba = time.time()
            duration_ba += end_ba - start_ba
    accuracy = 100 * float(correct) / float(total)
    #print(base_model.state_dict())
    perfromance_eval(confusion_matrix, accuracy, base_model, t_test.shape, f_test.shape, duration_ba, PHASE)     
    '''

''' #Beta training
    # for beta training
    confusion_matrix = [[0 for i in range(setting.num_classes)] for j in range(setting.num_classes)]
    feature_dict_train_trainer = copy.deepcopy(setting.feature_dict_original)
    feature_dict_train_tester = copy.deepcopy(setting.feature_dict_original)
    duration_train = 0
    log_step = 0
    #L = 0
    #early_stopping = EarlyStopping(patience=3, verbose=False)
    base_model.classifier.beta = nn.Parameter(torch.tensor(0.3))
    optimizer = optim.Adam(base_model.parameters(), lr=0.001) # for testing beta
    for epoch in tqdm(range(15), desc=f"Training progress: ", leave=True):
        running_loss = 0.0
        for subject in setting.subject_list:
            if subject != tester:
                time_train = []
                freq_train = []
                labels_train = []
                if subject == 'jj' or subject == 'ks':
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_time_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        if label_idx != setting.label_dict['nod'] and label_idx != setting.label_dict['drink'] and label_idx != setting.label_dict['panel']:
                            time_train += tmp[label_idx]
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_freq_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        if label_idx != setting.label_dict['nod'] and label_idx != setting.label_dict['drink'] and label_idx != setting.label_dict['panel']:
                            freq_train += tmp[label_idx]
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_labels_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        if label_idx != setting.label_dict['nod'] and label_idx != setting.label_dict['drink'] and label_idx != setting.label_dict['panel']:
                            labels_train += tmp[label_idx]
                elif subject == 'sh' or subject == 'ys' or subject == 'yj':
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_time_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        if label_idx != setting.label_dict['smoke']:
                            time_train += tmp[label_idx]
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_freq_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        if label_idx != setting.label_dict['smoke']:
                            freq_train += tmp[label_idx]
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_labels_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        if label_idx != setting.label_dict['smoke']:
                            labels_train += tmp[label_idx]
                else:
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_time_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        time_train += tmp[label_idx]
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_freq_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        freq_train += tmp[label_idx]
                    with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_labels_data.pickle', 'rb') as f:
                        tmp = pickle.load(f)
                    for label_idx in range(len(tmp)):
                        labels_train += tmp[label_idx]
                
                train_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(time_train)), torch.tensor(np.array(freq_train)), torch.tensor(np.array(labels_train)))
                
                base_model.train()
                for param in base_model.parameters():
                    param.requires_grad = False
                # Unfreeze only the classifier parameters
                for param in base_model.classifier.parameters():
                    param.requires_grad = True
                #base_model.classifier.beta.requires_grad = True

                train_dataloader = DataLoader(train_dataset, batch_size=25, shuffle=True)
                start_train = time.time()
                #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
                for step, (t_train, f_train, l_train) in enumerate(train_dataloader):
                    t_train, f_train, l_train = t_train.to(setting.device), f_train.to(setting.device), l_train.to(setting.device)

                    t_train = t_train.unsqueeze(1) #[25,1,51,500]
                    f_train = f_train.unsqueeze(1) #[25,1,89,500]
                    
                    optimizer.zero_grad()
                    labels_indices = torch.argmax(l_train, dim=1)
                    features, outputs = base_model(t_train.float(), f_train.float())
                    
                    if epoch == epochs - 1:
                        features = features.tolist()
                        for feature_idx, labels_idx in enumerate(labels_indices):
                            feature_dict_train_trainer[str(int(labels_idx))].append(features[feature_idx])
                    
                    loss = criterion(outputs, labels_indices)                    
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    running_loss += loss.item()
                    wandb.log({"Beta training loss": loss.item()})      
                    #L = running_loss
                    #early_stopping(L)
                    #if early_stopping.early_stop:
                        #break
                    
                #scheduler.step()    
    PHASE = 'Beta test'
    # Before adaptation
    
    confusion_matrix = [[0 for i in range(setting.num_classes)] for j in range(setting.num_classes)]
    correct = 0
    total = 0
    duration_ba = 0
    start_ba = time.time()
    
    base_model.eval()
    with torch.no_grad():
        for t_test, f_test, l_test in test_dataloader:
            t_test, f_test, l_test = t_test.to(setting.device), f_test.to(setting.device), l_test.to(setting.device)
            
            t_test = t_test.unsqueeze(1)
            f_test = f_test.unsqueeze(1)
            
            features, outputs = base_model(t_test.float(), f_test.float())
            #print(outputs) # 25, 7
            labels_indice = torch.argmax(l_test, dim=1)
            #print(labels_indice) # 25
            _, predicted = torch.max(outputs, 1)
            #print(predicted) # 25 
            total += l_test.size(0)
            #print(labels_indice)
            #print(predicted)
            #print(outputs)
            #################for confidence score of conflicted labels######
            differences = torch.ne(labels_indice, predicted)
            # Get indices of differing elements
            indices = torch.nonzero(differences, as_tuple=True)[0]
            # Output the indices
            #print("Indices of differing elements:", indices)
            #for idx in indices:
                #print("label: ", labels_indice[idx])
                #print("confidence score : ", outputs[idx])
            #################for confidence score of conflicted labels######

            for k in range(len(predicted)):
                confusion_matrix[labels_indice[k]][predicted[k]] += 1
            
            correct += (predicted == labels_indice).sum().item()
            end_ba = time.time()
            duration_ba += end_ba - start_ba
            
    accuracy = 100 * float(correct) / float(total)
    #print(base_model.state_dict())
    perfromance_eval(confusion_matrix, accuracy, base_model, t_test.shape, f_test.shape, duration_ba, PHASE)    
    PATH = './checkpoint/'
    torch.save({
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': [group['lr'] for group in optimizer.param_groups],
        'output_size': base_model.classifier.output_size,
        'vit_no_resize_single_domain_model_t': base_model.base_model_t.model,
        'num_patches_t': base_model.base_model_t.num_patches,
        'side_h_t' : base_model.base_model_t.side_h,
        'side_w_t' : base_model.base_model_t.side_w,
        'beta' : base_model.classifier.beta,
        'alpha' : base_model.classifier.alpha
        #'vit_no_resize_single_domain_model_f': base_model.base_model_f.model,
        #'num_patches_f': base_model.base_model_f.num_patches,
        #'side_h_f' : base_model.base_model_f.side_h,
        #'side_w_f' : base_model.base_model_f.side_w
        }, PATH + model+'_'+crop+'_'+tester+'_'+str(sample_size)+'_'+str(epochs)+'_'+str(N_shot)+'_'+str(adapt_epoch)+'_'+str(exp_iteration)+'_'+memo+'Beta_Test'+'.pt')           
    ''' #Beta training 
    
    
    
    
    """
    PATH = './checkpoint/'
    checkpoint = torch.load(PATH + model+'_'+crop+'_'+tester+'_'+str(sample_size)+'_'+str(epochs)+'_'+str(N_shot)+'_'+str(adapt_epoch)+'_'+str(exp_iteration)+'_'+memo+'Beta_Test'+'.pt')
    
    output_size = checkpoint['output_size']
    base_model.classifier.fc1 = nn.Linear(output_size, output_size//2).to(setting.device)
    base_model.classifier.fc2 = nn.Linear(output_size//2, output_size//4).to(setting.device)
    base_model.classifier.fc3 = nn.Linear(output_size//4, setting.num_classes).to(setting.device)
    base_model.classifier.initialized = True
    base_model.base_model_t.model = checkpoint['vit_no_resize_single_domain_model_t']
    base_model.base_model_t.num_patches = checkpoint['num_patches_t']     
    base_model.base_model_t.side_h = checkpoint['side_h_t']     
    base_model.base_model_t.side_w = checkpoint['side_w_t']  
    base_model.classifier.beta = checkpoint['beta']
    base_model.classifier.alpha = checkpoint['alpha']  
    #base_model.base_model_f.model = checkpoint['vit_no_resize_single_domain_model_f']
    #base_model.base_model_f.num_patches = checkpoint['num_patches_f']     
    #base_model.base_model_f.side_h = checkpoint['side_h_f']     
    #base_model.base_model_f.side_w = checkpoint['side_w_f']   
    base_model.base_model_t.initialized = True
    #base_model.base_model_f.initialized = True
    base_model.load_state_dict(checkpoint['model_state_dict'])
    #base_model.classifier.beta = nn.Parameter(torch.tensor(1.0)) # for beta training
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    PHASE = 'Beta test'
    confusion_matrix = [[0 for i in range(setting.num_classes)] for j in range(setting.num_classes)]
    correct = 0
    total = 0
    duration_ba = 0
    start_ba = time.time()
    
    base_model.eval()
    with torch.no_grad():
        for t_test, f_test, l_test in test_dataloader:
            t_test, f_test, l_test = t_test.to(setting.device), f_test.to(setting.device), l_test.to(setting.device)
            
            t_test = t_test.unsqueeze(1)
            f_test = f_test.unsqueeze(1)
            
            features, outputs = base_model(t_test.float(), f_test.float())
            #print(outputs) # 25, 7
            labels_indice = torch.argmax(l_test, dim=1)
            #print(labels_indice) # 25
            _, predicted = torch.max(outputs, 1)
            #print(predicted) # 25 
            total += l_test.size(0)
            #print(labels_indice)
            #print(predicted)
            #print(outputs)
            #################for confidence score of conflicted labels######
            differences = torch.ne(labels_indice, predicted)
            # Get indices of differing elements
            indices = torch.nonzero(differences, as_tuple=True)[0]
            # Output the indices
            #print("Indices of differing elements:", indices)
            #for idx in indices:
                #print("label: ", labels_indice[idx])
                #print("confidence score : ", outputs[idx])
            #################for confidence score of conflicted labels######

            for k in range(len(predicted)):
                confusion_matrix[labels_indice[k]][predicted[k]] += 1
            
            correct += (predicted == labels_indice).sum().item()
            end_ba = time.time()
            duration_ba += end_ba - start_ba
            
    accuracy = 100 * float(correct) / float(total)
    #print(base_model.state_dict())
    perfromance_eval(confusion_matrix, accuracy, base_model, t_test.shape, f_test.shape, duration_ba, PHASE)    
    """

    
    
    '''
    PHASE = 'After adaptation inference'
    # After adaptation
    if model == 'VGG':
        optimizer = optim.Adam(base_model.parameters(), lr=0.00001) # for VGG
    elif model == 'ViT_no_resize_single_domain' or model == 'ISAViT':
        optimizer = optim.Adam(base_model.parameters(), lr=0.00001) 
    

    adapt_dataloader = DataLoader(adapt_dataset, batch_size=5, shuffle=True)
    
    base_model.train()
    #for param in base_model.parameters():
        #param.requires_grad = True
    #base_model.classifier.beta.requires_grad = False

    feature_dict_adapt_trainer = copy.deepcopy(setting.feature_dict_original)
    feature_dict_adapt_tester = copy.deepcopy(setting.feature_dict_original)
    duration_adapt = 0
    start_adapt = time.time()
    for epoch in range(10):
        running_loss = 0.0
        for step, (t_adapt, f_adapt, l_adapt) in enumerate(adapt_dataloader):
            t_adapt, f_adapt, l_adapt = t_adapt.to(setting.device), f_adapt.to(setting.device), l_adapt.to(setting.device)
            ##
            t_adapt = t_adapt.unsqueeze(1)
            f_adapt = f_adapt.unsqueeze(1)
            ##
            optimizer.zero_grad()
            labels_indices = torch.argmax(l_adapt, dim=1)
            features, outputs = base_model(t_adapt.float(), f_adapt.float())
            
            loss = criterion(outputs, labels_indices)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()   
            wandb.log({"Adaptation loss (per iteration)": loss.item()})      
            end_adapt = time.time()
            duration_adapt += end_adapt - start_adapt
    
    # Testing after adaptation
    base_model.eval()
    confusion_matrix = [[0 for i in range(setting.num_classes)] for j in range(setting.num_classes)]
    correct = 0.0
    total = 0.0
    duration_test = 0
    start_test = time.time()
    with torch.no_grad():
        for i, (t_test, f_test, l_test) in enumerate(test_dataloader):
            t_test, f_test, l_test = t_test.to(setting.device), f_test.to(setting.device), l_test.to(setting.device)
            ##
            t_test = t_test.unsqueeze(1)
            f_test = f_test.unsqueeze(1)
            ##
            features, outputs = base_model(t_test.float(), f_test.float())
            labels_indice = torch.argmax(l_test, dim=1)
            _, predicted = torch.max(outputs, 1)
            if epoch == adapt_epoch - 1:
                features = features.tolist()
                for feature_idx, labels_idx in enumerate(labels_indice):
                    feature_dict_adapt_tester[str(int(labels_idx))].append(features[feature_idx])
            total += l_test.size(0)
            for k in range(len(predicted)):
                confusion_matrix[labels_indice[k]][predicted[k]] += 1
            correct += (predicted == labels_indice).sum().item()
            end_test = time.time()
            duration_test += end_test - start_test
    accuracy = 100 * correct / total
    perfromance_eval(confusion_matrix, accuracy, base_model, t_test.shape, f_test.shape, duration_test, PHASE) 

    # PATH = './checkpoint/'
    # torch.save({
    #     'model_state_dict': base_model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'learning_rate': [group['lr'] for group in optimizer.param_groups],
    #     'output_size': base_model.classifier.output_size,
    #     'vit_no_resize_single_domain_model_t': base_model.base_model_t.model,
    #     'num_patches_t': base_model.base_model_t.num_patches,
    #     'side_h_t' : base_model.base_model_t.side_h,
    #     'side_w_t' : base_model.base_model_t.side_w,
    #     'beta' : base_model.classifier.beta,
    #     'alpha' : base_model.classifier.alpha
    #     #'vit_no_resize_single_domain_model_f': base_model.base_model_f.model,
    #     #'num_patches_f': base_model.base_model_f.num_patches,
    #     #'side_h_f' : base_model.base_model_f.side_h,
    #     #'side_w_f' : base_model.base_model_f.side_w
    #     }, PATH + model+'_'+crop+'_'+tester+'_'+str(sample_size)+'_'+str(epochs)+'_'+str(N_shot)+'_'+str(adapt_epoch)+'_'+str(exp_iteration)+'_'+memo+'AA'+'.pt')           
    #'''

    """
    PATH = './checkpoint/'
    checkpoint = torch.load(PATH + model+'_'+crop+'_'+tester+'_'+str(sample_size)+'_'+str(epochs)+'_'+str(N_shot)+'_'+str(adapt_epoch)+'_'+str(exp_iteration)+'_'+memo+'AA'+'.pt')
    
    output_size = checkpoint['output_size']
    base_model.classifier.fc1 = nn.Linear(output_size, output_size//2).to(setting.device)
    base_model.classifier.fc2 = nn.Linear(output_size//2, output_size//4).to(setting.device)
    base_model.classifier.fc3 = nn.Linear(output_size//4, setting.num_classes).to(setting.device)
    base_model.classifier.initialized = True
    base_model.base_model_t.model = checkpoint['vit_no_resize_single_domain_model_t']
    base_model.base_model_t.num_patches = checkpoint['num_patches_t']     
    base_model.base_model_t.side_h = checkpoint['side_h_t']     
    base_model.base_model_t.side_w = checkpoint['side_w_t']  
    base_model.classifier.beta = checkpoint['beta']
    base_model.classifier.alpha = checkpoint['alpha']  
    #base_model.base_model_f.model = checkpoint['vit_no_resize_single_domain_model_f']
    #base_model.base_model_f.num_patches = checkpoint['num_patches_f']     
    #base_model.base_model_f.side_h = checkpoint['side_h_f']     
    #base_model.base_model_f.side_w = checkpoint['side_w_f']   
    base_model.base_model_t.initialized = True
    #base_model.base_model_f.initialized = True
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.classifier.beta = nn.Parameter(torch.tensor(1.0)) # for beta training
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    # Testing after adaptation
    base_model.eval()
    confusion_matrix = [[0 for i in range(setting.num_classes)] for j in range(setting.num_classes)]
    correct = 0.0
    total = 0.0
    duration_test = 0
    start_test = time.time()
    with torch.no_grad():
        for i, (t_test, f_test, l_test) in enumerate(test_dataloader):
            t_test, f_test, l_test = t_test.to(setting.device), f_test.to(setting.device), l_test.to(setting.device)
            ##
            t_test = t_test.unsqueeze(1)
            f_test = f_test.unsqueeze(1)
            ##
            features, outputs = base_model(t_test.float(), f_test.float())
            labels_indice = torch.argmax(l_test, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            
            total += l_test.size(0)
            for k in range(len(predicted)):
                confusion_matrix[labels_indice[k]][predicted[k]] += 1
            correct += (predicted == labels_indice).sum().item()
            end_test = time.time()
            duration_test += end_test - start_test
    accuracy = 100 * correct / total
    perfromance_eval(confusion_matrix, accuracy, base_model, t_test.shape, f_test.shape, duration_test, 'adapt_test') 


    # After adaptation
    if model == 'VGG':
        optimizer = optim.Adam(base_model.parameters(), lr=0.00001) # for VGG
    elif model == 'ViT_no_resize_single_domain' or model == 'ISAViT':
        optimizer = optim.Adam(base_model.parameters(), lr=0.001) 
    

    adapt_dataloader = DataLoader(adapt_dataset, batch_size=5, shuffle=True)
    
    base_model.train()
    for param in base_model.parameters():
        param.requires_grad = False
    # Unfreeze only the classifier parameters
    for param in base_model.classifier.parameters():
        param.requires_grad = True
    

    feature_dict_adapt_trainer = copy.deepcopy(setting.feature_dict_original)
    feature_dict_adapt_tester = copy.deepcopy(setting.feature_dict_original)
    duration_adapt = 0
    start_adapt = time.time()
    for epoch in range(50):
        running_loss = 0.0
        for step, (t_adapt, f_adapt, l_adapt) in enumerate(adapt_dataloader):
            t_adapt, f_adapt, l_adapt = t_adapt.to(setting.device), f_adapt.to(setting.device), l_adapt.to(setting.device)
            ##
            t_adapt = t_adapt.unsqueeze(1)
            f_adapt = f_adapt.unsqueeze(1)
            ##
            optimizer.zero_grad()
            labels_indices = torch.argmax(l_adapt, dim=1)
            features, outputs = base_model(t_adapt.float(), f_adapt.float())
            
            loss = criterion(outputs, labels_indices)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()   
            wandb.log({"Adaptation loss (per iteration)": loss.item()})      
            end_adapt = time.time()
            duration_adapt += end_adapt - start_adapt
    
    # Testing after adaptation
    base_model.eval()
    confusion_matrix = [[0 for i in range(setting.num_classes)] for j in range(setting.num_classes)]
    correct = 0.0
    total = 0.0
    duration_test = 0
    start_test = time.time()
    with torch.no_grad():
        for i, (t_test, f_test, l_test) in enumerate(test_dataloader):
            t_test, f_test, l_test = t_test.to(setting.device), f_test.to(setting.device), l_test.to(setting.device)
            ##
            t_test = t_test.unsqueeze(1)
            f_test = f_test.unsqueeze(1)
            ##
            features, outputs = base_model(t_test.float(), f_test.float())
            labels_indice = torch.argmax(l_test, dim=1)
            _, predicted = torch.max(outputs, 1)
            if epoch == adapt_epoch - 1:
                features = features.tolist()
                for feature_idx, labels_idx in enumerate(labels_indice):
                    feature_dict_adapt_tester[str(int(labels_idx))].append(features[feature_idx])
            total += l_test.size(0)
            for k in range(len(predicted)):
                confusion_matrix[labels_indice[k]][predicted[k]] += 1
            correct += (predicted == labels_indice).sum().item()
            end_test = time.time()
            duration_test += end_test - start_test
    accuracy = 100 * correct / total
    perfromance_eval(confusion_matrix, accuracy, base_model, t_test.shape, f_test.shape, duration_test, 'beta_AA') 
    #"""








    # For feature distribution for trainer after adaptation
    '''
    for subject in setting.subject_list:
        if subject != setting.tester:
            time_train = []
            freq_train = []
            labels_train = []
            if subject == 'jj' or subject == 'ks':
                with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_time_data.pickle', 'rb') as f:
                    tmp = pickle.load(f)
                for label_idx in range(len(tmp)):
                    if label_idx != setting.label_dict['nod'] and label_idx != setting.label_dict['drink']:
                        time_train += tmp[label_idx]
                with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_freq_data.pickle', 'rb') as f:
                    tmp = pickle.load(f)
                for label_idx in range(len(tmp)):
                    if label_idx != setting.label_dict['nod'] and label_idx != setting.label_dict['drink']:
                        freq_train += tmp[label_idx]
                with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_labels_data.pickle', 'rb') as f:
                    tmp = pickle.load(f)
                for label_idx in range(len(tmp)):
                    if label_idx != setting.label_dict['nod'] and label_idx != setting.label_dict['drink']:
                        labels_train += tmp[label_idx]
            else:
                with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_time_data.pickle', 'rb') as f:
                    tmp = pickle.load(f)
                for label_idx in range(len(tmp)):
                    time_train += tmp[label_idx]
                with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_freq_data.pickle', 'rb') as f:
                    tmp = pickle.load(f)
                for label_idx in range(len(tmp)):
                    freq_train += tmp[label_idx]
                with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(setting.num_classes)+'_labels_data.pickle', 'rb') as f:
                    tmp = pickle.load(f)
                for label_idx in range(len(tmp)):
                    labels_train += tmp[label_idx]
            
            train_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(time_train)), torch.tensor(np.array(freq_train)), torch.tensor(np.array(labels_train)))
            
            base_model.eval()
                    
            train_dataloader = DataLoader(train_dataset, batch_size=25, shuffle=True)
            with torch.no_grad():
                for step, (t_train, f_train, l_train) in enumerate(train_dataloader):
                    t_train, f_train, l_train = t_train.to(setting.device), f_train.to(setting.device), l_train.to(setting.device)

                    t_train = t_train.unsqueeze(1) #[25,1,51,500]
                    f_train = f_train.unsqueeze(1) #[25,1,89,500]
                    
                    optimizer.zero_grad()
                    labels_indices = torch.argmax(l_train, dim=1)
                    features, outputs = base_model(t_train.float(), f_train.float())

                    features = features.tolist()
                    for feature_idx, labels_idx in enumerate(labels_indices):
                        feature_dict_adapt_trainer[str(int(labels_idx))].append(features[feature_idx])
    if model != 'VGG':
        save_feature_distribution(feature_dict_adapt_trainer, 'trainer', PHASE)
        save_feature_distribution(feature_dict_adapt_tester, 'tester', PHASE)  
    '''