import os
import sys
import pandas as pd
import numpy as np
import torch
import glob
import pickle
from torch.utils.data import Dataset

class ExtendDataset(Dataset):
    #def __init__(self, time_file, freq_file, label_file, sample_size):
    def __init__(self, time_file, freq_file, sample_size): # sample_size is observation length (4 s)
        # Load the data and labels from CSV files
        self.time = []
        self.freq = []
        self.labels = []
        self.time_query = []
        self.freq_query = []
        self.labels_query = []
        self.relax_count = 0
        self.sw_count = 0
        self.phone_count = 0
        self.label_idx = 0
        #for time_path, freq_path, label_path in zip(time_file, freq_file, label_file):
        loc_count = 0
        for task in subject_list:
            key_task = 'ALERT_train/'+task+'relax1_t.csv'
            loc = time_file.index(key_task)
            tmp_t = time_file[loc_count]
            tmp_f = freq_file[loc_count]
            time_file[loc_count] = time_file[loc]
            freq_file[loc_count] = freq_file[loc]
            time_file[loc] = tmp_t
            freq_file[loc] = tmp_f
            loc_count += 1
        #for time_path, freq_path, label_path in zip(time_file, freq_file, label_file):
        for time_path, freq_path in zip(time_file, freq_file):
            time_df = pd.read_csv(time_path, header=None)
            freq_df = pd.read_csv(freq_path, header=None)
            #label_df = pd.read_csv(label_path, header=None)
            time_mat = time_df.values
            freq_mat = freq_df.values
            #label_mat = label_df.values
            time_mat = time_mat[1:, 1:]
            freq_mat = freq_mat[1:, 1:]
            # Crop the data recursively as a matrix of row X 25
            n_samples = time_mat.shape[1] // sample_size
            n_samples = n_samples * 2 
            for i in range(n_samples):  
                if i <= n_samples-2:
                    tmp = []
                    for key in label_dict:
                        if key in time_path:
                            for j in range(num_classes):
                                if j == label_dict[key]:
                                    self.label_idx = j
                                    tmp.append(1)
                                else:
                                    tmp.append(0)
                            #self.labels.append(tmp)
                            subject_labels[task][self.label_idx].append(tmp)
                    #self.time.append(time_mat[rangebin_start:rangebin_end, i*sample_size:(i+1)*sample_size])
                            if CA_dict[task] == 0:
                                max = 0
                                ca = 0
                                for row in range(170):
                                    if time_mat[row][0] > max:
                                        max = time_mat[row][0]
                                        ca = row
                                CA_dict[task] = ca
                            
                            if sys.argv[2] == 'CA':
                                #self.freq.append(freq_mat[0:177, i*sample_size:(i+1)*sample_size])
                                subject_time[task][self.label_idx].append(time_mat[(CA_dict[task]-10):(CA_dict[task]+5), int(i*sample_size/2):int(((i+1)*sample_size/2)+(sample_size/2))])
                                subject_freq[task][self.label_idx].append(freq_mat[0:89, int(i*sample_size/2):int(((i+1)*sample_size/2)+(sample_size/2))])
                            elif sys.argv[2] == 'CA_fft':
                                #self.freq.append(freq_mat[0:177, i*sample_size:(i+1)*sample_size])
                                subject_time[task][self.label_idx].append(time_mat[(CA_dict[task]-10):(CA_dict[task]+5), int(i*sample_size/2):int(((i+1)*sample_size/2)+(sample_size/2))])
                                #subject_freq[task][self.label_idx].append(freq_mat[0:89, i*sample_size:(i+1)*sample_size])
                                fft = np.fft.fft(time_mat[(CA_dict[task]-10):(CA_dict[task]+5), int(i*sample_size/2):int(((i+1)*sample_size/2)+(sample_size/2))], axis=1)
                                subject_freq[task][self.label_idx].append(abs(fft))
                            elif sys.argv[2] == 'cropX':
                                #self.freq.append(freq_mat[rangebin_start:rangebin_end, i*sample_size:(i+1)*sample_size])
                                subject_time[task][self.label_idx].append(time_mat[rangebin_start:rangebin_end, int(i*sample_size/2):int(((i+1)*sample_size/2)+(sample_size/2))])
                                subject_freq[task][self.label_idx].append(freq_mat[:, int(i*sample_size/2):int(((i+1)*sample_size/2)+(sample_size/2))])
                            elif sys.argv[2] == 'cropO':
                                subject_time[task][self.label_idx].append(time_mat[rangebin_start:rangebin_end, int(i*sample_size/2):int(((i+1)*sample_size/2)+(sample_size/2))])
                                subject_freq[task][self.label_idx].append(freq_mat[0:89, int(i*sample_size/2):int(((i+1)*sample_size/2)+(sample_size/2))])
                        
            
 

class MyDataset(Dataset):
    #def __init__(self, time_file, freq_file, label_file, sample_size):
    def __init__(self, time_file, freq_file, sample_size): # sample_size is observation length (4 s)
        # Load the data and labels from CSV files
        self.time = []
        self.freq = []
        self.labels = []
        self.time_query = []
        self.freq_query = []
        self.labels_query = []
        self.relax_count = 0
        self.sw_count = 0
        self.phone_count = 0
        self.label_idx = 0
        #for time_path, freq_path, label_path in zip(time_file, freq_file, label_file):
        loc_count = 0
        for task in subject_list:
            key_task = 'ALERT_train/'+task+'relax1_t.csv'
            loc = time_file.index(key_task)
            tmp_t = time_file[loc_count]
            tmp_f = freq_file[loc_count]
            time_file[loc_count] = time_file[loc]
            freq_file[loc_count] = freq_file[loc]
            time_file[loc] = tmp_t
            freq_file[loc] = tmp_f
            loc_count += 1

        for time_path, freq_path in zip(time_file, freq_file):

            time_df = pd.read_csv(time_path, header=None)
            freq_df = pd.read_csv(freq_path, header=None)
        
            time_mat = time_df.values
            freq_mat = freq_df.values
            
            time_mat = time_mat[1:, 1:]
            freq_mat = freq_mat[1:, 1:]
            # Crop the data recursively as a matrix of row X 25
            n_samples = time_mat.shape[1] // sample_size
            for task in subject_list:
                if task == time_path[11:13]:
                    for i in range(n_samples):
                        tmp = []
                        for key in label_dict:
                            if key in time_path:
                                for j in range(num_classes):
                                    if j == label_dict[key]:
                                        self.label_idx = j
                                        tmp.append(1)
                                    else:
                                        tmp.append(0)
                                #self.labels.append(tmp)
                                subject_labels[task][self.label_idx].append(tmp)
                        #self.time.append(time_mat[rangebin_start:rangebin_end, i*sample_size:(i+1)*sample_size])
                                if CA_dict[task] == 0:
                                    max = 0
                                    ca = 0
                                    for row in range(170):
                                        if time_mat[row][0] > max:
                                            max = time_mat[row][0]
                                            ca = row
                                    CA_dict[task] = ca
                                
                                if sys.argv[2] == 'CA':
                                    #self.freq.append(freq_mat[0:177, i*sample_size:(i+1)*sample_size])
                                    subject_time[task][self.label_idx].append(time_mat[(CA_dict[task]-10):(CA_dict[task]+5), i*sample_size:(i+1)*sample_size])
                                    subject_freq[task][self.label_idx].append(freq_mat[0:89, i*sample_size:(i+1)*sample_size])
                                elif sys.argv[2] == 'CA_fft':
                                    #self.freq.append(freq_mat[0:177, i*sample_size:(i+1)*sample_size])
                                    subject_time[task][self.label_idx].append(time_mat[(CA_dict[task]-10):(CA_dict[task]+5), i*sample_size:(i+1)*sample_size])
                                    #subject_freq[task][self.label_idx].append(freq_mat[0:89, i*sample_size:(i+1)*sample_size])
                                    fft = np.fft.fft(time_mat[(CA_dict[task]-10):(CA_dict[task]+5), i*sample_size:(i+1)*sample_size], axis=1)
                                    subject_freq[task][self.label_idx].append(abs(fft))
                                elif sys.argv[2] == 'cropX':
                                    #self.freq.append(freq_mat[rangebin_start:rangebin_end, i*sample_size:(i+1)*sample_size])
                                    subject_time[task][self.label_idx].append(time_mat[rangebin_start:rangebin_end, i*sample_size:(i+1)*sample_size])
                                    subject_freq[task][self.label_idx].append(freq_mat[0:89, i*sample_size:(i+1)*sample_size])
                                elif sys.argv[2] == 'cropO':
                                    subject_time[task][self.label_idx].append(time_mat[rangebin_start:rangebin_end, i*sample_size:(i+1)*sample_size])
                                    subject_freq[task][self.label_idx].append(freq_mat[0:89, i*sample_size:(i+1)*sample_size]) #:45, 45:89


def usage_exam():
    print("===============================================================================================================\n")
    print("                                 USAGE EXAMPLE\n")
    print("===============================================================================================================\n")
    print("Please write down the exact command")
    print("\n")
    print("   EX)     python3 makeDataset.py common/extend cropO/cropX/CA sample_size")
    print("\n")
    print("===============================================================================================================\n")
    sys.exit()


def make_pickle(crop, sample_size, num_classes):
    for subject in subject_list:
        with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(num_classes)+'_time_data.pickle', 'wb') as f:
            pickle.dump(subject_time[subject], f)
        with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(num_classes)+'_freq_data.pickle', 'wb') as f:
            pickle.dump(subject_freq[subject], f)
        with open('./pickles/'+crop+'_'+subject+'_'+str(sample_size)+'_'+str(num_classes)+'_labels_data.pickle', 'wb') as f:
            pickle.dump(subject_labels[subject], f)
    
   
if __name__ == "__main__":
    label_dict = {'relax' : 0, 'sw': 1, 'nod':2, 'drink': 3, 'phone': 4, 'rsmoke': 5, 'panel': 6}
    num_classes = len(label_dict) 
    CA_dict = {'dk':0, 'jj':0, 'ks':0, 'sh':0, 'wh':0, 'yj':0, 'ys':0, 'yh':0, 'jp':0}
    subject_list = ['dk', 'jj', 'ks', 'sh', 'wh', 'yj', 'ys', 'yh', 'jp']
    subject_time = {'dk' : [[] for i in range(num_classes)], 'jj': [[] for i in range(num_classes)], 'ks': [[] for i in range(num_classes)], 'sh': [[] for i in range(num_classes)], 'wh':[[] for i in range(num_classes)], 'yj': [[] for i in range(num_classes)], 'ys': [[] for i in range(num_classes)], 'yh':[[] for i in range(num_classes)], 'jp':[[] for i in range(num_classes)]}
    subject_freq = {'dk' : [[] for i in range(num_classes)], 'jj': [[] for i in range(num_classes)], 'ks': [[] for i in range(num_classes)], 'sh': [[] for i in range(num_classes)], 'wh':[[] for i in range(num_classes)], 'yj': [[] for i in range(num_classes)], 'ys': [[] for i in range(num_classes)], 'yh':[[] for i in range(num_classes)], 'jp':[[] for i in range(num_classes)]}
    subject_labels = {'dk' : [[] for i in range(num_classes)], 'jj': [[] for i in range(num_classes)], 'ks': [[] for i in range(num_classes)], 'sh': [[] for i in range(num_classes)], 'wh':[[] for i in range(num_classes)], 'yj': [[] for i in range(num_classes)], 'ys': [[] for i in range(num_classes)], 'yh':[[] for i in range(num_classes)], 'jp':[[] for i in range(num_classes)]}
    sample_size = int(sys.argv[3]) # data sample size (observation length) make it 400 later
    
    # Choosing crop or not
    try:
        if sys.argv[2] == 'cropO': # range cropping (about 2.7 m)
            # Only apply to temporal data
            rangebin_start = 120 
            rangebin_end = 171 
        elif sys.argv[2] == 'cropX': # range 0:177 (about 9 m), frequency 0:89
            rangebin_start = 0 
            rangebin_end = 177 
        elif sys.argv[2] == 'CA': # up to users
            rangebin_start = 140
            rangebin_end = 171
        elif sys.argv[2] == 'CA_fft': # up to users
            rangebin_start = 140
            rangebin_end = 172
        else:
                usage_exam()
    except:
        usage_exam()
    
    # Make file list
    time_files = [] 
    freq_files = []

    # Load files into file list
    for name in glob.glob("ALERT_train/*_t.csv"):
        time_files.append(name)
        freq_files.append(name[:-6]+"_f.csv")

    # Choosing data augmentation or not
    #try: 
    if sys.argv[1] == 'common':
        MyDataset(time_files, freq_files, sample_size)
        #MyDataset(time_files_test, freq_files_test, sample_size)

    elif sys.argv[1] == 'extend':
        dataset = ExtendDataset(time_files, freq_files, sample_size)
        dataset_test = MyDataset(time_files_test, freq_files_test, sample_size)
    '''
        else:
            usage_exam()
    except:
        usage_exam()
    '''
    # NPization for each data
    '''
    time = np.array(dataset.time)
    freq = np.array(dataset.freq)
    labels = np.array(dataset.labels)
    time_test = np.array(dataset_test.time)
    freq_test = np.array(dataset_test.freq)
    labels_test = np.array(dataset_test.labels)
    
    time_query = np.array(dataset_test.time_query)
    freq_query = np.array(dataset_test.freq_query)
    labels_query = np.array(dataset_test.labels_query)
    '''
    #try:
    
    make_pickle(sys.argv[2], sample_size, num_classes)    

    
    

