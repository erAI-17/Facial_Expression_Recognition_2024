import cv2
import matplotlib.pyplot as plt
from vedo import Plotter, load
import vtk
from vtkmodules.util import numpy_support
import numpy as np
import os
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import shutil

def read_CalD3rMenD3s(data):
    #!read all datasets and create unique annotation file where each row has schema [subj_id, code, label, add]
    datasets = ['CalD3r', 'MenD3s']
    emotions = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'neutral':4, 'sadness':5, 'surprise':6}
    
    #read data
    for dataset in datasets:
        path = f'../Datasets/CalD3RMenD3s/{dataset}'
        for emotion in emotions.keys():
            files_path = f'{path}/{emotion.capitalize()}/RGB'
            for filename in os.listdir(files_path):
                #remove "aligned_" from filename
                info_filename = filename.replace('aligned_', '')
                gender = info_filename.split("_")[0] 
                subj_id = info_filename.split("_")[1]   
                code = info_filename.split("_")[2]
                label = info_filename.split("_")[3]
                
                new_entry = [dataset, subj_id, label, emotions[label], '-', '-', code, gender]
                if new_entry not in data: #avoid duplicates (same sample with different modalities)
                    data.append(new_entry)   
    
    return data  




def read_BU3DFE(data):
    #!read all datasets and create unique annotation file where each row has schema [subj_id, code, label, add]
    emotions = {'AN': 0, 'DI': 1, 'FE': 2, 'HA': 3, 'NE': 4, 'SA': 5, 'SU': 6}
    full_emot = {'AN': 'anger', 'DI': 'disgust', 'FE': 'fear', 'HA': 'happiness', 'NE': 'neutral', 'SA': 'sadness', 'SU': 'surprise'}
    
    #read data
    path = f'../Datasets/BU3DFE/Subjects'
    for subject in os.listdir(path): 
        for filename in os.listdir(path + '/' + subject):
            if filename.endswith('_F2D.bmp'):    
                subj_id = filename.split("_")[0]   
                label = filename.split("_")[1][:2]
                intensity = filename.split("_")[1][2:4]
                race = filename.split("_")[1][4:6]
                
                new_entry = ['BU3DFE', subj_id, full_emot[label], emotions[label], intensity, race, '-', '-']
                if new_entry not in data: #avoid duplicates (same sample with different modalities)
                    data.append(new_entry) 
    
    return data                 



def read_Bosphorus(data):
    #!read all datasets and create unique annotation file where each row has schema [subj_id, code, label, add]
    emotions = {'ANGER': 0, 'DISGUST': 1, 'FEAR': 2, 'HAPPY': 3, 'NEUTRAL': 4, 'SADNESS': 5, 'SURPRISE': 6}
    full_emot = {'ANGER': 'anger', 'DISGUST': 'disgust', 'FEAR': 'fear', 'HAPPY': 'happiness', 'NEUTRAL': 'neutral', 'SADNESS': 'sadness', 'SURPRISE': 'surprise'}
    
    #read data
    path = f'../Datasets/Bosphorus/Subjects'
    for subject in os.listdir(path): 
        for filename in os.listdir(path + '/' + subject):
            if filename.endswith('rgb.png'):
                subj_id = filename.split("_")[0] + '_' + filename.split("_")[2]  
                label = filename.split("_")[1]
                
                new_entry = ['Bosphorus', subj_id, full_emot[label], emotions[label],  '-', '-', '-', '-']
                if new_entry not in data: #avoid duplicates (same sample with different modalities)
                    data.append(new_entry) 
                    
    return data  
 
                    

def global_annotations():
    #!read all datasets and create unique annotation file where each row has schema [subj_id, code, label, add]
    emotions = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'neutral':4, 'sadness':5, 'surprise':6}
    class_distribution = {emotion: 0 for emotion in emotions.keys()}
    
    data = []  
    datasets = ['CalD3rMenD3s', 'BU3DFE', 'Bosphorus']
    for dataset in datasets:
        if dataset == 'BU3DFE':
            data = read_BU3DFE(data)  
        elif dataset == 'Bosphorus':
            data = read_Bosphorus(data)
        elif dataset == 'CalD3rMenD3s':
            data = read_CalD3rMenD3s(data)
            
        # Debug: print number of entries after each dataset is processed
        print(f"Entries after {dataset}: {len(data)}")
            
                    
    #count class distribution
    for entry in data:
        class_distribution[entry[2]] += 1
                             
    #convert to dataframe
    complete_df = pd.DataFrame(data, columns=['dataset', 'subj_id', 'description_label', 'label', 'intensity', 'race', 'code', 'gender'])
    
    #save annotation train file
    annotation_file = os.path.join('../Datasets/Global', 'annotations_complete.pkl')
    with open(annotation_file, 'wb') as file:
        pickle.dump(complete_df, file)
    
    return class_distribution 

#!##
#!MAIN
#!##
if __name__ == '__main__':

    #!generate annotation files for each dataset, TEST and TRAIN
    class_distribution = global_annotations() #20% test, 80% train

    plt.bar(class_distribution.keys(), class_distribution.values(), color='skyblue', alpha=0.8)
    #Set the y-axis limit
    plt.ylim(0, 10000)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Distribution of Classes')
    #Add values on top of each bar
    for i, (key, value) in enumerate(class_distribution.items()):
        plt.text(i, value, str(value), ha='center', va='bottom')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()
