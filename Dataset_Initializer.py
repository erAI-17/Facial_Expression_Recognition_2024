import numpy as np
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import cv2

def train_test_annotations(test_size):
    """Splits dataset into TRAIN and TEST splits and generate annotation .pkl files where a row represents a sample with this schema:
    dataset (str): CalD3r, MenD3s
    subj_id (str): unique code
    code (str): same subj_id for same label, may have multiple samples
    label (str): anger, surprise,...
    add (list(str, str)): list storing additional info ( gender, pose,...) 
    """
    #!read all datasets and create unique annotation file where each row has schema [subj_id, code, label, add]
    datasets = ['CalD3r', 'MenD3s']
    emotions = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'neutral':4, 'sadness':5, 'surprise':6}
    grouped = {}
    class_distribution = {'Color':{emotion: 0 for emotion in emotions.keys()}, 'Depth':{emotion: 0 for emotion in emotions.keys()}}
    
    # Initialize accumulators for mean and std
    mean = {'Color': np.zeros(3), 'Depth': np.zeros(3)}
    std = {'Color': np.zeros(3), 'Depth': np.zeros(3)}
    sum_pix = {'Color': np.zeros(3), 'Depth': np.zeros(3)}
    sum_sq_pix = {'Color': np.zeros(3), 'Depth': np.zeros(3)}
    n_pix = {'Color': 0, 'Depth': 0}
    data = []
    
    max_depth = 0
    for dataset in datasets:
        path = f'../Datasets/{dataset}'
        
        for emotion in emotions.keys(): 
            for m in ['Color', 'Depth']:
                
                if m == 'Color':
                    files_path = f'{path}/{emotion.capitalize()}/RGB'
                else:
                    files_path = f'{path}/{emotion.capitalize()}/DEPTH'
                
                for filename in os.listdir(files_path): 
                    add = [filename.split("_")[0]] 
                    subj_id = filename.split("_")[1]   
                    code = filename.split("_")[2]
                    description_label = filename.split("_")[3]
                    
                    
                    new_entry = [dataset, subj_id, code, description_label, emotions[description_label], add]
                    if new_entry not in data: #avoid duplicates (same sample with different modalities)
                        data.append([dataset, subj_id, code, description_label, emotions[description_label], add])

                    #!update class distribution
                    class_distribution[m][description_label] += 1
                  
                    #!load image
                    img_path = f'{files_path}/{filename}'
                    if m == 'Color':
                        img = cv2.imread(img_path)
                        # Convert the image from BGR to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        img = img / 255.0  # Normalize to [0, 1]
                        
                        #!update mean and std
                        sum_pix[m] += np.sum(img, axis=(0, 1)) #sum all pixels in the image, separately for each channel (black pixels are 0)
                        sum_sq_pix[m] += np.sum(img ** 2, axis=(0, 1))
                       
                        # Create a mask to maintain only pixels where all three channels are below the threshold
                        mask = (img[:, :, 0] > 0) | (img[:, :, 1] >  0) | (img[:, :, 2] > 0)
                        #mask off 0 values (black pixels) in the frame, from each channel
                        img = img[mask]
                        n_pix[m] += img.shape[0] #mask converts into a 2D array
                        
                    elif m == 'Depth':
                        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                        img = np.expand_dims(img, axis=-1)
                        img = np.repeat(img, 3, axis=-1)
                        
                        max_depth = max(max_depth, img.max()) #!get max depth before normalization
                        
                        img = img / 9785.0  # Normalize to [0, 1] using max_depth=9785
                        
                        #!update mean and std
                        sum_pix[m] += np.sum(img, axis=(0, 1))
                        sum_sq_pix[m] += np.sum(img ** 2, axis=(0, 1))
                        
                        # Create a mask for each channel where pixel values are greater than the threshold=0 (to avoid balck frame pixels)
                        mask = (img[:, :, 0] > 0) | (img[:, :, 1] >  0) | (img[:, :, 2] > 0)
                        #mask off 0 values (black pixels) in the frame, from each channel
                        img = img[mask]
                        n_pix[m] += img.shape[0] #mask converts into a 2D array
                                        
    # Average mean and std over the number of samples
    for m in ['Color', 'Depth']:
        mean[m] = sum_pix[m]/n_pix[m]
        std[m] = np.sqrt(sum_sq_pix[m] / n_pix[m] - mean[m] ** 2)

            
    #!split data into train and test dataframes (making sure that all the samples with same subj_id, label and add fall inside the same split)
    for sample in data:
        key = (sample[0], sample[1], sample[2], sample[3], sample[4], tuple(sample[5])) #?#key: (dataset, subj_id, code, description_label, label)
        grouped.setdefault(key, []).append(sample)  #?#grouped: {'group1': [[sample1], [sample2],...], 'group2': [[]], ... }    
    
    # Convert grouped dictionary to a list of groups
    groups = list(grouped.values())  
        
    # Shuffle the groups
    np.random.seed(42)
    np.random.shuffle(groups)  
    
    labels = [group[0][4] for group in groups]  #? get the label of the first sample in each group
    
    # Split the groups into train and test sets following the distribution in labels
    train_groups, test_groups = train_test_split(groups, test_size=test_size, stratify=labels, random_state=42)  #? train_test_split from sklearn, automatically splits the list

    # Flatten the list of groups back into arrays
    train_set = ([sample for group in train_groups for sample in group])
    test_set = ([sample for group in test_groups for sample in group])
    
    #convert to dataframes
    train_df = pd.DataFrame(train_set, columns=['dataset','subj_id', 'code', 'description_label', 'label', 'add'])
    test_df  = pd.DataFrame(test_set, columns=['dataset','subj_id', 'code', 'description_label', 'label', 'add'])
        
    #save annotation train file
    annotation_file = os.path.join('../Datasets/', 'annotations_train.pkl')
    with open(annotation_file, 'wb') as file:
        pickle.dump(train_df, file)
        
    #save annotation test file
    annotation_file = os.path.join('../Datasets/', 'annotations_test.pkl')
    with open(annotation_file, 'wb') as file:
        pickle.dump(test_df, file)
    
    return class_distribution, mean, std   
    
#!##
#!MAIN
#!##
if __name__ == '__main__':
    
    #!generate annotation files for each dataset, TEST and TRAIN
    class_distribution, mean, std = train_test_annotations(test_size=0.2) #20% test, 80% train
    
    #plot histogram class distribution
    class_distribution = class_distribution['Color']
    plt.bar(class_distribution.keys(), class_distribution.values(), color='skyblue', alpha=0.8)
    #plt.xlabel('Class')
    #plt.ylabel('Frequency')
    #plt.title('Distribution of Classes')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()
    
    #!check annotation files 
    df = pd.read_pickle('../Datasets/' + '/annotations_test.pkl') 
    #df.to_csv('annotations_train.csv', index=False)
    print(df)
    print(df.shape)
    print(df.columns)  
    
    
   
    
        
    
    
    
    
    
    
