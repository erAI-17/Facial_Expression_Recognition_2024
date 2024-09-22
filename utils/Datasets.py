from abc import ABC
import pandas as pd
import torch.utils.data as data
from PIL import Image
import platform
import random
import os
import os.path
from utils.logger import logger
from utils.CalD3R_MenD3s_sample import CalD3R_MenD3s_sample
from utils.BU3DFE_sample import BU3DFE_sample
from utils.Global_sample import Global_sample
from utils.args import args
import matplotlib.pyplot as plt
import utils.utils

            
class CalD3RMenD3s_Dataset(data.Dataset, ABC):
    def __init__(self, 
                 name,
                 modalities, 
                 dataset_conf,
                 transform = None):    
 
        self.name = name
        self.modalities = modalities
        self.dataset_conf = dataset_conf
        self.transform = transform
        
        self.num_classes = utils.utils.get_domains_and_labels(args)  
        
        #create global dataset object
        pickle_name = 'annotations_complete.pkl'
        self.ann_list = []
        self.ann_list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, self.name, pickle_name))
        
        if self.num_classes == 6: #delete all surprise samples
            self.ann_list_file = self.ann_list_file[self.ann_list_file['description_label'] != 'surprise']
            #! not needed (surprise was last one)
            #original_order = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'neutral':4, 'sadness':5, 'surprise':6}
            #new_order = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'neutral':4, 'sadness':5}
            #self.ann_list_file['label'] = self.ann_list_file['description_label'].map(new_order)
            
        self.ann_list.extend([CalD3R_MenD3s_sample(row, self.dataset_conf) for row in self.ann_list_file.iterrows()])
        
        ##! if local run, reduce the validation set for faster debug 
        if platform.node() == 'MSI':
            reduced_size = int(len(self.ann_list) * 0.2)  # Calculate 20% of the current list size
            self.ann_list = random.sample(self.ann_list, reduced_size)  # Randomly select 20% of the items

    def __len__(self):
            return len(self.ann_list)
        
    def __getitem__(self, index):
        ann_sample = self.ann_list[index] #annotation sample
      
        #*load the sample's images for each modality
        sample = {}
        for m in self.modalities:
            img, label = self.get(m, ann_sample)

            if img is None:  #! If any modality image is None because of corrupted or missing file, take the sample at next index instead
                return self.__getitem__((index + 1) % len(self.ann_list))
            else:
                sample[m] = img
        
        #*apply transformations (convert to tensor, normalize, augment)!
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample, label


    def get(self, modality, ann_sample):
        '''
        Loads single image, applies transformations if required (online augmentation, normalization,...)
        '''    
        img = self._load_data(modality, ann_sample)
        
        if img is None: #!file corrupted or missing (handled in __get_item__)
            return None, None

        return img, ann_sample.label


    def _load_data(self, modality, ann_sample):
        '''
        Loads single image
        '''
        data_path = os.path.join(self.dataset_conf[modality].data_path, self.name, ann_sample.datasets_name, ann_sample.description_label.capitalize(), modality)
        
        #!CalD3r and MenD3s have different image templates 
        tmpl = "{}_{:03d}_{}_{}_{}.png" if ann_sample.datasets_name == 'CalD3r' else "{}_{:02d}_{}_{}_{}.png" if ann_sample.datasets_name == 'MenD3s' else None
        conv = {'RGB': 'Color', 'DEPTH': 'Depth'}
        
        try:
            if args.align_face:
                img_path = os.path.join(data_path, 'aligned_' + tmpl.format(ann_sample.gender, ann_sample.subj_id, ann_sample.code, ann_sample.description_label, conv[modality]))
            else:
                img_path = os.path.join(data_path, tmpl.format(ann_sample.gender, ann_sample.subj_id, ann_sample.code, ann_sample.description_label, conv[modality]))
                
            img = Image.open(img_path)
            
            if img is None: #!image not found or corrupt, cv2 returns None
                print("Img not found at path:", os.path.join(data_path, tmpl.format(ann_sample.gender, ann_sample.subj_id, ann_sample.code, ann_sample.description_label, conv[modality])))
                raise FileNotFoundError 
                    
            return img
        
        except Exception as e: 
            logger.error(f"Error loading {modality} image: {e}")
            
            return None
        
    
class BU3DFE_Dataset(data.Dataset, ABC):
    def __init__(self, 
                 name,
                 modalities, 
                 dataset_conf,
                 transform = None):

        self.name = name
        self.modalities = modalities
        self.dataset_conf = dataset_conf
        self.transform = transform
        
        self.num_classes = utils.utils.get_domains_and_labels(args)  
             
        #create global dataset object
        pickle_name = 'annotations_complete.pkl'
        self.ann_list = []
        self.ann_list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, self.name, pickle_name))
        
        if self.num_classes == 6: #delete all Neutral samples and reorder the labels
            self.ann_list_file = self.ann_list_file[self.ann_list_file['description_label'] != 'NE']
            #original_order = {'AN':0, 'DI':1, 'FE':2, 'HA':3, 'NE':4, 'SA':5, 'SU':6}  
            new_order = {'AN':0, 'DI':1, 'FE':2, 'HA':3, 'SA':4, 'SU':5}
            self.ann_list_file['label'] = self.ann_list_file['description_label'].map(new_order)
            
        self.ann_list.extend([BU3DFE_sample(row, self.dataset_conf) for row in self.ann_list_file.iterrows()])
        
        ##! if only highest intensity level is used, remove the rest (2400 -> 1200 samples)
        if args.high_intensity == True:
            self.ann_list = [sample for sample in self.ann_list if sample.intensity == '03' or sample.intensity == '04']
        
        # ##! if local run, reduce the validation set for faster debug 
        # if platform.node() == 'MSI':
        #     reduced_size = int(len(self.ann_list) * 0.2)  # Calculate 20% of the current list size
        #     self.ann_list = random.sample(self.ann_list, reduced_size)  # Randomly select 20% of the items

    def __len__(self):
            return len(self.ann_list)
        
    def __getitem__(self, index):
        ann_sample = self.ann_list[index] #annotation sample
      
        #*load the sample's images for each modality
        sample = {}
        for m in self.modalities:
            img, label = self.get(m, ann_sample)

            if img is None:  #! If any modality image is None because of corrupted or missing file, take the sample at next index instead
                return self.__getitem__((index + 1) % len(self.ann_list))
            else:
                sample[m] = img
        
        #*apply transformations (convert to tensor, normalize, augment)!
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def get(self, modality, ann_sample):
        '''
        Loads single image, applies transformations if required (online augmentation, normalization,...)
        '''    
        img = self._load_data(modality, ann_sample)
        
        if img is None: #!file corrupted or missing (handled in __get_item__)
            return None, None
 
        return img, ann_sample.label


    def _load_data(self, modality, ann_sample):
        '''
        Loads single image
        '''
        data_path = os.path.join(self.dataset_conf[modality].data_path, self.name, 'Subjects', ann_sample.subj_id)
        
        tmpl = "{}_{}_{}"
        conv = {'RGB': 'F2D.bmp', 'DEPTH': 'F3D_depth.png'}
        try:                
            img = Image.open(os.path.join(data_path, tmpl.format(ann_sample.subj_id, str(ann_sample.description_label + ann_sample.intensity + ann_sample.race), conv[modality])))
            
            if img is None: #!image not found or corrupt
                print("Img not found at path:", os.path.join(data_path, tmpl.format(ann_sample.subj_id, str(ann_sample.description_label + ann_sample.intensity + ann_sample.race), conv[modality])))
                raise FileNotFoundError 
                    
            return img
        
        except Exception as e: 
            logger.error(f"Error loading RGB image: {e}")
            return None
        
        
        
class Global_Dataset(data.Dataset, ABC):
    def __init__(self, 
                name,
                modalities, 
                dataset_conf,
                transform = None):

        self.name = name
        self.modalities = modalities
        self.dataset_conf = dataset_conf
        self.transform = transform
        
        self.num_classes = utils.utils.get_domains_and_labels(args)  
                
        #create global dataset object
        pickle_name = 'annotations_complete.pkl'
        self.ann_list = []
        self.ann_list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, self.name, pickle_name))
                
        self.ann_list.extend([Global_sample(row, self.dataset_conf) for row in self.ann_list_file.iterrows()])
    
        ##! if local run, reduce the validation set for faster debug 
        if platform.node() == 'MSI':
            reduced_size = int(len(self.ann_list) * 0.1)  # Calculate 10% of the current list size
            self.ann_list = random.sample(self.ann_list, reduced_size)  # Randomly select 20% of the items

    def __len__(self):
            return len(self.ann_list)
        
    def __getitem__(self, index):
        ann_sample = self.ann_list[index] #annotation sample
        
        #*load the sample's images for each modality
        sample = {}
        for m in self.modalities:
            img, label = self.get(m, ann_sample)

            if img is None:  #! If any modality image is None because of corrupted or missing file, take the sample at next index instead
                return self.__getitem__((index + 1) % len(self.ann_list))
            else:
                sample[m] = img
        
        #*apply transformations (convert to tensor, normalize, augment)!
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def get(self, modality, ann_sample):
        '''
        Loads single image, applies transformations if required (online augmentation, normalization,...)
        '''    
        img = self._load_data(modality, ann_sample)
        
        if img is None: #!file corrupted or missing (handled in __get_item__)
            return None, None

        return img, ann_sample.label


    def _load_data(self, modality, ann_sample):
        '''
        Loads single image
        '''
        if ann_sample.datasets_name == 'BU3DFE':
            data_path = os.path.join(self.dataset_conf[modality].data_path, ann_sample.datasets_name, 'Subjects', ann_sample.subj_id)
            
            tmpl = "{}_{}_{}"
            conv = {'RGB': 'F2D.bmp', 'DEPTH': 'F3D_depth.png'}
            emot_conv = {'anger': 'AN', 'disgust': 'DI', 'fear': 'FE', 'happiness': 'HA', 'neutral': 'NE', 'sadness': 'SA', 'surprise': 'SU'}
            try:                
                img = Image.open(os.path.join(data_path, tmpl.format(ann_sample.subj_id, str(emot_conv[ann_sample.description_label] + ann_sample.intensity + ann_sample.race), conv[modality])))
                
                if img is None: #!image not found or corrupt
                    print("Img not found at path:", os.path.join(data_path, tmpl.format(ann_sample.subj_id, str(ann_sample.description_label + ann_sample.intensity + ann_sample.race), conv[modality])))
                    raise FileNotFoundError 
                        
                return img
            
            except Exception as e: 
                logger.error(f"Error loading RGB image: {e}")
                return None
        
        
        elif ann_sample.datasets_name == 'CalD3r' or ann_sample.datasets_name == 'MenD3s':
            data_path = os.path.join(self.dataset_conf[modality].data_path, 'CalD3rMenD3s', ann_sample.datasets_name, ann_sample.description_label.capitalize(), modality)
            
            #!CalD3r and MenD3s have different image templates 
            tmpl = "{}_{:03d}_{}_{}_{}.png" if ann_sample.datasets_name == 'CalD3r' else "{}_{:02d}_{}_{}_{}.png" if ann_sample.datasets_name == 'MenD3s' else None
            conv = {'RGB': 'Color', 'DEPTH': 'Depth'}
            
            try:
                if args.align_face:
                    img_path = os.path.join(data_path, 'aligned_' + tmpl.format(ann_sample.gender, int(ann_sample.subj_id), ann_sample.code, ann_sample.description_label, conv[modality]))
                else:
                    img_path = os.path.join(data_path, tmpl.format(ann_sample.gender, ann_sample.subj_id, ann_sample.code, ann_sample.description_label, conv[modality]))
                    
                img = Image.open(img_path)
                
                if img is None: #!image not found or corrupt, cv2 returns None
                    print("Img not found at path:", os.path.join(data_path, tmpl.format(ann_sample.gender, int(ann_sample.subj_id), ann_sample.code, ann_sample.description_label, conv[modality])))
                    raise FileNotFoundError 
                        
                return img
            
            except Exception as e: 
                logger.error(f"Error loading {modality} image: {e}")
                return None
            
            
            
        elif ann_sample.datasets_name == 'Bosphorus':
            data_path = os.path.join(self.dataset_conf[modality].data_path, ann_sample.datasets_name, 'Subjects', ann_sample.subj_id.split('_')[0])
            
            tmpl = "{}_{}_{}_{}.png"
            conv = {'RGB': 'rgb', 'DEPTH': 'depthmap'}
            emot_conv = {'anger': 'ANGER', 'disgust': 'DISGUST', 'fear': 'FEAR', 'happiness': 'HAPPY', 'neutral': 'NEUTRAL', 'sadness': 'SADNESS', 'surprise': 'SURPRISE'}
            try:   
                path = os.path.join(data_path, tmpl.format(ann_sample.subj_id.split('_')[0], emot_conv[ann_sample.description_label], ann_sample.subj_id.split('_')[1], conv[modality]))       
                img = Image.open(os.path.join(data_path, tmpl.format(ann_sample.subj_id.split('_')[0], emot_conv[ann_sample.description_label], ann_sample.subj_id.split('_')[1], conv[modality])))
                
                if img is None: #!image not found or corrupt
                    print("Img not found at path:", os.path.join(data_path, tmpl.format(ann_sample.subj_id.split('_')[0], emot_conv[ann_sample.description_label], ann_sample.subj_id.split('_')[1], conv[modality])))
                    raise FileNotFoundError 
                        
                return img
            
            except Exception as e: 
                logger.error(f"Error loading RGB image: {e}")
                return None