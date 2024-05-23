import glob
from abc import ABC
import pandas as pd
import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import os.path
from utils.logger import logger
from utils.CalD3R_MenD3s_sample import CalD3R_MenD3s_sample

class CalD3R_MenD3s_Dataset(data.Dataset, ABC):
    def __init__(self, 
                 modalities, 
                 mode, 
                 dataset,
                 transform=None, 
                 additional_info=False, 
                 **kwargs):
        """
        modalities: list(str, str, ...) #? ["RGB"] OR ["D"] OR ["RGB", "D"] OR ["MESH"]
        mode: str #? train OR test
        
        dataset:  
            - annotations_path: str #? general annotation for multi modal data
            - dataset[modality]: #?for each modality
                - data_path: str 
                - tmpl: str #?template of single data name (for example, for an image: "img_{:010d}.jpg")
        
        transform: bool #? image normalization, online augmentation (crop, ...)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = modalities
        self.mode = mode 
        self.dataset = dataset
        self.transform = transform
        self.additional_info = additional_info
        

        if self.mode == "train":
            pickle_name = "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = "_test.pkl"

        #*Read annotations
        self.ann_list_file = pd.read_pickle(os.path.join(self.dataset.annotations_path, pickle_name))
        logger.info(f"Dataloader for {self.mode} with {len(self.list_file)} samples generated")
        
        self.ann_sample_list = [CalD3R_MenD3s_sample(row_name, self.dataset) for row_name in self.ann_list_file.iterrows()]
            
            
    def __getitem__(self, index):
        ann_sample = self.ann_sample_list[index] #annotation sample
      
        #*load the sample's images for each modality
        sample = {}
        for m in self.modalities:
            img, label = self.get(m, ann_sample)
            sample[m] = {img}
        if self.additional_info:
            return sample, ann_sample.label, ann_sample.uid
        else:
            return sample, ann_sample.label


    def get(self, modality, ann_sample):
        '''
        Loads single image, applies transformations if required (online augmentation, normalization,...)
        '''    
        img = self._load_data(modality, ann_sample)
            
        if self.transform is not None: #*ONLINE AUGMENTATION, NORMALIZATION
            transformed_img = self.transform[modality](img)
        else: 
            transformed_img = img
            
        return transformed_img, ann_sample.label


    def _load_data(self, modality, ann_sample):
        '''
        Loads single image
        '''
        data_path = self.dataset[modality].data_path
        tmpl = self.dataset[modality].tmpl

        if modality == 'RGB' or modality=="D":
            try:
                img = Image.open(os.path.join(data_path, tmpl.format(ann_sample.gender, ann_sample.subj_id, ann_sample.code, ann_sample.label, modality))).convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                raise FileNotFoundError
            
            return img

        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
            return len(self.sample_list)
