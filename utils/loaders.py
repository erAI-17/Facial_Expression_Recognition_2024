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
                 name,
                 modalities, 
                 mode, 
                 dataset_conf,
                 transform=None, 
                 additional_info=False, 
                 **kwargs):    
        """
        Parameters
        ----------
        name: (str)
            datasets names to be merged (underscore separated). For example: 'CalD3r_MenD3s'
        modalities: list(str)
            ["RGB"] OR ["DEPTH"] OR ["RGB", "DEPTH"] OR ["MESH"]
        mode: str
            train OR test
            
        dataset_conf:  
            - annotations_path: str 
                general annotation for multi modal data
            - dataset_conf[modality]: 
                - data_path: str 
                - tmpl: str 
                    template of single data name (for example, for an image: "img_{:010d}.jpg")
        
        transform: bool 
            image normalization, online augmentation (crop, ...)
        additional_info: bool
            set to True if you want to receive also the uid and the video name from the get furthre notice
        """
        self.modalities = modalities
        self.mode = mode 
        self.dataset_conf = dataset_conf
        self.transform = transform
        self.additional_info = additional_info
        

        if self.mode == "train":
            pickle_name = 'annotation' + '_train.pkl'
        else:
            pickle_name = 'annotation' + '_test.pkl'

        #!Read annotations for each dataset selected in args.name,  and create unique ann_list
        datasets_names = name.split('_')
        self.ann_list = []
        for dataset_name in datasets_names:
            self.ann_list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, dataset_name, pickle_name))
            logger.info(f"Dataloader for {self.mode} with {len(self.list_file)} samples generated")
        
            self.ann_list.extend([CalD3R_MenD3s_sample(dataset_name, row, self.dataset_conf) for row in self.ann_list_file.iterrows()])
            
            
    def __getitem__(self, index):
        ann_sample = self.ann_list[index] #annotation sample
      
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
        data_path = os.path.join(self.dataset_conf[modality].data_path, ann_sample.dataset_name, ann_sample.label.capitalize(), modality)
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality=="DEPTH" or modality=='MESH': #? if modality=='MESH', load anyway the RGB and depth_map and create the mesh later before forward pass
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
