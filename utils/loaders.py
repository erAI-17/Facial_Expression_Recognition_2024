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
    def __init__(self, modalities, mode, dataset,
                 transform=None, load_feat=False, additional_info=False, **kwargs):
        """
        modalities: list(str, str, ...) #? ["RGB"] OR ["D"] OR ["RGB", "D"] OR ["MESH"]
        mode: str #? train OR test
        
        dataset:  
            - annotations_path: str #? general annotation for multi modal data
            - dataset[modality]: #?for each modality
                - data_path: str 
                - tmpl: str #?template of single data name (for example, for an image: "img_{:010d}.jpg")
        
        transform: bool #? image normalization, online augmentation (crop, ...)
        load_feat: bool #? if mode="save" then load_feat="False". If mode="train" OR mode="test" then load_feat="True"
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = modalities
        self.mode = mode 
        self.dataset = dataset
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat
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
        
        #*Load Features for each modality (if mode="train" OR mode="test" )
        if self.load_feat:
            self.mod_features = None
            for m in self.modalities:
                if m == 'RGB':
                    mod_features = pd.DataFrame(pd.read_pickle(os.path.join(self.dataset[m].features_name + "_" +
                                                                          pickle_name))['features'])[["uid", "features_" + m]] #pickle_name
                elif m=='D':    
                    mod_features = pd.DataFrame(pd.read_pickle(os.path.join(self.dataset[m].features_name + 
                                                                            pickle_name)))[["uid", "features_" + m]]
                elif m=='MESH':    
                    mod_features = pd.DataFrame(pd.read_pickle(os.path.join(self.dataset[m].features_name + 
                                                                            pickle_name)))[["uid", "features_" + m]]    
                
                if self.mod_features is None:
                    self.mod_features = mod_features
                else:
                    self.mod_features = pd.merge(self.mod_features, mod_features, how="inner", on="uid")
            
            
    def __getitem__(self, index):
        '''
        if training OR testing (self.mode= "train" AND load.feat="True"), it loads the previously extracted features FOR each modality, and the class
        else if saving features (self.mode= "save" AND load.feat="False"), it loads the image FOR each modality, and the class
        '''
        ann_sample = self.ann_sample_list[index] #annotation sample
      
        if self.load_feat: #*training or testing : load the features for each modality
            sample = {}
            sample_mod_features = self.mod_features[self.mod_features["uid"] == int(ann_sample.uid)]
            assert len(sample_mod_features) == 1
            for m in self.modalities:
                sample[m] = sample_mod_features["features_" + m].values[0]
            if self.additional_info:
                return sample, ann_sample.label, ann_sample.uid
            else:
                return sample, ann_sample.label

        else: #*saving: load the images for each modality
            sample = {}
            for m in self.modalities:
                img, label = self.get(m, ann_sample)
                sample[m] = {img}
            if self.additional_info:
                return sample, ann_sample.label, ann_sample.uid
            else:
                return sample, ann_sample.label

    def get(self, modality, sample):
        '''
        Loads single image, applies transformations if required (online augmentation, normalization,...)
        '''    
        img = self._load_data(modality, sample)
            
        if self.transform is not None: #*ONLINE AUGMENTATION
            transformed_img = self.transform[modality](img)
        else: 
            transformed_img = img
            
        return transformed_img, sample.label

    def _load_data(self, modality, sample, idx):
        '''
        Loads single image
        '''
        data_path = self.dataset[modality].data_path
        tmpl = self.dataset[modality].tmpl

        if modality == 'RGB' or modality=="D":
            try:
                img = Image.open(os.path.join(data_path, tmpl.format())).convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                raise FileNotFoundError
            
            return [img]
        
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
            return len(self.sample_list)
