class BU3DFE_sample(object):
    def __init__(self, row, dataset_conf): 
        '''
        row: list(str)
            single row from pkl annotation file, representing single sample
        dataset: str
            dataset object .yaml
        '''
        self._index = str(row[0]) 
        self._series = row[1]
        self.dataset_conf = dataset_conf
    
    @property
    def uid(self):
        return int(self._index)
    
    @property
    def datasets_name(self):
        return str(self.dataset_conf.name)
    
    @property
    def subj_id(self):
        return str(self._series['subj_id'])
    
    @property
    def intensity(self):
        return str(self._series['intensity'])
    
    @property
    def race(self):
        return str(self._series['race'])
    
    @property
    def label(self):
        if 'label' not in self._series.keys().tolist():
            raise NotImplementedError
        return int(self._series['label'])
    
    @property
    def description_label(self):
        return str(self._series['description_label'])
    