class CalD3R_MenD3s_sample(object):
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
        return str(self._series['dataset'])
    
    @property
    def subj_id(self):
        return int(self._series['subj_id'])
    
    @property
    def code(self):
        return str(self._series['code'])

    @property
    def label(self):
        if 'label' not in self._series.keys().tolist():
            raise NotImplementedError
        return int(self._series['label'])
    
    @property
    def description_label(self):
        return str(self._series['description_label'])
    
    @property
    def add(self):
        return self._series['add'] 
    
    @property
    def gender(self):
        return str(self._series['add'][0]) 


