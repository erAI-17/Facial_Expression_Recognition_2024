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
        return int(self._series['uid'])
    
    @property
    def gender(self):
        return self._series['gender'] 
    
    @property
    def subj_id(self):
        return self._series['subj_id']
    
    @property
    def code(self):
        return self._series['code']

    @property
    def label(self):
        if 'class' not in self._series.keys().tolist():
            raise NotImplementedError
        return self._series['label']

    

