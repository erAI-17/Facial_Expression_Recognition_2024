from .video_record import VideoRecord
class ActionSenseRecord(VideoRecord):
    '''
    tup: a line from one of pkl files in annotations (train_val/...)
    dataset_conf: whole JSON "dataset" from .yaml
    '''
    def __init__(self, tup, dataset_conf):
        self._index = str(tup[0])
        self._series = tup[1]
        self.dataset_conf = dataset_conf
    
    @property
    def uid(self):
        return int(self._series['uid'])
    
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
        return self._series['class']
    
