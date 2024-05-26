class CalD3R_MenD3s_sample(object):
    def __init__(self, datasets_name, row, dataset_conf): 
        '''
        row: list(str)
            single row from pkl annotation file, representing single sample
        dataset: str
            dataset object .yaml
        '''
        #?PRIVATE attributes have initial underscore and need getters (and setters if requires) to be set
        self._index = str(row[0]) 
        self._datasets_name = datasets_name
        self._series = row[1]
        self.dataset_conf = dataset_conf
    
    #?PUBLIC SETTERS as properties 
    @property
    def uid(self):
        return int(self._index)
    
    @property
    def datasets_name(self):
        return self._datasets_name
    
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
        return str(self._series['label'])
    
    @property
    def add(self):
        return self._series['add'] 
    
    @property
    def gender(self):
        return str(self._series['add'][0]) 


#?- **Direct Attribute**: 
#   class Class(object):
#       def __init__(self, attribute): 
#           self.attribute = attribute
#?    - **Pros**: Simpler and more straightforward. Direct access to the value. 
#?   - **Cons**: Less flexible. If the computation or retrieval logic changes, you need to update all places where the attribute is set. 
#? 


#?- **Property**: 
#?#?When you use  @property, you can access the method as if it were an attribute, without needing to call it explicitly. 
#?This approach is useful for creating read-only attributes or for computing values dynamically when they are accessed. 
#   class Class(object):
#       def __init__(self, attribute): 
#           self.attribute = attribute
#
#       @property 
#       def attribute(self):
#           return int(self.attribute)
#
#?    - **Pros**: More flexible. Encapsulates the logic for retrieving the value. If the logic changes, you only need to update the property method. 
#?    - **Cons**: Slightly more complex. Each access involves a method call, which might have a negligible performance impact. 

