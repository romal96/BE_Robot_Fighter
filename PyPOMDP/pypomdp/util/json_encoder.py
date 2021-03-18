import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class toJSON():
    def __init__(self, filename, obj):
        self._file = filename
        self._data = {'alphavec':[]}
        self._o = obj

    # function to add to JSON 
    def write_json(self):
        with open(self._file,'w+') as f: 
            for av in self._o :              
                self._data['alphavec'].append(av.__dict__)
            json.dump(self._data, f, indent=4,cls=NumpyEncoder)
      
        f.close()
        
    def saving_belief_points(self, beliefs):
        self._data['beliefs'] = beliefs
        #for b in beliefs :
        #    self._data['beliefs'].append(b.__dict__)


