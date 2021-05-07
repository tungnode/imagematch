import datetime
import time
import logging
import requests
import json
import numpy as np
import os.path
import multiprocessing
from typing import Tuple, Optional, Callable, List, Iterable
from queue import Queue
import hnswlib
from json import JSONEncoder
from constants import resources_folder
import time
import gc
from multiprocessing.managers import BaseManager

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
if __name__ == "__main__":
    
        BaseManager.register('get_features_queue')
        BaseManager.register('get_indexed_files_dict')
        m = BaseManager(address=('127.0.0.1', 50000), authkey=b'abracadabra')
        m.connect()
        vector_feature_queue = m.get_features_queue()
        indexed_files = m.get_indexed_files_dict()
        data_set = []
        index_to_file_name = {}
        start_time = time.time()
        dims = 1792
        
        vector_features_file_path = resources_folder+"vectors_features.json"
        vector_index_file_names_path = resources_folder+"index_to_file_name.json"
        index_file_path = resources_folder+'hnswlib.bin'
        number_100k = 10000

        read_file =  open(vector_index_file_names_path, "r")
        index_to_file_name = json.load(read_file)
        # files_in_index = {}
        # read_file = open(resources_folder+"index_to_file_name.json", "r")
        # index_to_file_name = json.load(read_file)
        # for name in index_to_file_name.values():
        #     name = name[name.rfind('\\'):]
        #     name = name[:name.find('.')]
        #     name = name.lower()
        #     files_in_index[name] = name
        
        indexed_number = len(index_to_file_name)
        vector_features_batch_number = len(next(os.walk(resources_folder+"vector_features_batches"))[2])
        number_features_in_batch = number_100k
        counter = 0    
        while True:
            try:
                address_token_img_type,features_set = vector_feature_queue.get()
                vector_features_file_path = resources_folder+"vector_features_batches\\"+str(vector_features_batch_number)
                if counter == 100 or address_token_img_type is None:
                    #     with open(vector_index_file_names_path, "r") as read_file:
                    #         from_file_index_to_file_name = json.load(read_file)
                    # else:
                    #     print("Vectors features not found. Exit")
                    #     exit()
                    index = hnswlib.Index(space='cosine', dim=dims) # possible options are l2, cosine or ip
                    if os.path.exists(index_file_path) == False:
                        print("index file not exist, exit")
                        exit()
                    index.load_index(index_file_path,max_elements = indexed_number)
                    index.add_items(data_set)
                    
                    print("--------------- Saving index to files")
                    index.save_index(index_file_path)
                    if number_features_in_batch <=0:
                        vector_features_batch_number += 1
                        vector_features_file_path = resources_folder+"vector_features_batches\\"+str(vector_features_batch_number)
                        number_features_in_batch = number_100k
                    else:
                        if os.path.exists(vector_features_file_path):
                            with open(vector_features_file_path, "r") as read_file:
                                in_file_data_set = list(np.asarray(json.load(read_file)['data']))
                                data_set = in_file_data_set + data_set
                    with open(vector_features_file_path, 'w') as out:
                        json.dump({'data':data_set},  out,cls=NumpyArrayEncoder,)
                    with open(vector_index_file_names_path, 'w') as out:
                        json.dump(index_to_file_name,  out)
                    counter = 0
                    del index
                    data_set.clear() 
                    gc.collect()
              

                    
                if address_token_img_type is not None and features_set is not None:
                    
                    file_path = resources_folder + "vectors\\"+address_token_img_type+".npz"
                    key_of_files_in_index = "\\"+address_token_img_type.split(".")[0]
                    key_of_files_in_index = key_of_files_in_index.lower()
                    if indexed_files.get(key_of_files_in_index) != None:
                        print("file was indexed",address_token_img_type)
                        continue
                    print("--------------- indexing",address_token_img_type)
                    data_set.append(features_set)                
                    index_to_file_name[str(indexed_number)] = file_path
                    indexed_files.update({key_of_files_in_index:file_path})
                    indexed_number += 1
                    counter += 1
                    number_features_in_batch -= 1
                else:
                    break
                
                gc.collect()
            except Exception as e:
                print("Exception while indexing image",e)    
                continue
        print("Finished indexing images")