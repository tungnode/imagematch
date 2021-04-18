import hnswlib
import numpy as np
import pickle
import time
import threading
import glob
# import tensorflow as tf
# from tensorflow.python.client import timeline
from scipy import spatial
import multiprocessing
import random
import sys
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
import json
import os.path
import logging
logging.basicConfig(level=logging.DEBUG)

def load_glob_worker(all_files,file_name_index,data_set,start,end,lock):
    # all_files = glob.glob('.\\vectors_test\\*.npz')
    n = random.randint(3,10)
    vector_data = []
    for index in range(start,end):
        # data_set.append(np.loadtxt(all_files[index]))
        file_name = os.path.basename(all_files[index]).split('.')[0]
        file_name_index[index] = file_name
        if n > 0:
            vector_data.append(np.loadtxt(all_files[index]))
            n -= 1
        else:    
            lock.acquire()
            data_set.extend(vector_data)
            lock.release()
            n = random.randint(1,10)
            vector_data.clear()
    if len(vector_data) != 0:
        lock.acquire()
        data_set.extend(vector_data)
        lock.release()
if __name__ == '__main__':
    # load image vector files
    all_files = glob.glob('.\\vectors_test_data\\*.npz')
    num_elements = len(all_files)
    dim = 1792
    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    data_set = []
    file_name_index = manager.dict()
    start_time = time.time()
    named_nearest_neighbors = []
    
 
    # number_of_processes = 6
    # processes_list = []
    # item_per_process = int(num_elements/number_of_processes)

    # for p in range(number_of_processes):
    #     processes_list.append(multiprocessing.Process(target=load_glob_worker, args=(all_files,file_name_index,data_set,p*item_per_process,p*item_per_process+item_per_process,lock)))

    # for p in range(number_of_processes):
    #     processes_list[p].start()

    # for p in range(number_of_processes):
    #     processes_list[p].join()
    load_glob_worker(all_files,file_name_index,data_set,0,num_elements,lock)
    # Generating sample data
    data = np.float32(np.random.random((num_elements, dim)))
    data_labels = np.arange(num_elements)

    # Declaring index
    p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip

    # Initializing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements = num_elements, ef_construction = 2000, M = 16)

    # Element insertion (can be called several times):
    p.add_items(data_set)

    # Controlling the recall by setting ef:
    p.set_ef(50) # ef should always be > k

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(data_set, k = 5)
    print(labels)
    print(distances)
    for master_index,master_distance in enumerate(distances):
        for neighbor_index,neighbor_distance in enumerate(master_distance):
            if neighbor_distance < 0.01:
                print(file_name_index[master_index])
                print(file_name_index[master_index])
    # Index objects support pickling
    # WARNING: serialization via pickle.dumps(p) or p.__getstate__() is NOT thread-safe with p.add_items method!
    # Note: ef parameter is included in serialization; random number generator is initialized with random_seed on Index load
    p_copy = pickle.loads(pickle.dumps(p)) # creates a copy of index p using pickle round-trip

    ### Index parameters are exposed as class properties:
    print(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}") 
    print(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
    print(f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}")
    print(f"Search speed/quality trade-off parameter: ef={p_copy.ef}")