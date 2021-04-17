import nmslib
import numpy
import time
import numpy as np
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
        data_set.append(np.loadtxt(all_files[index]))
        # file_name = os.path.basename(all_files[index]).split('.')[0]
        file_name_index[index] = all_files[index]
    #     if n > 0:
    #         vector_data.append(np.loadtxt(all_files[index]))
    #         n -= 1
    #     else:    
    #         lock.acquire()
    #         data_set.extend(vector_data)
    #         lock.release()
    #         n = random.randint(1,10)
    #         vector_data.clear()
    # if len(vector_data) != 0:
    #     lock.acquire()
    #     data_set = data_set + vector_data
    #     lock.release()

# def load_glob(all_vectors_data_set,file_index,start,end,lock):
#     all_files = glob.glob('.\\vectors_test_data\\*.npz')
#     vector_data = []
#     for index in range(start,end):
#         file_index[index] = all_files[index]
#         all_vectors_data_set.append(np.loadtxt(all_files[index]))

def is_legit_token(master_file_name,neighbor_file_name,map_of_token_with_owners):
    try:
        master_token = master_file_name.split("_")[0]
        neighbor_token = neighbor_file_name.split("_")[0]
        master_token_owners = map_of_token_with_owners.get(master_token)
        neighbor_token_owners = map_of_token_with_owners.get(neighbor_token)

        for master_address in master_token_owners:
            if ((master_address != '0x0000000000000000000000000000000000000000') and (master_address == neighbor_token_owners.get(master_address))):
                return True
    except Exception as error:
        print(error)
        return True    
    
    return False


def is_already_exist(master_file_name,neighbor_file_name,neighbor_master_set):
    new_neighbor_master_set = frozenset([master_file_name,neighbor_file_name])
    if new_neighbor_master_set not in neighbor_master_set:
        neighbor_master_set[new_neighbor_master_set] = new_neighbor_master_set
        return False
    else:
        return True

with open('..\\copyHunter\\scripts\\owners_data.json') as json_file:
    map_of_token_with_owners = json.load(json_file)
neighbor_master_map = {}
if __name__ == '__main__':
    # load image vector files
    all_files = glob.glob('.\\vectors\\*.npz')
    total_files = len(all_files)
    print(total_files)
    lock = multiprocessing.Lock()
    # multiprocessing.Array('i', 4)
    data_set = []# manager.list()
    file_name_index = {}#manager.dict()
    start_time = time.time()
    named_nearest_neighbors = []
    
 
    # number_of_processes = 5
    # processes_list = []
    # item_per_process = int(total_files/number_of_processes)

    # for p in range(number_of_processes):
    #     processes_list.append(multiprocessing.Process(target=load_glob_worker, args=(all_files,file_name_index,data_set,p*item_per_process,p*item_per_process+item_per_process,lock)))

    # for p1 in range(number_of_processes):
    #     processes_list[p1].start()

    # for p2 in range(number_of_processes):
    #     processes_list[p2].join()
    load_glob_worker(all_files,file_name_index,data_set,0,total_files,lock)
    print("======================",len(data_set))
    # data_set = []
    # file_indexes = {}
    # load_glob(data_set,file_indexes,0,total_files,lock)

    # create a random matrix to index
    # data = numpy.random.randn(100000, 100).astype(numpy.float32)

    # initialize a new index, using a HNSW index on Cosine Similarity
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(data_set)
    index.createIndex({'post': 2,'efConstruction':1000,'M':50,}, print_progress=True)
    # index.createIndex({'numPivot':5000,'numPivotIndex':5000})
    print(index)
    # query for the nearest neighbours of the first datapoint
    # ids, distances = index.knnQuery(data_set, k=10)
    
   
    # get all nearest neighbours for all the datapoint
    # using a pool of 4 threads to compute
    all_nearest_data = index.knnQueryBatch(data_set, k=10, num_threads=4)
    for master_index, master in enumerate(all_nearest_data):
        master_neighbors_index =master[0]
        master_neighbors_distances = master[1]
        for index_of_distance_to_neighbor, distance_to_neighbor in enumerate(master_neighbors_distances):
            # if index_of_distance_to_neighbor != 0:
                if distance_to_neighbor <= 0.02:
                    master_file_path = file_name_index[master_index]
                    neighbor_index = master_neighbors_index[index_of_distance_to_neighbor]
                    neighbor_file_path = file_name_index[neighbor_index]
                    if( master_file_path != neighbor_file_path):
                    # similarity = 1 - spatial.distance.cosine(data_set[master_index], data_set[neighbor_index])
                    # rounded_similarity = int((similarity * 10000)) / 10000.0
                    # if((rounded_similarity >= 0.99) 
                    #     and (is_legit_token(master_file_path,neighbor_file_path,map_of_token_with_owners) == False)
                    #     and (is_already_exist(master_file_path,neighbor_file_path,neighbor_master_map) == False)):
                        named_nearest_neighbors.append({
                                    # 'spatial_similarity': str(rounded_similarity),
                                    'nmslib_similarity': str(distance_to_neighbor),
                                    'master_pi': master_file_path,
                                    'similar_pi': neighbor_file_path})    

    print("Step.2 - Similarity score calculation - Finished ")
    # Writes the 'named_nearest_neighbors' to a json file
    with open('nearest_neighbors.json', 'w') as out:
        json.dump(list(named_nearest_neighbors), out)
    print("Step.3 - Data stored in 'nearest_neighbors.json' file ")
    print("--- Prosess completed in %.2f minutes ---------" %
        ((time.time() - start_time) / 60))