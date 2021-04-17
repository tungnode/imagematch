import nmslib
import numpy
import time
import numpy as np
import threading
import glob
import tensorflow as tf
from tensorflow.python.client import timeline
from scipy import spatial
import multiprocessing
import random
import sys
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
import json

def load_glob_worker(all_files,file_name_index,data_set,start,end,lock):
    # all_files = glob.glob('.\\vectors_test\\*.npz')
    n = 1#random.randint(1,10)
    # vector_data = []
    for index in range(start,end):
        data_set.append(np.loadtxt(all_files[index]))
        file_name_index[index] = all_files[index]
        # if n > 0:
        #     vector_data.append(np.loadtxt(all_files[file_name]))
        #     n -= 1
        # else:    
        #     lock.acquire()
        #     all_vectors_data_set.extend(vector_data)
        #     lock.release()
        #     n = random.randint(1,10)
        #     vector_data.clear()

def load_glob(all_vectors_data_set,file_index,start,end,lock):
    all_files = glob.glob('.\\vectors_test\\*.npz')
    vector_data = []
    for index in range(start,end):
        file_index[index] = all_files[index]
        all_vectors_data_set.append(np.loadtxt(all_files[index]))



if __name__ == '__main__':
    # load image vector files
    all_files = glob.glob('.\\vectors_test\\*.npz')
    total_files = len(all_files)
    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    data_set = manager.list()
    file_name_index = manager.dict()
    start_time = time.time()
    named_nearest_neighbors = []


    number_of_processes = 1
    processes_list = []
    item_per_process = int(total_files/number_of_processes)

    for p in range(number_of_processes):
        processes_list.append(multiprocessing.Process(target=load_glob_worker, args=(all_files,file_name_index,data_set,p*item_per_process,p*item_per_process+item_per_process,lock)))

    for p in range(number_of_processes):
        processes_list[p].start()

    for p in range(number_of_processes):
        processes_list[p].join()
    print("======================",len(data_set))
    # data_set = []
    # file_indexes = {}
    # load_glob(data_set,file_indexes,0,total_files,lock)

    # create a random matrix to index
    # data = numpy.random.randn(100000, 100).astype(numpy.float32)

    # initialize a new index, using a HNSW index on Cosine Similarity
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(data_set)
    index.createIndex({'post': 2}, print_progress=True)

    # query for the nearest neighbours of the first datapoint
    ids, distances = index.knnQuery(data_set, k=10)
    print("=====================")
    print(distances)
    print("=====================")
    print(ids)
    # for i, distance in enumerate(distances):
    #     if distance <= 0.02:
    #         file_path = file_name_index[i]
    #         named_nearest_neighbors.append({
    #                         'similarity': str(distance),
    #                         'master_pi': file_name_index[0],
    #                         'similar_pi': file_name_index[i]})
    
    print("Step.2 - Similarity score calculation - Finished ")
    # Writes the 'named_nearest_neighbors' to a json file
    with open('nearest_neighbors.json', 'w') as out:
        json.dump(list(named_nearest_neighbors), out)
    print("Step.3 - Data stored in 'nearest_neighbors.json' file ")
    print("--- Prosess completed in %.2f minutes ---------" %
        ((time.time() - start_time) / 60))

    # get all nearest neighbours for all the datapoint
    # using a pool of 4 threads to compute
    all_nearest_data = index.knnQueryBatch(data_set, k=5, num_threads=4)
    for master_index, master in enumerate(all_nearest_data):
        master_neighbors_index =master[0]
        master_neighbors_distances = master[1]
        for index_of_distance_to_neighbor, distance_to_neighbor in enumerate(master_neighbors_distances):
            if index_of_distance_to_neighbor != 0:
                if distance_to_neighbor <= 0.02:
                    master_file_path = file_name_index[master_index]
                    neighbor_file_path = file_name_index[master_neighbors_index[index_of_distance_to_neighbor]]
                    named_nearest_neighbors.append({
                                'similarity': str(distance_to_neighbor),
                                'master_pi': master_file_path,
                                'similar_pi': neighbor_file_path})    

    print("Step.2 - Similarity score calculation - Finished ")
    # Writes the 'named_nearest_neighbors' to a json file
    with open('nearest_neighbors.json', 'w') as out:
        json.dump(list(named_nearest_neighbors), out)
    print("Step.3 - Data stored in 'nearest_neighbors.json' file ")
    print("--- Prosess completed in %.2f minutes ---------" %
        ((time.time() - start_time) / 60))