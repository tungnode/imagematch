import nmslib
import numpy
import time
import numpy as np
import threading
import glob
from scipy import spatial
import multiprocessing
import random
import sys
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
import json
import os.path
import logging
from annoy import AnnoyIndex
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

def is_legit_token(master_file_name,neighbor_file_name):
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


def is_already_exist(master_file_name,neighbor_file_name):
    new_neighbor_master_set = frozenset([master_file_name,neighbor_file_name])
    if new_neighbor_master_set not in neighbor_master_set:
        neighbor_master_set[new_neighbor_master_set] = new_neighbor_master_set
        return False
    else:
        return True
def annoy_similarity(file_index_to_file_name,file_index_to_file_vector):
    dims = 1792
    n_nearest_neighbors = 20
    trees = 10000
    t = AnnoyIndex(dims, metric='angular')
    for k in file_index_to_file_vector:
        t.add_item(k, file_index_to_file_vector[k])
    t.build(trees)

    
    all_nearest_neighboor = []
    # Loops through all indexed items
    for i in file_index_to_file_name.keys():
        # Assigns master file_name, image feature vectors
        # and product id values
        master_file_name =  os.path.basename(file_index_to_file_name[i]).split('.')[0]
        master_vector = file_index_to_file_vector[i]
        # Calculates the nearest neighbors of the master item
        nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)
        # Loops through the nearest neighbors of the master item
        for j in nearest_neighbors:
            print(j)
            # Assigns file_name, image feature vectors and
            # product id values of the similar item
            neighbor_file_name = os.path.basename(file_index_to_file_name[j]).split('.')[0]
            neighbor_file_vector = file_index_to_file_vector[j]

            # Calculates the similarity score of the similar item
            similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
            rounded_similarity = int((similarity * 10000)) / 10000.0

            # Appends master product id with the similarity score
            # and the product id of the similar items
            if ((rounded_similarity > 0.98) and (master_file_name != neighbor_file_name) 
                and (is_legit_token(master_file_name,neighbor_file_name) == False)
                and (is_already_exist(master_file_name,neighbor_file_name) == False)):
                    all_nearest_neighboor.append({
                            'similarity': rounded_similarity,
                            'master_pi': master_file_name,
                            'similar_pi': neighbor_file_name})

        print("---------------------------------")
        print("Similarity index       : %s" % i)
        print("Master Image file name : %s" % file_index_to_file_name[i])
        print("Nearest Neighbors.     : %s" % nearest_neighbors)
        print("--- %.2f minutes passed ---------" % ((time.time() - start_time) / 60))
    return all_nearest_neighboor



with open('..\\copyHunter\\scripts\\owners_data.json') as json_file:
    map_of_token_with_owners = json.load(json_file)
neighbor_master_set = {}
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
    # Defining data structures as empty dict
    annoy_file_index_to_file_name = {}
    annoy_file_index_to_file_vector = {}

    all_nearest_data = index.knnQueryBatch(data_set, k=10, num_threads=4)
    named_nearest_neighbors = []
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
                        annoy_file_index_to_file_vector[master_index] = data_set[master_index]
                        annoy_file_index_to_file_name[master_index] = master_file_path
                        annoy_file_index_to_file_vector[neighbor_index] = data_set[neighbor_index]
                        annoy_file_index_to_file_name[neighbor_index] = neighbor_file_path
    annoy_nearest_neighbor = annoy_similarity(annoy_file_index_to_file_name,annoy_file_index_to_file_vector)
    print("Step.2 - Similarity score calculation - Finished ")
    # Writes the 'named_nearest_neighbors' to a json file
    with open('nearest_neighbors.json', 'w') as out:
        json.dump(list(annoy_nearest_neighbor), out)
    print("Step.3 - Data stored in 'nearest_neighbors.json' file ")
    print("--- Prosess completed in %.2f minutes ---------" %
        ((time.time() - start_time) / 60))