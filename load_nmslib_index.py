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
import hnswlib
from json import JSONEncoder
from os import path

logging.basicConfig(level=logging.DEBUG)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
def load_glob_worker(files,file_name_index,image_vector_features,start,end,lock):
    # all_files = glob.glob('.\\vectors_test\\*.npz')
    n = random.randint(3,10)
    vector_data = []
    for index in range(start,end):
        image_vector_features.append(np.loadtxt(files[index]))
        # file_name = os.path.basename(all_files[index]).split('.')[0]
        file_name_index[str(index)] = files[index]


def is_legit_token(master_file_name,neighbor_file_name):
    try:
        master_token = master_file_name.split(".")[0]
        neighbor_token = neighbor_file_name.split(".")[0]
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
    
    n_nearest_neighbors = 10
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


resouces_folder = '.\\prod_data\\'
with open(resouces_folder+'addresses_tokens_owners_data.json') as json_file:
    map_of_token_with_owners = json.load(json_file)
neighbor_master_set = {}
dims = 1792
if __name__ == '__main__':
    # load image vector files
    
    lock = multiprocessing.Lock()
    
    data_set = []
    file_name_index = {}
    start_time = time.time()


    
    vector_features_file_path = resouces_folder+"vectors_features.json"
    vector_index_file_names_path = resouces_folder+"index_file_names.json"
    index_file_path = resouces_folder+'hnswlib.bin'
    vectors_folder = resouces_folder+"vectors\\*.npz"
    nearest_neighbors_file_path = resouces_folder+'nearest_neighbors.json'
    if path.exists(vector_features_file_path):
        with open(vector_features_file_path, "r") as read_file:
            data_set = numpy.asarray(json.load(read_file)['data'])
        with open(vector_index_file_names_path, "r") as read_file:
            file_name_index = json.load(read_file)
        # file_name_index = json.loads(vector_index_file_names_path)
    else:
        all_files = glob.glob(vectors_folder)
        total_files = len(all_files)
        print(total_files)
        load_glob_worker(all_files,file_name_index,data_set,0,total_files,lock)
        with open(vector_features_file_path, 'w') as out:
            json.dump({'data':data_set},  out,cls=NumpyArrayEncoder,)
        with open(vector_index_file_names_path, 'w') as out:
            json.dump(file_name_index,  out)
    print("======================",len(data_set))
    total_files = len(data_set)
    index = hnswlib.Index(space='cosine', dim=dims) # possible options are l2, cosine or ip
    if path.exists(index_file_path) == False:
        index.init_index(max_elements = total_files,ef_construction = 2000, M = 16)
        index.add_items(data_set)
        index.set_ef(50)
        index.save_index(index_file_path)
    else:
        index.load_index(index_file_path,max_elements = len(data_set))
            
    
    # del index

    # second_data_set_files = glob.glob('.\\vectors_gif_2\\*.npz')
    # second_data_set = []
    # second_file_name_index = {}
    # load_glob_worker(second_data_set_files,second_file_name_index,second_data_set,0,len(second_data_set_files),lock)
    # for new_feature_index,new_file_feature in enumerate(second_data_set):
    #     data_set.append(new_file_feature)
    #     file_name_index[total_files+new_feature_index] = second_file_name_index[new_feature_index]

   
    # index = hnswlib.Index(space='cosine', dim=dims) # possible options are l2, cosine or ip
    # index.load_index(index_path,max_elements = len(data_set))
    # index.add_items(second_data_set)

    # labels, distances = p.knn_query(data_set, k = 5)
    # index.createIndex({'post': 2,'efConstruction':1000,'M':50,}, print_progress=True)
    # index.createIndex({'numPivot':5000,'numPivotIndex':5000})
    print(index)
 

    
   
    # get all nearest neighbours for all the datapoint
    # using a pool of 4 threads to compute
    # Defining data structures as empty dict
    annoy_file_index_to_file_name = {}
    annoy_file_index_to_file_vector = {}

    # all_nearest_data = index.knnQueryBatch(data_set, k=10, num_threads=4)
    labels, distances = index.knn_query(data_set, k = 10) 
    named_nearest_neighbors = []
    for master_index, master in enumerate(distances):
       
        for index_of_distance_to_neighbor, distance_to_neighbor in enumerate(master):
            # master_neighbors_index =master[0]
            # master_neighbors_distances = master[1]
            # if index_of_distance_to_neighbor != 0:
            if distance_to_neighbor <= 0.02:
                master_file_path = file_name_index[str(master_index)]
                neighbor_index = labels[master_index][index_of_distance_to_neighbor]
                neighbor_file_path = file_name_index[str(neighbor_index)]
                if( master_file_path != neighbor_file_path):
               
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
    with open(nearest_neighbors_file_path, 'w') as out:
        json.dump(list(annoy_nearest_neighbor), out)
    print("Step.3 - Data stored in 'nearest_neighbors.json' file ")
    print("--- Prosess completed in %.2f minutes ---------" %
        ((time.time() - start_time) / 60))