# cluster_image_feature_vectors.py
#################################################
# Imports and function definitions
#################################################
# Numpy for loading image feature vectors from file
import numpy as np
# Time for measuring the process time
import time
# Glob for reading file names in a folder
import glob
import os.path
# json for storing data in json file
import json
# Annoy and Scipy for similarity calculation
from annoy import AnnoyIndex
from scipy import spatial
import threading
import multiprocessing
from multiprocessing import Pool
import tensorflow as tf


#################################################
#################################################
# This function reads from 'image_data.json' file
# Looks for a specific 'filename' value
# Returns the product id when product image names are matched
# So it is used to find product id based on the product image name
#################################################
def is_legit_token(master_file_name,neighbor_file_name,map_of_token_with_owners):
    try:
        master_token = master_file_name.split("_")[0]
        neighbor_token = neighbor_file_name.split("_")[0]
        master_token_owners = map_of_token_with_owners.get(master_token);
        neighbor_token_owners = map_of_token_with_owners.get(neighbor_token);

        for master_address in master_token_owners:
            if ((master_address != '0x0000000000000000000000000000000000000000') and (master_address == neighbor_token_owners.get(master_address))):
                return True;
    except Exception as error:
        print(error)
        return True;            
    
    return False;


def is_already_exist(master_file_name,neighbor_file_name,neighbor_master_set):
    new_neighbor_master_set = frozenset([master_file_name,neighbor_file_name])
    if new_neighbor_master_set not in neighbor_master_set:
        neighbor_master_set[new_neighbor_master_set] = new_neighbor_master_set
        return False;
    else:
        return True;


def create_annoy_index(files_list,start_time,start_index,stop_index):
    for file_index in range(start_index,stop_index):
        file_path = files_list[file_index]
        # Reads feature vectors and assigns them into the file_vector
        file_vector = np.loadtxt(file_path)

        # Assigns file_name, feature_vectors and corresponding product_id
        
        file_name = os.path.basename(file_path).split('.')[0]
        file_index_to_file_name[file_index] = file_name
        file_index_to_file_vector[file_index] = file_vector

        # Adds image feature vectors into annoy index
        # lock.acquire()
        t.add_item(file_index, file_vector)
        # lock.release()
        print("---------------------------------")
        print("Annoy index     : %s" % file_index)
        print("Image file name : %s" % file_name)
        print("--- %.2f minutes passed ---------" % ((time.time() - start_time) / 60))
        # if file_index == trees:
        #     break
#################################################

def similarity_check(start_time,file_index_to_file_name,file_index_to_file_vector,start_index,end_index,map_of_token_with_owners,neighbor_master_set,named_nearest_neighbors):
    dims = 1792
    n_nearest_neighbors = 20
    trees = 15000
    t = AnnoyIndex(dims, metric='angular')
    t.load(".\\annoy_trees")
    # Loops through all indexed items
    for i in range(start_index,end_index):
        # Assigns master file_name, image feature vectors
        # and product id values
        master_file_name = file_index_to_file_name[i]
        master_vector = file_index_to_file_vector[i]
        # Calculates the nearest neighbors of the master item
        nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)
        # Loops through the nearest neighbors of the master item
        for j in nearest_neighbors:
            print(j)
            # Assigns file_name, image feature vectors and
            # product id values of the similar item
            neighbor_file_name = file_index_to_file_name[j]
            neighbor_file_vector = file_index_to_file_vector[j]

            # Calculates the similarity score of the similar item
            similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
            rounded_similarity = int((similarity * 10000)) / 10000.0

            # Appends master product id with the similarity score
            # and the product id of the similar items
            if ((rounded_similarity >= 0.98) and (master_file_name != neighbor_file_name) 
                and (is_legit_token(master_file_name,neighbor_file_name,map_of_token_with_owners) == False)
                and (is_already_exist(master_file_name,neighbor_file_name,neighbor_master_set) == False)):
                    named_nearest_neighbors.append({
                            'similarity': rounded_similarity,
                            'master_pi': master_file_name,
                            'similar_pi': neighbor_file_name})

        print("---------------------------------")
        print("Similarity index       : %s" % i)
        print("Master Image file name : %s" % file_index_to_file_name[i])
        print("Nearest Neighbors.     : %s" % nearest_neighbors)
        print("--- %.2f minutes passed ---------" % ((time.time() - start_time) / 60))


def create_multi_processes(start_time):
    number_of_files = len(file_index_to_file_name)
     # creating a lock
    lock = multiprocessing.Lock()
    item_per_process = int(number_of_files/2)
    processes_list = []
    for i in range(1):
        p = multiprocessing.Process(target=similarity_check, args=(start_time,file_index_to_file_name,file_index_to_file_vector,i*item_per_process,i*item_per_process+item_per_process,map_of_token_with_owners,neighbor_master_set,named_nearest_neighbors))
        processes_list.append(p)

    last_process = multiprocessing.Process(target=similarity_check, args=(start_time,file_index_to_file_name,file_index_to_file_vector,1*item_per_process,number_of_files-1*item_per_process,map_of_token_with_owners,neighbor_master_set,named_nearest_neighbors))
    processes_list.append(last_process)

    for pro in processes_list:
        pro.start()
    for pro in processes_list:
        pro.join()
    

   
#################################################
# This function:
# Reads all image feature vectores stored in /feature-vectors/*.npz
# Adds them all in Annoy Index
# Builds ANNOY index
# Calculates the nearest neighbors and image similarity metrics
# Stores image similarity scores with productID in a json file
#################################################
def consumer(lock,queue,named_nearest_neighbors):
    
    while True:
        msg = queue.get()
        if (msg['status'] == 'DONE'):
            break
        master_file_path = msg['master_file_path']
        neighbor_file_path = msg['neighbor_file_path']
        # print("---------------------------------")
        # print("Step.1 - ANNOY index generation - Started at %s"
        #     % time.ctime())
        # print("---------------------------------")
        
        master_file_vector = np.loadtxt(master_file_path)
        neighbor_file_vector = np.loadtxt(neighbor_file_path)

        # Assigns file_name, feature_vectors and corresponding product_id
        
        master_file_name = os.path.basename(master_file_path).split('.')[0]
        neighbor_file_name = os.path.basename(neighbor_file_path).split('.')[0]
    
        # print("Step.1 - ANNOY index generation - Finished")
        # print("Step.2 - Similarity score calculation - Started ")
        
        
        similarity = 1 - spatial.distance.cosine(master_file_vector, neighbor_file_vector)
        rounded_similarity = int((similarity * 10000)) / 10000.0

        if rounded_similarity >= 0.98:
            lock.acquire()
            named_nearest_neighbors.append({
                                'similarity': rounded_similarity,
                                'master_pi': master_file_name,
                                'similar_pi': neighbor_file_name})
            lock.release()
            return {
                    'similarity': rounded_similarity,
                    'master_pi': master_file_name,
                    'similar_pi': neighbor_file_name}
    
number_of_processes = 6
def producer(queue,start,end):
    allfiles = glob.glob('.\\vectors\\*.npz')
    for master_file_index in range(start,end):
        for neighbor_file_index in range (master_file_index,7000):
            print(master_file_index,"_",neighbor_file_index)
            queue.put({'status':'working','master_file_path':allfiles[master_file_index],'neighbor_file_path':allfiles[neighbor_file_index]})
            

    for _ in range(number_of_processes):
        queue.put({'status':'DONE'})
#################################################
# Global variables ##############################

if __name__ == '__main__':
    # with tf.device('/gpu:0'):
        start_time = time.time()
        allfiles = glob.glob('.\\vectors\\*.npz')
        total_files = len(allfiles)
  

        
       
            

        print("Step.2 - Similarity score calculation - Finished ")
        # Writes the 'named_nearest_neighbors' to a json file
        with open('nearest_neighbors.json', 'w') as out:
            json.dump(list(named_nearest_neighbors), out)
        print("Step.3 - Data stored in 'nearest_neighbors.json' file ")
        print("--- Prosess completed in %.2f minutes ---------" %
            ((time.time() - start_time) / 60))
            
            # manager = multiprocessing.Manager()
            # Defining data structures as empty dict
            
            
            # neighbor_master_set = manager.dict()
        
       