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




#################################################
#################################################
# This function reads from 'image_data.json' file
# Looks for a specific 'filename' value
# Returns the product id when product image names are matched
# So it is used to find product id based on the product image name
#################################################
def is_legit_token(master_file_name,neighbor_file_name):
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

neighbor_master_set = set();
def is_already_exist(master_file_name,neighbor_file_name):
    new_neighbor_master_set = frozenset([master_file_name,neighbor_file_name])
    if new_neighbor_master_set not in neighbor_master_set:
        neighbor_master_set.add(new_neighbor_master_set)
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
        annoy_list[0].add_item(file_index, file_vector)
        # lock.release()
        print("---------------------------------")
        print("Annoy index     : %s" % file_index)
        print("Image file name : %s" % file_name)
        print("--- %.2f minutes passed ---------" % ((time.time() - start_time) / 60))
        # if file_index == trees:
        #     break
#################################################
#################################################
# This function:
# Reads all image feature vectores stored in /feature-vectors/*.npz
# Adds them all in Annoy Index
# Builds ANNOY index
# Calculates the nearest neighbors and image similarity metrics
# Stores image similarity scores with productID in a json file
#################################################
def cluster():
    start_time = time.time()

    print("---------------------------------")
    print("Step.1 - ANNOY index generation - Started at %s"
          % time.ctime())
    print("---------------------------------")

    number_of_files = len(allfiles)

    # creating a lock
    lock = threading.Lock()
    item_per_thread = int(number_of_files/16)
    thread_list = []
    for i in range(8):
        th = multiprocessing.Process(target=create_annoy_index, args=(allfiles,start_time,i*item_per_thread,i*item_per_thread+item_per_thread,))
        thread_list.append(th)

    last_thread = multiprocessing.Process(target=create_annoy_index, args=(allfiles,start_time,16*item_per_thread,number_of_files-16*item_per_thread,))
    thread_list.append(last_thread)

    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()
    

    # Builds annoy index
    t.build(trees)

    print("Step.1 - ANNOY index generation - Finished")
    print("Step.2 - Similarity score calculation - Started ")
    
    named_nearest_neighbors = []
    # Loops through all indexed items
    for i in file_index_to_file_name.keys():
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
            if ((rounded_similarity >= 0.9) and (master_file_name != neighbor_file_name) 
                and (is_legit_token(master_file_name,neighbor_file_name) == False)
                and (is_already_exist(master_file_name,neighbor_file_name) == False)):
                    named_nearest_neighbors.append({
                            'similarity': rounded_similarity,
                            'master_pi': master_file_name,
                            'similar_pi': neighbor_file_name})

        print("---------------------------------")
        print("Similarity index       : %s" % i)
        print("Master Image file name : %s" % file_index_to_file_name[i])
        print("Nearest Neighbors.     : %s" % nearest_neighbors)
        print("--- %.2f minutes passed ---------" % ((time.time() - start_time) / 60))

    print("Step.2 - Similarity score calculation - Finished ")
    # Writes the 'named_nearest_neighbors' to a json file
    with open('nearest_neighbors.json', 'w') as out:
        json.dump(named_nearest_neighbors, out)
    print("Step.3 - Data stored in 'nearest_neighbors.json' file ")
    print("--- Prosess completed in %.2f minutes ---------" %
          ((time.time() - start_time) / 60))

#################################################
# Global variables ##############################

if __name__ == '__main__':
    with open('..\\copyHunter\\scripts\\owners_data.json') as json_file:
        map_of_token_with_owners = json.load(json_file)

    # Reads all file names which stores feature vectors
    allfiles = glob.glob('.\\vectors\\*.npz')

    # Defining data structures as empty dict
    file_index_to_file_name = {}
    file_index_to_file_vector = {}
    # Configuring annoy parameters
    dims = 1792
    n_nearest_neighbors = 20
    trees = 15000
    manager = multiprocessing.Manager()
    annoy_list = manager.list()
    annoy_list.append(AnnoyIndex(dims, metric='angular'))
    try:
        cluster()
    except Exception as error:
        print(error)
