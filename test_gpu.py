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

def worker(all_vectors_data_set,start,end):
    # vectors_data_set = tf.data.Dataset.from_tensors(all_vectors_data_set)
    # batch_vectors_data_set = vectors_data_set
    for count_batch in batch_vectors_data_set:
        print("===========================")
        # print(count_batch)
        print(tf.keras.losses.cosine_similarity(count_batch,all_vectors_data_set,axis=-1))


    # all_files = glob.glob('.\\vectors_test\\*.npz')
    # for file_index in range(start,end):
    #     file_vector = np.loadtxt(all_files[file_index])
    #     # print(file_vector)
    #     # vectors_data_set.append(tf.data.Dataset.from_tensors(file_vector))
    #     vectors_data_set.append(file_vector)

    # for file_index in range(start,end):
    #     print(tf.keras.losses.cosine_similarity(all_vectors_data_set[file_index],all_vectors_data_set,axis=-1))



def load_glob_worker(all_vectors_data_set,start,end,lock):
    all_files = glob.glob('.\\vectors\\*.npz')
    n = random.randint(3,10)
    vector_data = []
    for file_name in range(start,end):
        if n > 0:
            vector_data.append(np.loadtxt(all_files[file_name]))
            n -= 1
        else:    
            lock.acquire()
            all_vectors_data_set.extend(vector_data)
            lock.release()
            n = random.randint(3,10)
            vector_data.clear()


if __name__ == '__main__':
    all_files = glob.glob('.\\vectors\\*.npz')
    total_files = 6000#len(all_files)
    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    all_vectors_data_set = manager.list()
    start_time = time.time()
    
    number_of_processes = 6
    processes_list = []
    item_per_process = int(total_files/number_of_processes)

    for p in range(number_of_processes):
        processes_list.append(multiprocessing.Process(target=load_glob_worker, args=(all_vectors_data_set,p*item_per_process,p*item_per_process+item_per_process,lock)))

    for p in range(number_of_processes):
        processes_list[p].start()
    
    for p in range(number_of_processes):
        processes_list[p].join()
    print("======================",len(all_vectors_data_set))
    
    with SharedMemoryManager() as smm:
        # Create a shared memory of size np_arry.nbytes
        
        # vectors_data = list(all_vectors_data_set)
        # print(vectors_data.__sizeof__())
        # shm = smm.SharedMemory(len(all_vectors_data_set)*1792*sys.getsizeof(vectors_data[0]))
        # # b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
        # shm_np_array = np.recarray(shape=(len(all_vectors_data_set),1792), dtype=np.float64, buf=shm.buf)
        # np.copyto(shm_np_array, list(all_vectors_data_set))
        # list(all_vectors_data_set).clear()
        # print(len(shm_np_array))
        
   
        gpu_processes_list = []
        number_of_gpu_processes = 1
        item_per_process = int(len(all_vectors_data_set)/number_of_gpu_processes)
        for p in range(number_of_gpu_processes):
            gpu_processes_list.append(multiprocessing.Process(target=worker, args=(all_vectors_data_set,p*item_per_process,p*item_per_process+item_per_process)))
        for p in range(number_of_gpu_processes):
            gpu_processes_list[p].start()
        
        for p in range(number_of_gpu_processes):
            gpu_processes_list[p].join()
        # worker(all_vectors_data_set,0,total_files)    
        print(total_files)
        # shm.close()
        # shm.unlink()
# print("=================================================")
# for file_path in range(3,6):
#     file_vector = np.loadtxt(all_files[file_path])
#     # print(file_vector)
#     # vectors_data_set.append(tf.data.Dataset.from_tensors(file_vector))
#     vectors_data_set1.append(file_vector)    

    # print(vectors_data_set)

# print("=============== vector * vector")

# print(tf.keras.losses.cosine_similarity(vectors_data_set,vectors_data_set1,axis=-1))
# print("=============== element * vector")
# for vector in vectors_data_set:
#     print(tf.keras.losses.cosine_similarity(vectors_data_set1,vector,axis=-1))
# # print("=============== e * e")
# for v1 in vectors_data_set:
#     print("VVVVVV22222")
#     for v2 in vectors_data_set1:
#         print(tf.keras.losses.cosine_similarity(v1,v2,axis=-1))
# print("=============== spatial")

# for v1 in vectors_data_set:
#     print("VVVVVV22222")
#     for v2 in vectors_data_set1:
#         similarity = 1 - spatial.distance.cosine(v1, v2)
#         rounded_similarity = int((similarity * 10000)) / 10000.0
#         print(rounded_similarity)
# print(spatial.distance.cosine(vectors_data_set,vectors_data_set1))

# files_name_data_set = tf.data.Dataset.list_files('.\\vectors_test\\*.npz')
# vectors_data_set = files_name_data_set.map(lambda filename: tf.data.Dataset.from_tensors(np.loadtxt(all_files[filename]))).map(lambda string_vals:  tf.strings.to_number(string_vals, tf.float64))
# # vectors_data_set = tf.data.Dataset.from_tensors(vectors_data_set)
# for i in vectors_data_set:
#     print(i)
# print(vectors_data_set)
# # print(list(vectors_data_set.as_numpy_iterator()))

# tensors = [[.34, .45, .67, .65], [.14, .35, .67, .65],
#             [.54, .95, .07, .5], [.64, .75, .81, .05]]

# # Similarity of one vector with all other vectors
# print(tf.keras.losses.cosine_similarity(
#     tensors[0],
#       tensors,
#     axis=-1
# ))