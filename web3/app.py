import datetime
import time
import logging
import requests
import shutil
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os.path
import multiprocessing
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable, List, Iterable
from jsonified_state import JSONifiedState
from event_scanner_state import EventScannerState
from web3 import Web3
from event_scanner import EventScanner
from web3.contract import Contract
from web3.datastructures import AttributeDict
from web3.exceptions import BlockNotFound
from eth_abi.codec import ABICodec
from queue import Queue
import hnswlib
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def download_image(gateway,file_name,url):
        if url.startswith('http') == False:
            dot_location =  url.find(".")
            file_type = url[dot_location:] if dot_location > 0 else ""
            url = gateway+url[url.rindex('ipfs'):]
        
        image_response = requests.get(url,stream = True)
        
        if image_response.status_code == 200:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            image_response.raw.decode_content = True
            
            # Open a local file with wb ( write binary ) permission.
            # with open("test_image"+file_type,'wb') as f:
            #     shutil.copyfileobj(image_response.raw, f)
            return file_name+file_type, image_response.content


def vectorized_image(image_name,image_content,tfhub_module):
            decoded_img = tf.io.decode_image(image_content,channels=3,expand_animations=False)
             # Resizes the image to 224 x 224 x 3 shape tensor
            decoded_img = tf.image.resize_with_pad(decoded_img, 224, 224)
            # Converts the data type of uint8 to float32 by adding a new axis
            # img becomes 1 x 224 x 224 x 3 tensor with data type of float32
            # This is required for the mobilenet model we are using
            decoded_img = tf.image.convert_image_dtype(decoded_img, tf.float32)[tf.newaxis, ...]

            
            features = tfhub_module(decoded_img)
            # Remove single-dimensional entries from the 'features' array
            feature_set = np.squeeze(features)
           

            # Saves the image feature vectors into a file for later use
            outfile_name = os.path.basename(image_name) + ".npz"

            out_path = os.path.join(resouces_folder+'.\\vectors\\', outfile_name)
            # Saves the 'feature_set' to a text file
            np.savetxt(out_path, feature_set, delimiter=',')
            return feature_set
def download_images_worker(img_urls_queue:multiprocessing.Queue, img_file_paths_queue:multiprocessing.Queue):
    gateway_queue = Queue(maxsize=5)
    gateway_queue.put('https://ipfs.io/')
    gateway_queue.put('https://gateway.ipfs.io/')
    gateway_queue.put('https://ipfs.drink.cafe/')
    gateway_queue.put('https://dweb.link/')
    
    while True:
        try:
            gateway = gateway_queue.get()
            file_name,url = img_urls_queue.get()
            if file_name is None and url is None:
                img_file_paths_queue.put((None,None))
                break
            print("################### downloading",url)
            # download_image(gateway,file_name,url)
            img_file_paths_queue.put(download_image(gateway,file_name,url))
        except Exception as e:
            print(e)
            # put it back so it will be handle by other gateways
            # TODO: need to check specific error
            img_urls_queue.put((file_name,url))    
            continue
        finally:
            gateway_queue.put(gateway)
    print("Finished downloading images")

def img_processing_worker(img_file_paths_queue:multiprocessing.Queue,vector_feature_queue:multiprocessing.Queue):
    # Definition of module with using tfhub.dev
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    # Loads the module
    tfhub_module = hub.load(module_handle)
    while True:
        try:

            file_name,img_content = img_file_paths_queue.get()
            if file_name is None and img_content is None:
                vector_feature_queue.put((None,None))
                break
            print("================== processing",file_name)
            features_set = vectorized_image(file_name,img_content,tfhub_module)
            vector_feature_queue.put((file_name,features_set))
        except Exception as e:
            print(e)
            continue
    print("Finished processing images")


def vectors_indexing_worker(vector_feature_queue:multiprocessing.Queue):
    data_set = []
    file_name_index = {}
    start_time = time.time()
    dims = 1792
    
    vector_features_file_path = resouces_folder+"vectors_features.json"
    vector_index_file_names_path = resouces_folder+"index_file_names.json"
    index_file_path = resouces_folder+'hnswlib.bin'

    if os.path.exists(vector_features_file_path):
        with open(vector_features_file_path, "r") as read_file:
            data_set = list(np.asarray(json.load(read_file)['data']))
        with open(vector_index_file_names_path, "r") as read_file:
            file_name_index = json.load(read_file)
    else:
        print("Vectors features not found. Exit")
        exit()
    total_files = len(data_set)
    index = hnswlib.Index(space='cosine', dim=dims) # possible options are l2, cosine or ip
    if os.path.exists(index_file_path) == False:
        index.init_index(max_elements = total_files,ef_construction = 2000, M = 16)
        index.add_items(data_set)
        index.set_ef(50)
        index.save_index(index_file_path)
    else:
        index.load_index(index_file_path,max_elements = len(data_set))
    
    counter = 0    
    while True:
        try:
            file_name,features_set = vector_feature_queue.get()
            if file_name is None and features_set is None:
                break
            index.resize_index(index.get_max_elements()+1)
            index.add_items([features_set])
            data_set.append(features_set)
            file_name_index[str(len(data_set)-1)] = resouces_folder + "vectors\\"+file_name
            if counter == 1:
                index.save_index(index_file_path)
                with open(vector_features_file_path, 'w') as out:
                    json.dump({'data':data_set},  out,cls=NumpyArrayEncoder,)
                with open(vector_index_file_names_path, 'w') as out:
                    json.dump(file_name_index,  out)
                counter = 0
            else:
                counter += 1
        except Exception as e:
            print(e)    
            continue
    print("Finished indexing images")

resouces_folder = '.\\test_data\\'
if __name__ == "__main__":
    
    import sys
    import json
    from web3.providers.rpc import HTTPProvider

    # We use tqdm library to render a nice progress bar in the console
    # https://pypi.org/project/tqdm/
    from tqdm import tqdm

    
    # https://etherscan.io/token/0x9b6443b0fb9c241a7fdac375595cea13e6b7807a
    RCC_ADDRESS = Web3.toChecksumAddress("0x60f80121c31a0d46b5279700f9df786054aa5ee5")
    # RCC_ADDRESS = Web3.toChecksumAddress("0x9C008A22D71B6182029b694B0311486e4C0e53DB")

    # Reduced ERC-20 ABI, only Transfer event
    ABI = """[
        {
            "anonymous": false,
            "inputs": [
                {"indexed": true, "name": "from", "type": "address"},
                {"indexed": true, "name": "to", "type": "address"},
                {"indexed": true, "name": "tokenId", "type": "uint256"}],
            "name": "Transfer",
            "type": "event"
        }
    ]"""

    

    def run():

        # if len(sys.argv) < 2:
        #     print("Usage: eventscanner.py http://your-node-url")
        #     sys.exit(1)

        api_url = 'https://mainnet.infura.io/v3/a671a77998514426b1ca3733157fb5ab'#sys.argv[1]

        # Enable logs to the stdout.
        # DEBUG is very verbose level
        logging.basicConfig(level=logging.INFO)

        provider = HTTPProvider(api_url)

        # Remove the default JSON-RPC retry middleware
        # as it correctly cannot handle eth_getLogs block range
        # throttle down.
        provider.middlewares.clear()

        web3 = Web3(provider)

        # Prepare stub ERC-20 contract object
        abi = json.loads(ABI)
        ERC721 = web3.eth.contract(abi=abi)

        # Restore/create our persistent state
        state = JSONifiedState()
        state.restore()
        img_urls_queue = multiprocessing.Queue(maxsize=10)
        img_file_paths_queue = multiprocessing.Queue(maxsize=10)
        vector_feature_queue = multiprocessing.Queue(maxsize=10)
        download_worker = multiprocessing.Process(target=download_images_worker,args=(img_urls_queue,img_file_paths_queue))
        processing_worker = multiprocessing.Process(target=img_processing_worker,args=(img_file_paths_queue,vector_feature_queue))
        indexing_worker = multiprocessing.Process(target=vectors_indexing_worker,args=(vector_feature_queue,))
        download_worker.start()
        processing_worker.start()
        indexing_worker.start()
        # chain_id: int, web3: Web3, abi: dict, state: EventScannerState, events: List, filters: {}, max_chunk_scan_size: int=10000
        scanner = EventScanner(
            img_urls_queue=img_urls_queue,
            web3=web3,
            contract=ERC721,
            state=state,
            events=[ERC721.events.Transfer],
            filters={"address": RCC_ADDRESS},
            # How many maximum blocks at the time we request from JSON-RPC
            # and we are unlikely to exceed the response size limit of the JSON-RPC server
            # max_chunk_scan_size=10000
            max_chunk_scan_size=100
        )

        # Assume we might have scanned the blocks all the way to the last Ethereum block
        # that mined a few seconds before the previous scan run ended.
        # Because there might have been a minor Etherueum chain reorganisations
        # since the last scan ended, we need to discard
        # the last few blocks from the previous scan results.
        chain_reorg_safety_blocks = 10
        scanner.delete_potentially_forked_block_data(state.get_last_scanned_block() - chain_reorg_safety_blocks)

        # Scan from [last block scanned] - [latest ethereum block]
        # Note that our chain reorg safety blocks cannot go negative
        start_block = 12326024 # max(state.get_last_scanned_block() - chain_reorg_safety_blocks, 0)
        end_block = 12326224 #scanner.get_suggested_scan_end_block()
        blocks_to_scan = end_block - start_block

        print(f"Scanning events from blocks {start_block} - {end_block}")

        # Render a progress bar in the console
        start = time.time()
        with tqdm(total=blocks_to_scan) as progress_bar:
            def _update_progress(start, end, current, current_block_timestamp, chunk_size, events_count):
                if current_block_timestamp:
                    formatted_time = current_block_timestamp.strftime("%d-%m-%Y")
                else:
                    formatted_time = "no block time available"
                progress_bar.set_description(f"Current block: {current} ({formatted_time}), blocks in a scan batch: {chunk_size}, events processed in a batch {events_count}")
                progress_bar.update(chunk_size)

            # Run the scan
            result, total_chunks_scanned = scanner.scan(start_block, end_block, progress_callback=_update_progress)
            img_urls_queue.put((None,None))

        state.save()
        
        download_worker.join()
        processing_worker.join()
        indexing_worker.join()
        duration = time.time() - start
        print(f"Scanned total {len(result)} Transfer events, in {duration} seconds, total {total_chunks_scanned} chunk scans performed")

    run()