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
from constants import resources_folder
import time
import gc
import httplib2
from multiprocessing.managers import BaseManager
import threading

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def is_vector_feature_exist(address_token):
            address_token = resources_folder+"vectors\\"+address_token+".npz"
            return os.path.exists(address_token)
                
          
def download_image(gateway,address_token,url):
        dot_location =  url.lower().rfind(".")
        file_type = url[dot_location:] if dot_location > 0 else ""
        if url.startswith('http') == False:
            url = gateway+url[url.rindex('ipfs'):]

        file_name_with_extension = address_token+file_type
        if is_vector_feature_exist(file_name_with_extension)  == True:
            return file_name_with_extension,None
        requestMaker = httplib2.Http(".cache")
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}
        (res_headers,image_response) = requestMaker.request(url,"GET",headers=headers)
        
        if res_headers.status == 200 and len(image_response) > 0:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            # image_response.raw.decode_content = True
            
            # Open a local file with wb ( write binary ) permission.
            # with open("test_image"+file_type,'wb') as f:
            #     shutil.copyfileobj(image_response.raw, f)
            return file_name_with_extension, image_response
        else:
            if len(image_response) == 0:
                print("empty image")
            return None,None


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

            out_path = os.path.join(resources_folder+'.\\vectors\\', outfile_name)
            # Saves the 'feature_set' to a text file
            np.savetxt(out_path, feature_set, delimiter=',')
            del decoded_img
            del features
            # del feature_set
            # return None
            return feature_set

def download_process(img_urls_queue:multiprocessing.Queue, img_file_paths_queue:multiprocessing.Queue,retry_queue:multiprocessing.Queue):
    thread_1 = threading.Thread(target=download_images_worker,args=(img_urls_queue,img_file_paths_queue,retry_queue))
    thread_1.start()
    download_images_worker(img_urls_queue,img_file_paths_queue,retry_queue)
    thread_1.join()
def download_images_worker(img_urls_queue:multiprocessing.Queue, img_file_paths_queue:multiprocessing.Queue,retry_queue:multiprocessing.Queue):
    gateway_queue = Queue()
    gateway_queue.put('https://ipfs.io/')
    gateway_queue.put('https://gateway.ipfs.io/')
    gateway_queue.put('https://ipfs.drink.cafe/')
    gateway_queue.put('https://dweb.link/')
    gateway_queue.put('https://infura.io/')
    gateway_queue.put('https://gateway.pinata.cloud/')
    gateway_queue.put('https://ipfs.fleek.co/')
    gateway_queue.put('https://cloudflare-ipfs.com/')
    gateway_queue.put('https://ipfs.denarius.io/')
    gateway_queue.put('https://cf-ipfs.com/')
    while True:
        try:
            
            gateway = gateway_queue.get()
            address_token,url,retry_number = img_urls_queue.get()
            if address_token is None and url is None:
                img_file_paths_queue.put((None,None))
                break
            print("############### downloading {} from {}".format(address_token,url))
            address_token_img_type,img_content = download_image(gateway,address_token,url)
            if address_token_img_type is not None:
                img_file_paths_queue.put((address_token_img_type,img_content))
            else:
                print("Unable to download image {} from url {}".format(address_token,url))
            del img_content
        except Exception as e:
            print("Exception while downloading image:",e)
            # put it back so it will be handle by other gateways
            # TODO: need to check specific error
            try:
                if retry_number <= 10:
                    retry_number += 1
                    retry_queue.put_nowait((address_token,url,retry_number))
            except Exception as e:
                print('Error while putting retry file into retry queue',e)
            time.sleep(5)    
            continue
        finally:
            gateway_queue.put(gateway)
    print("Finished downloading images")

def img_vectorizing_worker(img_file_paths_queue:multiprocessing.Queue,features_queue:Queue):
    # Definition of module with using tfhub.dev
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    # Loads the module
    tfhub_module = hub.load(module_handle)
    while True:
        try:
            
            address_token_img_type,img_content = img_file_paths_queue.get()
            if address_token_img_type is None and img_content is None:
                features_queue.put((None,None))
                break
            
            if img_content is not None:
                print("=============== vectorizing",address_token_img_type)
                features_set = vectorized_image(address_token_img_type,img_content,tfhub_module)
                features_queue.put((address_token_img_type,features_set))
            else:
                print("=============== loading vector feature",address_token_img_type)
                features_set = np.loadtxt(resources_folder+"vectors\\"+address_token_img_type+".npz")

                features_queue.put((address_token_img_type,features_set))
            del img_content
            gc.collect()
            
        except Exception as e:
            print("Exception while processing image:",e)
            continue
    print("Finished processing images")


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

        api_url = 'https://mainnet.infura.io/v3/efdca0ddba8943dbbff758c6e75e4ed9'#sys.argv[1]

        # Enable logs to the stdout.
        # DEBUG is very verbose level
        logging.basicConfig(level=logging.INFO)

        provider = HTTPProvider(api_url)

        # Remove the default JSON-RPC retry middleware
        # as it correctly cannot handle eth_getLogs block range
        # throttle down.
        provider.middlewares.clear()

        web3 = Web3(provider)
        multiprocessing_manager = multiprocessing.Manager()

        # Prepare stub ERC-20 contract object
        abi = json.loads(ABI)
        ERC721 = web3.eth.contract(abi=abi)

        # Restore/create our persistent state
        state = JSONifiedState()
        state.restore()
        queue_size = 20
        img_urls_queue = multiprocessing.Queue(maxsize=queue_size)
        img_file_paths_queue = multiprocessing.Queue(maxsize=queue_size)
        retry_queue = multiprocessing.Queue(maxsize=0)

        BaseManager.register('get_features_queue')
        BaseManager.register('get_indexed_files_dict')
        m = BaseManager(address=('192.168.0.4', 50000), authkey=b'abracadabra')
        m.connect()
        features_queue = m.get_features_queue()
        indexed_files = m.get_indexed_files_dict()
       
        download_1 = multiprocessing.Process(target=download_process,args=(img_urls_queue,img_file_paths_queue,retry_queue))
        # download_2 = multiprocessing.Process(target=download_images_worker,args=(img_urls_queue,img_file_paths_queue,retry_queue))
        # download_3 = multiprocessing.Process(target=download_images_worker,args=(img_urls_queue,img_file_paths_queue,retry_queue))
        # download_4 = multiprocessing.Process(target=download_images_worker,args=(img_urls_queue,img_file_paths_queue,retry_queue))
        # download_5 = multiprocessing.Process(target=download_images_worker,args=(img_urls_queue,img_file_paths_queue,retry_queue))

        processing_worker = multiprocessing.Process(target=img_vectorizing_worker,args=(img_file_paths_queue,features_queue))
        
        download_1.start() 
        # download_2.start()
        # download_3.start()
        # download_4.start()
        # download_5.start()
        processing_worker.start()
      
        # chain_id: int, web3: Web3, abi: dict, state: EventScannerState, events: List, filters: {}, max_chunk_scan_size: int=10000
        scanner = EventScanner(
            indexed_files=indexed_files,
            retry_queue=retry_queue,
            img_urls_queue=img_urls_queue,
            web3=web3,
            contract=ERC721,
            state=state,
            events=[ERC721.events.Transfer],
            filters={"address": RCC_ADDRESS},
            # How many maximum blocks at the time we request from JSON-RPC
            # and we are unlikely to exceed the response size limit of the JSON-RPC server
            # max_chunk_scan_size=10000
            max_chunk_scan_size=1000
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
        start_block = 11881985  # max(state.get_last_scanned_block() - chain_reorg_safety_blocks, 0)
        end_block = 11883985# scanner.get_suggested_scan_end_block()
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
            img_urls_queue.put((None,None,None))
            img_urls_queue.put((None,None,None))
            # img_urls_queue.put((None,None,None))
            # img_urls_queue.put((None,None,None))
            # img_urls_queue.put((None,None,None))

        state.save()
        
        download_1.join()
        # download_2.join()
        # download_3.join()
        # download_4.join()
        # download_5.join()
        processing_worker.join()
        
        # update_indexed_item_dict_worker.join()

        duration = time.time() - start
        print(f"Scanned total {len(result)} Transfer events, in {duration} seconds, total {total_chunks_scanned} chunk scans performed")

    run()
