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

            out_path = os.path.join('.\\vectors_test\\', outfile_name)
            # Saves the 'feature_set' to a text file
            np.savetxt(out_path, feature_set, delimiter=',')
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

def img_processing_worker(img_file_paths_queue:multiprocessing.Queue):
    # Definition of module with using tfhub.dev
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    # Loads the module
    tfhub_module = hub.load(module_handle)
    while True:
        try:

            file_name,img_content = img_file_paths_queue.get()
            print("================== processing",file_name)
            vectorized_image(file_name,img_content,tfhub_module)
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    # Simple demo that scans all the token transfers of RCC token (11k).
    # The demo supports persistant state by using a JSON file.
    # You will need an Ethereum node for this.
    # Running this script will consume around 20k JSON-RPC calls.
    # With locally running Geth, the script takes 10 minutes.
    # The resulting JSON state file is 2.9 MB.
    import sys
    import json
    from web3.providers.rpc import HTTPProvider

    # We use tqdm library to render a nice progress bar in the console
    # https://pypi.org/project/tqdm/
    from tqdm import tqdm

    # RCC has around 11k Transfer events
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
        download_worker = multiprocessing.Process(target=download_images_worker,args=(img_urls_queue,img_file_paths_queue))
        processing_worker = multiprocessing.Process(target=img_processing_worker,args=(img_file_paths_queue,))
        download_worker.start()
        processing_worker.start()
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
        start_block = 12325024 # max(state.get_last_scanned_block() - chain_reorg_safety_blocks, 0)
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

        state.save()
        download_worker.join()
        processing_worker.join()
        duration = time.time() - start
        print(f"Scanned total {len(result)} Transfer events, in {duration} seconds, total {total_chunks_scanned} chunk scans performed")

    run()
