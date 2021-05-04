from event_scanner_state import EventScannerState
from web3 import Web3
import shutil
import json
import datetime
import time
from web3.datastructures import AttributeDict 
import requests
from queue import Queue
from os import path
from constants import resources_folder
from multiprocessing import Manager
class JSONifiedState(EventScannerState):
        """Store the state of scanned blocks and all events.

        All state is an in-memory dict.
        Simple load/store massive JSON on start up.
        """

        def __init__(self):
            # self.resources_folder = resouces_folder
            self.state = None
            self.address_token_owners = {}
            self.non_exist_tokens = {}
            self.tokens_file_name = resources_folder+"addresses_tokens_owners_data.json"
            self.fname = resources_folder+"test-state.json"
            # How many second ago we saved the JSON file
            self.last_save = 0
            self.gateway_queue = Queue(maxsize=5)
            self.gateway_queue.put('https://ipfs.io/')
            self.gateway_queue.put('https://gateway.ipfs.io/')
            self.gateway_queue.put('https://ipfs.drink.cafe/')
            self.gateway_queue.put('https://dweb.link/')

        def reset(self):
            """Create initial state of nothing scanned."""
            self.state = {
                "last_scanned_block": 0,
                "blocks": {},
            }

        def restore(self):
            """Restore the last scan state from a file."""
            try:
                self.state = json.load(open(self.fname, "rt"))
                print(f"Restored the state, previously {self.state['last_scanned_block']} blocks have been scanned")
            except (IOError, json.decoder.JSONDecodeError):
                print("State starting from scratch")
                self.reset()
            
            try:
                self.address_token_owners = json.load(open(self.tokens_file_name,"rt"))
                print("Loaded address_token_owners")
            except Exception as e:
                print(e) 

            try:
                self.non_exist_tokens = json.load(open(resources_folder+'nonExistTokens.json',"rt"))
                print("Loaded non exist tokens")
            except Exception as e:
                print(e)        

        def save(self):
            """Save everything we have scanned so far in a file."""
            with open(self.fname, "wt") as f:
                json.dump(self.state, f)
            self.last_save = time.time()

            with open(self.tokens_file_name, "wt") as fw:
                json.dump(self.address_token_owners, fw)
            
            with open(resources_folder+'nonExistTokens.json', "wt") as fw:
                json.dump(self.non_exist_tokens, fw)


        #
        # EventScannerState methods implemented below
        #

        def get_last_scanned_block(self):
            """The number of the last block we have stored."""
            return self.state["last_scanned_block"]

        def delete_data(self, since_block):
            """Remove potentially reorganised blocks from the scan data."""
            for block_num in range(since_block, self.get_last_scanned_block()):
                if block_num in self.state["blocks"]:
                    del self.state["blocks"][block_num]

        def start_chunk(self, block_number, chunk_size):
            pass

        def end_chunk(self, block_number):
            """Save at the end of each block, so we can resume in the case of a crash or CTRL+C"""
            # Next time the scanner is started we will resume from this block
            self.state["last_scanned_block"] = block_number

            # Save the database file for every minute
            if time.time() - self.last_save > 60:
                self.save()
        
        def get_token_uri(self, web3, contract_address, token_id):
            gateway = self.gateway_queue.get()
            try:
                simplified_abi = [{
                    "constant": True,
                    "inputs": [
                        {
                            "internalType": "uint256",
                            "name": "tokenId",
                            "type": "uint256"
                        }
                    ],
                    "name": "tokenURI",
                    "outputs": [
                        {
                            "internalType": "string",
                            "name": "",
                            "type": "string"
                        }
                    ],
                    "payable": False,
                    "stateMutability": "view",
                    "type": "function"
                }]
                contract = web3.eth.contract(address=web3.toChecksumAddress(contract_address), abi=simplified_abi)
                img_uri = contract.functions.tokenURI(token_id).call()
                token_uri = img_uri
                if img_uri != None and img_uri != '':
                    if img_uri.startswith('http') == False:
                        
                        img_uri = gateway+img_uri[img_uri.rindex('ipfs'):]
                    json_response = requests.get(img_uri, stream = True)
                    img_uri = json_response.json()['image']
                    if img_uri.find('.webp') > 0 or img_uri.find('.mp4') > 0:
                        return None,None
                    else:
                        return token_uri,img_uri
                else:
                    return None,None    
                
            except Exception as e:
                print("Exception while getting token URI",e)
                if str(e).find("nonexistent token") > 0:
                    key = contract_address+"_"+str(token_id)
                    self.non_exist_tokens[key] = key

                return None,None
            finally:
                self.gateway_queue.put(gateway)    
                        
        def add_owners_to_state(self,address_token,owners,token_uri,img_uri):
            existing_owners = self.address_token_owners.get(address_token)
            if existing_owners == None:
                existing_owners = {}
            if "0x0000000000000000000000000000000000000000" != owners['from']:    
                existing_owners[owners['from']] = owners['from']
            if "0x0000000000000000000000000000000000000000" != owners['to']:    
                existing_owners[owners['to']] = owners['to']
            if token_uri is not None:
                existing_owners['token_uri'] = token_uri
            if img_uri is not None:    
                existing_owners['img_uri'] = img_uri
            self.address_token_owners[address_token] = existing_owners
    
        def is_vector_feature_exist(self,address_token):
            address_token = resources_folder+"vectors\\"+address_token
            if (path.exists(address_token+".jpeg.npz") or path.exists(address_token+".png.npz") 
                or path.exists(address_token+".jpg.npz") or path.exists(address_token+".gif.npz")
                or path.exists(address_token+".npz")):
                return True
            else:
                return False    


        def process_event(self, files_in_index, web3: Web3,  event: AttributeDict) -> str:
            """Record a ERC-20 transfer in our database."""
            # Events are keyed by their transaction hash and log index
            # One transaction may contain multiple events
            # and each one of those gets their own log index

            # event_name = event.event # "Transfer"
            log_index = event.logIndex  # Log index within the block
            # transaction_index = event.transactionIndex  # Transaction index within the block
            txhash = event.transactionHash.hex()  # Transaction hash
            block_number = event.blockNumber

            # Convert ERC-721 Transfer event to internal format
            args = event["args"]
            transfer = {
                "from": args["from"],
                "to": args.to,
                "tokenId": args.tokenId
            }
            address_token = event['address']+"_"+str(args.tokenId)
            address_token = address_token.lower()
            token_uri = None
            img_uri = None
            if (self.non_exist_tokens.get(address_token) == None
                and files_in_index.get("\\"+address_token) == None):
                    token_uri,img_uri = self.get_token_uri(web3,event['address'],args.tokenId)
            if token_uri is not None and img_uri is not None:
                self.add_owners_to_state(address_token,transfer,token_uri,img_uri)
            # print(transfer)
            # Create empty dict as the block that contains all transactions by txhash
            if block_number not in self.state["blocks"]:
                self.state["blocks"][block_number] = {}

            block = self.state["blocks"][block_number]
            if txhash not in block:
                # We have not yet recorded any transfers in this transaction
                # (One transaction may contain multiple events if executed by a smart contract).
                # Create a tx entry that contains all events by a log index
                self.state["blocks"][block_number][txhash] = {}

            # Record ERC-721 transfer in our database
            self.state["blocks"][block_number][txhash][log_index] = args.tokenId

            # Return a pointer that allows us to look up this event later if needed
            return address_token,img_uri, f"{block_number}-{txhash}-{log_index}"