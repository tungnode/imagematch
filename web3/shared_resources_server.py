import json
from constants import resources_folder 
from multiprocessing.managers import BaseManager
from queue import Queue
queue = Queue()
indexed_files_dicts = {}
read_file = open(resources_folder+"index_to_file_name.json", "r")
index_to_file_name = json.load(read_file)
for name in index_to_file_name.values():
    name = name[name.rfind('\\'):]
    name = name[:name.find('.')]
    name = name.lower()
    indexed_files_dicts[name] = name
BaseManager.register('get_features_queue', callable=lambda:queue)
BaseManager.register('get_indexed_files_dict', callable=lambda:indexed_files_dicts)
m = BaseManager(address=('', 50000), authkey=b'abracadabra')
s = m.get_server()
s.serve_forever()