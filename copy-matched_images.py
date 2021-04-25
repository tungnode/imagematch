from shutil import copyfile
import glob
# json for storing data in json file
import json
import os.path
from os import path


with open('.\\nearest_neighbors.json') as json_file:
    similar_tokens = json.load(json_file)

for similar_pair in similar_tokens:

    src_folders = []
    src_folders.append("..\\copyHunter\\scripts\\images_50k\\")
    src_folders.append("..\\copyHunter\\scripts\\images_25k\\")
    file_types = []
    file_types.append(".jpeg")
    file_types.append(".png")
    file_types.append(".jpg")
    file_types.append(".gif")
    file_types.append("")

    for folder_index, folder in enumerate(src_folders):
        for file_type_index, file_type in enumerate(file_types):
            master_src_file = folder+similar_pair['master_pi']+file_type
            master_des_file = ".\\matchedImages\\"+similar_pair['master_pi']+file_type
            if(path.exists(master_src_file)):
                copyfile(master_src_file,master_des_file)

            neighbor_src_file = folder+similar_pair['similar_pi']+file_type
            neighbor_des_file = ".\\matchedImages\\"+similar_pair['similar_pi']+file_type
            if(path.exists(neighbor_src_file)):
                copyfile(neighbor_src_file,neighbor_des_file)    
    