from shutil import copyfile
import glob
# json for storing data in json file
import json
import os.path
from os import path


with open('.\\nearest_neighbors.json') as json_file:
    similar_tokens = json.load(json_file)

for similar_pair in similar_tokens:
    if(path.exists("..\\copyHunter\\scripts\\images\\"+similar_pair['master_pi']+".jpeg")):
        copyfile("..\\copyHunter\\scripts\\images\\"+similar_pair['master_pi']+".jpeg", ".\\matchedImages\\"+similar_pair['master_pi']+".jpeg")
    elif(path.exists("..\\copyHunter\\scripts\\images\\"+similar_pair['master_pi']+".jpg")):
        copyfile("..\\copyHunter\\scripts\\images\\"+similar_pair['master_pi']+".jpg", ".\\matchedImages\\"+similar_pair['master_pi']+".jpg")
    else:
        copyfile("..\\copyHunter\\scripts\\images\\"+similar_pair['master_pi']+".png", ".\\matchedImages\\"+similar_pair['master_pi']+".png")

    if(path.exists("..\\copyHunter\\scripts\\images\\"+similar_pair['similar_pi']+".jpeg")):
        copyfile("..\\copyHunter\\scripts\\images\\"+similar_pair['similar_pi']+".jpeg", ".\\matchedImages\\"+similar_pair['similar_pi']+".jpeg")
    elif(path.exists("..\\copyHunter\\scripts\\images\\"+similar_pair['similar_pi']+".jpg")):
        copyfile("..\\copyHunter\\scripts\\images\\"+similar_pair['similar_pi']+".jpg", ".\\matchedImages\\"+similar_pair['similar_pi']+".jpg")
    else:
        copyfile("..\\copyHunter\\scripts\\images\\"+similar_pair['similar_pi']+".png", ".\\matchedImages\\"+similar_pair['similar_pi']+".png")


    