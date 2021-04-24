from os import listdir
import os
from PIL import Image
import PIL
# images_dir = "./corrupted/"
images_dir = "../copyHunter/scripts/images/"
need_remove = []   
for filename in listdir(images_dir):
    print(filename)
    try:
      img = Image.open(images_dir+filename) # open the image file
      img.verify() # verify that it is, in fact an image
    except (IOError, SyntaxError) as e:
      print('Bad file:', filename) # print out the names of corrupt files
      need_remove.append(images_dir+filename)
    except (Exception, OSError) as boomErr:
        continue

for filename in need_remove:
  try:
    os.remove(filename)
  except (Exception) as e:
    continue  