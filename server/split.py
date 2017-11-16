from contextlib import closing
from videosequence import VideoSequence
import os
import glob
import shutil
import csv
from skimage import transform
import sys

try:
    mp4File = str(sys.argv)
except OSError:
    print("make sure to pass a mp4 in as a paramter")
     
andyornot = input("Enter 1 if all images are andy and 0 if not:")

with closing(VideoSequence(mp4File)) as frames:
    for idx, frame in enumerate(frames[1:]):
        frame.save("images/frame{:}.jpg".format(idx))



with open('labels.csv','w') as myfile:
   wrtr = csv.writer(myfile, delimiter=',', quotechar='"')
   for jpgfile in glob.iglob(os.path.join('images/', "*.jpg")):
        wrtr.writerow(andyornot)
        myfile.flush()

