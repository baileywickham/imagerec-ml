from contextlib import closing
from videosequence import VideoSequence
import os
import glob
import shutil
import csv
from skimage import transform

mp4File = input("Enter name of mp4 file:")

src_dir = input("Enter full path of the folder split.py is held in:")

dst_dir = input("Enter full path of the intended folder for jpg's:")

andyornot = input("Enter 1 if all images are andy and 0 if not:")

with closing(VideoSequence(mp4File)) as frames:
    for idx, frame in enumerate(frames[1:]):
        frame.save("frame{:}.jpg".format(idx))



with open('labels.csv','w') as myfile:
   wrtr = csv.writer(myfile, delimiter=',', quotechar='"')
   for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
        wrtr.writerow(andyornot)
        myfile.flush()


for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
    im = skimage.transform.resize(im, (64, 64))
    shutil.copy(jpgfile, dst_dir)
    os.remove(jpgfile)

