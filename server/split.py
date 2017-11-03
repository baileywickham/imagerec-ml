from contextlib import closing
from videosequence import VideoSequence
import os
import glob
import shutil

mp4File = input("Enter name of mp4 file:")

src_dir = input("Enter full path of the folder split.py is held in:")

dst_dir = input("Enter full path of the intended folder for jpg's:")


with closing(VideoSequence(mp4File)) as frames:
    for idx, frame in enumerate(frames[1:]):
        frame.save("frame{:}.jpg".format(idx))





for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
    shutil.copy(jpgfile, dst_dir)
    os.remove(jpgfile)

