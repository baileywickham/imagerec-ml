from contextlib import closing
from videosequence import VideoSequence
import os
import glob
import csv
import sys

# This file is for splitting video files.
#Get any command line args, if possible
commandLineArgs = list()
try:
	commandLineArgs = sys.argv
except OSError:
	pass #Ignore

#If command line args are available, the first one is the mp4, and the second one is whether they're andy or not
#If no command line args are present, or not enough, then we can just ask the user
mp4File = str(commandLineArgs[1]) if len(commandLineArgs) > 1 else input("Enter mp4 file: ")
andyornot = int(commandLineArgs[2]) if len(commandLineArgs) > 2 else int(input("Enter if they're andy or not: "))

#Split frames
with closing(VideoSequence(mp4File)) as frames:
	for idx, frame in enumerate(frames[1:]):
		frame.save("images/frame{:}.jpg".format(idx))

#Save labels
with open('labels.csv', 'w') as myfile:
	wrtr = csv.writer(myfile, delimiter=',', quotechar='"')
	for jpgfile in glob.iglob(os.path.join('images/', "*.jpg")):
		wrtr.writerow(andyornot)
		myfile.flush()
