import cv2
import numpy as np
import glob
import os
from natsort import natsorted, ns

img_array = []
test = os.getcwd()

# directory = "/nobackup22/loconno2/mri_simulations/frames_wedtwo/"
directory = path = os.path.dirname(os.path.abspath(__file__)) + "/frames_friz/"
fileType = "*.png"
videoName = 'frames_friz.avi'
files = natsorted(glob.glob(directory + fileType))
files = files[::]
nFrames = len(files)

index = 0
for filename in files:
    img = cv2.imread(filename)
    try:
        height, width, layers = img.shape
    except:
        print('frame has no shape????')
        print(filename)
        continue
    size = (width,height)
    img_array.append(img)
    index = index + 1
    print("Appending frame " + str(index) + " of " + str(nFrames))
 
print("Initializing video..")
out = cv2.VideoWriter(directory + videoName, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
# fourcc for .avi => cv2.VideoWriter_fourcc(*'DIVX')
# fourcc for .mp4 => 0x7634706d

for i in range(len(img_array)):
    out.write(img_array[i])
    print("Writing frame " + str(i) + " of " + str(nFrames))

print("Done!")
out.release()
