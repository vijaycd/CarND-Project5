import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ystart = 400
ymax   = 656

croph = 64
cropw = 64

i = 0


filepath = 'image*.jpeg'
images = glob.glob(filepath)
for filename in images:
    image = mpimg.imread (filename)
    xmax = image.shape[1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for y in range (ystart, ymax, croph):
        for x in range (0, xmax, cropw):
            i += 1
            roi = image [y:y+croph, x:x+cropw]
            fn = 'roi'+str(i)+'.jpeg'
            cv2.imwrite(fn, roi)
            print (y, y+croph, x, x+cropw, 'generating', fn)
        
 