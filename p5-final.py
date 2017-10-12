#S
# C. D. Vijay, Udacity - CarND-Term 1, Project 5: Vehicle Detection and Tracking
#Test video: https://youtu.be/wPGa2zAPuLw  Prj video: https://youtu.be/J_uK_nuDYic
# IMPORTS
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import argparse

from lesson_functions import * #Remove after bringing all functions into this file
from sklearn.model_selection import train_test_split #my sklearn is v0.18.1

#Class to track heat across frames
class TrackHeat:
    def __init__(self):
        self.hmap_list = []
    def append(self, hmap):
        self.hmap_list.append(hmap)


def PreprocessImage (image, normalize):
    
    #Normalize image data 
    if normalize: 
        #print ('PNG dataset detected; but cv2 reads in a different format,so normalize')
        image = image.astype(np.float32)/255
   
    x, y, z = image.shape
    red_total_avg = (np.sum(image [:,0,0])) / (x*y)
    green_total_avg = (np.sum(image [0,:,0]))  / (x*y)
    blue_total_avg = (np.sum(image [0,0,:])) / (x*y)
    plt.imshow(image)
    
    #print ("Total RGB in the img: ", red_total_avg, green_total_avg, blue_total_avg)

    #Multiply image by avgs
    redmul = image*red_total_avg
    greenmul = image*green_total_avg
    bluemul = image*blue_total_avg
    #Now multiply all 
    composite = redmul*greenmul*bluemul
    return composite


def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    
    
    
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    imgchannel1 = cv2.resize(img[:,:,0], size).ravel()
    imgchannel2 = cv2.resize(img[:,:,1], size).ravel()
    imgchannel3 = cv2.resize(img[:,:,1], size).ravel()
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return np.hstack((imgchannel1, imgchannel2, imgchannel3))

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg! REMOVED
def color_hist(img, nbins=32):
    
    # Compute the histogram of the color channels separately
    imgchannel1_hist = np.histogram(img[:,:,0], bins=nbins)
    imgchannel2_hist = np.histogram(img[:,:,1], bins=nbins)
    imgchannel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((imgchannel1_hist[0], imgchannel2_hist[0], imgchannel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a LIST of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
        
    normalize = 0  #When extracting features, whether png or jpeg, we will not normalize but when detecting, we will if needed 
    
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        #NEW STEP 
        image = PreprocessImage(image, normalize)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        else: 
            if debug: print ("Omitting spatial features in extract_features()")
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        else: 
            if debug: print ("Omitting histogram features in extract_features()")
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        else: 
            if debug: print ("Omitting HOG features in extract_features()")
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)): #was 64,64
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1])) 
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    #print ('in slide window', window_list)
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(255, 0, 0), thick=4):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function to extract features from a SINGLE image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, vis=False):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            if vis == True:
                hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)
    
    #9) Return concatenated array of features
    if vis == True:
           return np.concatenate(img_features), hog_image
    else:
           return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    
    #Make an empty heatmap
    heatmap = np.zeros_like(img[:,:,0])
    
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #print ('in srch windows', window)
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
#Visualize
def visualize (fig, rows, cols, imgs, titles):
    for i, img in enumerate (imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            #plt.title(titles[i])
        else:
            plt.imshow(img)
            #plt.title(titles[i])
    plt.axis('off')
    plt.show()
    
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

       
#Subsampling        
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                spatial_feat=True, hist_feat=True, hog_feat=True):
      
    global pngflag
    boxes = []
    count = 0
    draw_img = np.copy(img)
    normalize = 0 #will be turned ON only when a PNG dataset is detected (ie pngflag =1)
    
    
    #Make an empty heatmap
    heatmap = np.zeros_like(img[:,:,0])
    img_tosearch = img [ystart:ystop, :,:]  #Image is being cropped here
    #print ("Findcars: img_tosearch shape", img_tosearch.shape)
    ctrans_tosearch = img_tosearch #convert_color(img_tosearch, conv='RGB2YCrCb')
    #print ("Findcars: ctrans_tosearch shape, scale=", ctrans_tosearch.shape, scale)
   
    
    if scale != 1.0:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    #Here, do the AveRGB
    #NEW STEP 
    if pngflag:
        normalize = 1
    ctrans_tosearch = PreprocessImage(ctrans_tosearch, normalize)
    
    
    #print ('Find cars(): shape of ctrans_tosearch, scale, ystart, ystop:', ctrans_tosearch.shape, scale, ystart, ystop)
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
    
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step, same as overlap=0.75, 8 cells/blk, moving window 2 cells over = having 6 (out of 8) prev cells in the next blk
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        
    for xb in range(nxsteps):
        t1 = time.time()
        for yb in range(nysteps):
            count +=1 # Start counting windows
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch, 64x64x3
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
            # Get color features
            if spatial_feat==True:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                
            if hist_feat==True:
                hist_features = color_hist(subimg, nbins=hist_bins)
                
                
            # Scale features and make a prediction
            if (spatial_feat == True and hist_feat == True and hog_feat == False):
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features)).reshape(1, -1))    
            elif (spatial_feat == True and hist_feat == True and hog_feat == True):
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            elif (spatial_feat == False and hist_feat == True and hog_feat == True):
                test_features = X_scaler.transform(np.hstack((hist_features, hog_features)).reshape(1, -1))    
            elif (spatial_feat == False and hist_feat == False and hog_feat == True):
                test_features = X_scaler.transform(hog_features)
            elif (spatial_feat == True and hist_feat == False and hog_feat == True):
                test_features = X_scaler.transform(np.hstack((spatial_features, hog_features)).reshape(1, -1))
            elif (spatial_feat == False and hist_feat == True and hog_feat == False):
                test_features = X_scaler.transform(np.hstack((hist_features)).reshape(1, -1))
                
            test_prediction = svc.predict(test_features)
            conf = svc.decision_function (test_features) #Distance from the separating SVM hyperplane, not a percentage
            #if conf > 0.0: print ('Confidence =', conf)
            
            if test_prediction == 1 and abs(conf) > 0.2:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] +=1  #Same as add_heat() in lesson
               

    #print (heatmap[:,:]==1)
    #draw_img =   draw_boxes1(draw_img, boxes) 
    #print ('exiting find cars()')
    return boxes, heatmap

from scipy.ndimage.measurements import label

def apply_threshold (heatmap, threshold):
    #Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    return heatmap
    
def draw_labeled_boxes (img, labels):
    #Iterate through the list of detected cars
    for car_num in range(1, labels[1] + 1):
        #Find pixels with each car_num label value
        nonzero = (labels [0] == car_num).nonzero()
        #Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        #Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        #Draw box on img
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img

ystart = 400  #used to be 400x660
ystop  = 660

#Set up parameters for the functions to be called
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9 # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block

spatial_feat = False # Spatial features (resampling an img to become smaller or larger using OpenCV resize():on or off
spatial_size = (32, 32) # Spatial binning dimensions

hist_feat = True # Histograms of colors features: on or off
hist_bins = 32   # Number of histogram bins

hog_feat = True  # HOG features on or off
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    
hmap_list = []
scales = [1.0, 1.5, 2.0, 2.5] #Used only for experimentation, but settled finally on scale=1
#scales = [1.0, 3.5]#, 2.0, 2.5]
#Instantiate object to track heat across frames
track_heat = TrackHeat()
numframes = 20 #number of video frames to average over
heat_threshold = 1
scale = 1    
        

def ProcessImage (img):
    #Will return a single image with labeled boxes drawn on it

    img1 = np.copy(img)
      
    for s in scales:    
        boxes, hmap = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                                   spatial_feat, hist_feat, hog_feat)
    
    #draw_img = draw_boxes(img1, boxes)
    #plt.imshow(draw_img)
    #plt.show()
   
    track_heat.append(hmap)
    if len (track_heat.hmap_list) > numframes:
        track_heat.hmap_list = track_heat.hmap_list[1:numframes+1] # drop the first one, retain the others
    array_trackHeat = np.array(track_heat.hmap_list)
    
    heat_avg = np.mean(array_trackHeat, axis=0)
    map = apply_threshold(heat_avg, heat_threshold)
    #plt.imshow(map)
    #plt.show()
    
    hmap = np.clip (map, 0, 255)
    labels = label (hmap)
    #plt.imshow(hmap)
    #plt.show()
 
    draw_img = draw_labeled_boxes(img1, labels)
    #plt.imshow(draw_img)
    #plt.show()
  
    return draw_img    
    
    
############################EXECUTION STARTS HERE##########################################
from moviepy.editor import VideoFileClip

parser = argparse.ArgumentParser(description='Udacity-CarND-T1-P5: Vehicle Detection ')
parser.add_argument('-d', action="store", dest="debug" )
parser.add_argument ('-v', action="store", dest="video")
args=parser.parse_args()

if (args.debug == '1'):
      debug = 1
else :
      debug = 0

if debug: 
    print ('\nDebug Mode ON...\n')
print ('\nVehicle Detection, Car ND-P5, C D Vijay')
print ('\nPICK ONE: -OR- Enter to Exit')
print ('______________________________________________________________')
print ('1  Test Video    (test dataset, 1200/1200 car/noncar imgs/jpg)')
print ('2  Project Video (test dataset, 1200/1200 car/noncar imgs/jpg)')
print ('3  Test Video    (full dataset, 6000/9000 car/noncar imgs/png)')
print ('4  Project Video (full dataset, 6000/9000 car/noncar imgs/png)')
print ('5  Test Images')

choice = input('\nYour Choice (Enter to exit): ')

pngflag = 0

# Read imgs or a video filename 
if   (choice == '0'): 	#Problem child
    name = 'prjvidclip.mp4'
    filepath = "imgs\*.jpeg"
    video = 1
elif   (choice == '1'): 	
    name = 'test_video.mp4'
    filepath = "imgs\*.jpeg"
    video = 1
elif (choice == '2'):  
    name = 'project_video.mp4'
    filepath = "imgs\*.jpeg"
    video = 1
elif (choice == '3'): 	
    name = 'test_video.mp4'
    filepath = "imgs1\*.png"
    video = 1
    pngflag = 1
elif (choice == '4'):
    name = 'project_video.mp4'
    filepath = "imgs1\*.png"
    video = 1  
    pngflag = 1    
elif (choice == '5'):
    #Work on test images -- color and gradient and sobel x/y thresholding
    t_images = glob.glob('test_images/test*.jpg')
    filepath = "imgs\*.jpeg"
    video = 0
else: 
    exit()
      
#Read all the car and noncar images for SVM training
images = glob.glob(filepath)

cars = []
notcars = []
for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        
        cars.append(image)
print ("\nNumber of car images:     ", len(cars))
print ("Number of non-car images: ", len(notcars))

#Set up parameters for the functions to be called
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9 # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block

spatial_feat = False # Spatial features (resampling an img to become smaller or larger using OpenCV resize():on or off
spatial_size = (32, 32) # Spatial binning dimensions

hist_feat = True # Histograms of colors features: on or off
hist_bins = 32   # Number of histogram bins

hog_feat = True  # HOG features on or off
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

y_start_stop = [400, 656] # Min and max in y to search in slide_window()


#Test routines by running two images, one car, one noncar
#random1 = np.random.randint(0, len(cars))
#random2 = np.random.randint(0, len(notcars))
#Read in the two imgs
#car_image = mpimg.imread(cars[random1])
#notcar_image = mpimg.imread(notcars[random2])
#car_features, car_hog_image = single_img_features(car_image, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, 
#                                                 cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat,
#                                                 hist_feat=hist_feat, hog_feat=hog_feat, vis=True)

#notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space=color_space, spatial_size=spatial_size, 
#                                                 hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, 
#                                                 cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat,
#                                                 hist_feat=hist_feat, hog_feat=hog_feat, vis=True)
            
car_features    = extract_features(cars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

#TESTING SINGLE IMAGES                              
#images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
#titles = ['Car img', 'Car HOG img', 'Noncar img', 'Noncar HOG img']
#fig = plt.figure (figsize=(12,3))
#visualize(fig, 1, 4, images, titles)

#TRAINING DATASETS 
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
#svc = SVC(kernel='linear', C=0.1) #NOT using Linear SVC as there is something with intercept scaling that makes it unfit
svc = LinearSVC(C=0.1) #But this trains much faster than SVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

if choice == '5':

    #rint ('len of test img array', len(test_images))
    images = []
    titles = []

    for image in t_images:
        t1=time.time()
        img = mpimg.imread(image)  #Scaling not reqd as the training was on jpg, and these are jpg
        
        draw_img = np.copy(img)
        
        #NEW STEP 
        img = PreprocessImage(img, 0)
        
        # xy_window=(96, 96), xy_overlap=(0.9, 0.1))
        # xy_window=(96, 96), xy_overlap=(0.5, 0.5)) not bad, fewer FPs
        windows = slide_window (img, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5))
        #Temp 3 lines
        #imcopy = draw_boxes1(draw_img, windows)
        #plt.imshow(imcopy)
        #plt.show()
        
        hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       

        windowed_img = draw_boxes(draw_img, hot_windows, color=(255, 0, 0), thick=6)                    
        images.append(windowed_img)
        titles.append('Windowed Box')
        print (round(time.time() - t1, 2), 'sec to process one img', len(hot_windows), 'hot windows found')
        #plt.imshow(windowed_img)
        #plt.show()
        
    fig = plt.figure(figsize=(8,8), dpi=300) # w x h
    visualize (fig, 5, 2, images, titles) # rows x cols
    exit()

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255

out_images = []
out_titles = []
ystart = 400  #used to be 400x660
ystop  = 660
scale  = 1

if video:
    cap = cv2.VideoCapture(name)
    framenum = 1

imgindex = 0
num_det = 0 #Number of car detections incl FPs

font = cv2.FONT_HERSHEY_SIMPLEX      

hmap_list = []
#scales = [1.0, 1.5, 2.0, 2.5]
scales = [1.0, 3.5]#, 2.0, 2.5]
#Instantiate object to track heat across frames
track_heat = TrackHeat()
numframes = 20 #number of video frames to average over
heat_threshold = 2
font = cv2.FONT_HERSHEY_SIMPLEX
'''
while True:

    if video:
        ret, img  = cap.read()
        if ret == False: 
            break
        fn = str(framenum)
        img1 = np.copy(img)
        print ("Processing frame ", fn)

    
    for s in scales:    
        boxes, hmap = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                                   spatial_feat, hist_feat, hog_feat)
    #out_images.append(main_img)
    #plt.imshow(main_img)
    #plt.show()
    track_heat.append(hmap)
    if len (track_heat.hmap_list) > numframes:
        track_heat.hmap_list = track_heat.hmap_list[1:numframes+1] # drop the first one, retain the others
        
    array_trackHeat = np.array(track_heat.hmap_list)
    
    heat_avg = np.mean(array_trackHeat, axis=0)
    map = apply_threshold(heat_avg, heat_threshold)
    hmap = np.clip (map, 0, 255)
    labels = label (hmap)
    draw_img = draw_labeled_boxes(img1, labels)
    
    print (labels[1], ' car(s) found', 'in frame ', framenum)
    num_det += labels[1]
    #plt.imshow(labels[0], cmap='gray')
    #plt.show()
    personal = "Udacity CarND Program, 2017, Project 5"
    str2 = 'Interstate 280N'
    personal = "C. D. Vijay, Udacity CarND Program, 2017"
    cv2.putText(draw_img, 'Frame: '+fn, (10,15), font, 0.5, (255,255,255), 0, cv2.LINE_AA)
    cv2.putText(draw_img, str2, (10,30), font, 0.5, (0,255,255), 0, cv2.LINE_AA)
    cv2.putText(draw_img, personal, (930, 15), font, 0.5, (255,0,0), 0, cv2.LINE_AA)
    cv2.putText(draw_img, 'Filename: '+name, (1050, 30), font, 0.5, (255,0,255), 0, cv2.LINE_AA)
    cv2.imshow('Vehicle Detection & Tracking', draw_img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
       break
    #else: print ('Need to open a comment line for Project video, line 695')
    framenum += 1        

'''
print ("Processing frames for vehicle detection in: ", name)
t1 = time.time()
clip1 = VideoFileClip(name)
processed_clip = clip1.fl_image(ProcessImage)

import sys
fname = sys.argv[0]
savedfile = fname.replace(".py", ".mp4")

#Write video to file
processed_clip.write_videofile(savedfile, audio=False)
print ('Time reqd. to process and write to video file: ', round(time.time()-t1, 2)/60, 'min.')
'''
#Show the array of images thru OpenCV
i = 1
for windowed_img in images:
    name = name + ' Frame ' + str(i)
    personal = "C. D. Vijay, Udacity CarND Program, 2017"
    cv2.putText(windowed_img, personal, (930, 15), font, 0.5, (255,0,0), 0, cv2.LINE_AA)
    cv2.putText(windowed_img, name, (580, 710), font, 0.5, (255,0,255), 0, cv2.LINE_AA)
    cv2.imshow('Vehicle Detection', windowed_img)
    i += 1
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

'''