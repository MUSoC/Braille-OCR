import cv2
import argparse
import numpy as np

img= cv2.imread('pic.jpg',0)


gaus= cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,16)
kernel= np.ones((1,2), np.uint8)
#dilation=cv2.dilate(gaus,kernel,iterations=2)
opening= cv2.morphologyEx(gaus,cv2.MORPH_OPEN,kernel)
#clahe = cv2.createCLAHE(clipLimit=50.0, tileGridSize=(1,1000))
#cl1 = clahe.apply(dilation)
ret,mask = cv2.threshold(opening,50,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('mask',opening)
se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
mask = np.dstack([mask, mask, mask]) 
rows,cols,l=mask.shape
cv2.imwrite('C:\Users\Mahe\Desktop\mask.jpg',mask)


# cropping 
refPt = []
cropping = False
 
def click_and_crop(event, x, y, flags, param):
        global refPt, cropping
 
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
                refPt = [(x, y)]
                cropping = True
 
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates and indicate that
                # the cropping operation is finished
                refPt.append((x, y))
                cropping = False
 
                # draw a rectangle around the region of interest
                cv2.rectangle(mask, refPt[0], refPt[1], (0, 0, 255), 2)
                cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image',  rows,cols)
                cv2.imshow("image", mask)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = mask.copy()
cv2.namedWindow("image" ,cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',  rows,cols)
cv2.setMouseCallback("image", click_and_crop)
 
# keep looping until the 'q' key is pressed
while True:
        # display the image and wait for a keypress
        cv2.namedWindow("image" ,cv2.WINDOW_NORMAL) 
        cv2.resizeWindow('image',  rows,cols)
        cv2.imshow("image", mask)
        key = cv2.waitKey(1) & 0xFF
 
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
                image = clone.copy()
 
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
                break
 
# if there are two reference points, then crop the region of interest from the image and display it
if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imwrite('roi.jpg',roi)

        cv2.namedWindow("ROI" ,cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ROI',refPt[0][1]-refPt[1][1], refPt[0][0]-refPt[1][0] )
        cv2.imshow("ROI", roi)
       
cv2.waitKey(0)
cv2.destroyAllWindows()




