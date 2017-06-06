import cv2
import argparse
import numpy as np

img= cv2.imread('scan2.jpg')
img2gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


gaus= cv2.adaptiveThreshold(img2gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
kernel= np.ones((6,6), np.uint8)
dilation=cv2.dilate(gaus,kernel,iterations=1)
opening= cv2.morphologyEx(dilation,cv2.MORPH_OPEN,kernel)
clahe = cv2.createCLAHE(clipLimit=50.0, tileGridSize=(1,1000))
cl1 = clahe.apply(dilation)
#saving the processed image
cv2.imwrite('C:\Users\Mahe\Desktop\cl1.jpg',cl1)


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
                cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
                cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 1518,1898)
                cv2.imshow("image", image)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1518,1898)
cv2.setMouseCallback("image", click_and_crop)
 
# keep looping until the 'q' key is pressed
while True:
        # display the image and wait for a keypress
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1518,1898)
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
 
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
                image = clone.copy()
 
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
                break
 
# if there are two reference points, then crop the region of interest
# from the image and display it
if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.namedWindow('ROI',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ROI',refPt[0][1]-refPt[1][1], refPt[0][0]-refPt[1][0] )
        
        cv2.imshow("ROI", roi)   



cv2.waitKey(0)
cv2.destroyAllWindows()


