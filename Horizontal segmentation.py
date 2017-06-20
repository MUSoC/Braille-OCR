import cv2
import numpy as np
import math
th2=cv2.imread('roi.jpg')
horizontal = th2

rows,cols,W = th2.shape
print th2.shape
horizontalsize = cols
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,20))
horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
cv2.imwrite("horizontal2.jpg", horizontal)



img = cv2.imread('horizontal2.jpg')

#defining the edges
edges = cv2.Canny(img,50,150,apertureSize = 3)


#finding the end points of the hough lines
lines = cv2.HoughLines(edges,1,np.pi/180,200)
m={}
for i in range(0,60):
    for rho,theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 2000*(-b))
        y1 = int(y0 + 2000*(a))
        x2 = int(x0 - 2000*(-b))
        y2 = int(y0 - 2000*(a))
        
    m['line_'+str(i)]=(x1,y1),(x2,y2)        
       

