import cv2
import numpy as np
th2=cv2.imread('roi.jpg')
horizontal = th2
#vertical = th2
rows,cols,W = th2.shape
horizontalsize = cols
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,20))
#horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))

#cv2.imshow("horizontal", horizontal)
cv2.imwrite("horizontal2.jpg", horizontal)



img = cv2.imread('horizontal2.jpg')
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,200)
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
        print (x1,y1),(x2,y2)

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('new', img)
cv2.imwrite('hough_lines',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
