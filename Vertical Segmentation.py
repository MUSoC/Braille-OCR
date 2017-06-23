import cv2
import numpy as np
import math


img=cv2.imread('part6.jpg')

rows,cols,w = img.shape


for i in range(rows):
    for j in range(cols):
         pixel=img[i][j]
         if pixel[2]>pixel[1] and pixel[2]>pixel[0]:
             img[i][j]=[0,0,0]

cv2.imwrite('img6.jpg',img)


horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (15,rows+50))
horizontal = cv2.dilate(img, horizontalStructure, (-1, -1))

cv2.imwrite("dilatedpart.jpg", horizontal)

k=cv2.imread('dilatedpart.jpg')

#defining the edges
edges = cv2.Canny(k,50,150,apertureSize = 3)

cv2.imwrite('verticaledges.jpg',edges)

minLineLength = 100
maxLineGap = 10



lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        m.append(((x1,y1),(x2,y2)))


cv2.imwrite('hough.jpg',img)

w=cv2.imread('hough.jpg')
cv2.imshow('hell',pax)

print m

sorted_m=sorted(m, key=lambda x: x[0][0])
print sorted_m
p=[]
for i in range(len(m)-1):
    p.append(img[m[i][0][1]:m[i][1][1],m[i][0][0]:m[i+1][0][0]])
    #p.append(img[m[i][0][0]:m[i+1][0][0],m[i][0][1]:m[i][1][1]])



#cv2.namedWindow("char3" ,cv2.WINDOW_NORMAL)
#cv2.resizeWindow('char3',m[0][0][1]:m[0][1][1],m[0][0][0]:m[0+1][0][0])
cv2.imwrite('char3.jpg',p[3])

#cv2.imwrite('character_2.jpg',p[2])
cv2.waitKey(0)
