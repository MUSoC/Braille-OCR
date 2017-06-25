import cv2
import numpy as np
import math

th2=cv2.imread('roi.jpg')
r,c,w=th2.shape
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (c,20))
horizontal = cv2.dilate(th2, horizontalStructure, (-1, -1))
cv2.imwrite("horizontal2.jpg", horizontal)



img = cv2.imread('horizontal2.jpg')

#defining the edges
edges = cv2.Canny(img,50,150,apertureSize = 3)


#finding the end points of the hough lines
lines = cv2.HoughLines(edges,1,np.pi/180,200)
m=[]
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
        
    m.append(((x1,y1),(x2,y2)))        
       
#print m
print "length of list m is:",len(m)

sorted_m=sorted(m, key=lambda x: x[0][1])


#removing extra lines in between the rows
for i in range(0,59):
    try:
        dx=sorted_m[i+1][0][0]-sorted_m[i][0][0]
        dy=sorted_m[i+1][0][1]-sorted_m[i][0][1]
        distance =math.sqrt(dx*dx+dy*dy)
        if distance <100:
            del sorted_m[i]
            
    except IndexError:
        pass          
print "length of list sorted_m is:", len(sorted_m)


#drawing line
for i in range (0,len(sorted_m)):
    cv2.line(th2,sorted_m[i][0],sorted_m[i][1],(0,0,255),3)
    
         
cv2.imwrite('hough_lines.jpg',th2)


cv2.imread('hough_lines.jpg')
p=[]
for i in range (0,29):
    p.append( th2[sorted_m[i][0][1]:sorted_m[i+1][0][1],sorted_m[i][0][0]:sorted_m[i][1][0]])

#for testing for 1 row
cv2.imshow('part7',p[7])
cv2.imwrite('character7',p[7])
cv2.waitKey(0)
cv2.destroyAllWindows()

