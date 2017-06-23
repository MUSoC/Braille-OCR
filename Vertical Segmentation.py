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


#cv2.imshow('edges', edges)
cv2.imwrite('verticaledges.jpg',edges)

minLineLength = 100
maxLineGap = 10

#lines = cv2.HoughLines(edges,1,np.pi/180,200)
m=[]


lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        m.append(((x1,y1),(x2,y2)))

sorted_m=sorted(m, key=lambda x: x[0][0])

for i in range(len(sorted_m)):
    print "sorted_m[", i , "]:", sorted_m[i]

for i in range(len(sorted_m)):
    cv2.line(img,m[i][0],m[i][1],(0,0,255),2)
        
cv2.imwrite('hough.jpg',img)



pax=cv2.imread('hough.jpg')

p=[]
for i in range(len(sorted_m)-1):
    p.append(pax[m[i][1][1]:m[i][0][1],sorted_m[i][0][0]: sorted_m[i+1][0][0]])
    
print p
for i in range (len(p)):
    r,c,w=p[i].shape
    for x in range(r):
        for y in range(c):
             pixel=p[i][x][y]
             if pixel==[255,255,255]:
                 del p[i]
    
for i in range(len(p)):
    
    cv2.imwrite("char["+str(i)+"].jpg",p[i])
    cv2.imread("char["+str(i)+"].jpg",cv2.IMREAD_GRAYSCALE)
    r,c,w=p[i].shape
    #for x in range(r):
     #   for y in range(c):
      #       pixel=p[i][x][y]
       #      if pixel[2]>pixel[1] and pixel[2]>pixel[0]:
        #         p[i][x][y]=[0,0,0]

    resized_image=cv2.resize(p[i],(80,50))
    cv2.imwrite("char["+str(i)+"].jpg",resized_image)




print len(p)    
#cv2.imwrite('character_2.jpg',p[2])
cv2.waitKey(0)
























