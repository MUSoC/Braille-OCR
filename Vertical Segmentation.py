import cv2
import numpy as np
import math


img=cv2.imread('part6.jpg')
rows,cols,w = img.shape

#removing red pixels
for i in range(rows):
    for j in range(cols):
         pixel=img[i][j]
         if pixel[2]>pixel[1] and pixel[2]>pixel[0]:
             img[i][j]=[0,0,0]

cv2.imwrite('img6.jpg',img)


#dilating
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (15,rows+50))
horizontal = cv2.dilate(img, horizontalStructure, (-1, -1))
cv2.imwrite("dilatedpart.jpg", horizontal)


k=cv2.imread('dilatedpart.jpg')

#defining the edges
edges = cv2.Canny(k,50,150,apertureSize = 3)
cv2.imwrite('verticaledges.jpg',edges)



m=[]
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        m.append(((x1,y1),(x2,y2)))



#sorting list m as per x coordinate in the tuple
sorted_m=sorted(m, key=lambda x: x[0][0])

#printing sorted_m
for i in range(len(sorted_m)):
    print "sorted_m[", i , "]:", sorted_m[i]


#drawing lines
for i in range(len(sorted_m)):
    cv2.line(img,m[i][0],m[i][1],(0,0,255),2)
cv2.imwrite('hough.jpg',img)


#defining function distance
def distance(f,g):
    if f==sorted_m[-1][0]:
        return 0
    else:
    
        dx=g[0]-f[0]
        dy=g[1]-f[1]
        d =math.sqrt(dx*dx+dy*dy)
        return d


hough_lines=cv2.imread('hough.jpg')
p=[]

#saving each character in list p
for i in range(len(sorted_m)-1):
    p.append(hough_lines[m[i][1][1]:m[i][0][1],sorted_m[i][0][0]: sorted_m[i+1][0][0]])
    

for i in range(len(p)):
    if p[i].size in range(3500,5000):
        
        RD= distance(sorted_m[i+1][0] ,sorted_m[i+2][0])
        LD=distance( sorted_m[i-1][0],sorted_m[i][0])
            
        h1, w1 = p[i].shape[:2]
        black_image = np.zeros((p[i].shape[0], 42-p[i].shape[1],3))
        h2, w2 = black_image.shape[:2]

        #create empty matrix
        vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
        if RD>LD:
            #combine 2 images
            vis[:h1, :w1,:3] = p[i]
            vis[:h2, w1:w1+w2,:3] = black_image
            resized_image=cv2.resize(vis,(40,50))
            p[i]=resized_image
            cv2.imwrite('character['+str(i)+'].jpg',p[i])
        elif LD>RD:
            vis[:h2, :w2,:3] = black_image
            vis[:h1, w2:w1+w2,:3] = p[i]
            resized_image=cv2.resize(vis,(40,50))
            p[i]=resized_image
            cv2.imwrite('character['+str(i)+'].jpg',p[i])
    else:
        resized_image=cv2.resize(p[i],(40,50))
        cv2.imwrite('character['+str(i)+'].jpg',resized_image)



print "char[0] size:",p[0].size
print "char[14] size:",p[14].size
print "char[20] size:",p[20].size
print "char[26] size:",p[26].size
print "char[40] size:",p[40].size
print "char[24] size:",p[24].size
print "char[34] size:",p[34].size
print "char[36] size:",p[36].size

l=[p[0].size,p[14].size,p[20].size,p[26].size,p[40].size,p[24].size,p[34].size,p[36].size]
print "min:",min(l)
print "max:",max(l)
print p[20].shape



print "*********************"

print "char[10] size:",p[10].size
print "char[12] size:",p[12].size
print "char[2] size:",p[2].size
print "char[6] size:",p[6].size
print "char[8] size:",p[8].size
print "char[4] size:",p[4].size
print "char[16] size:",p[16].size
print "char[18] size:",p[18].size
print "char[22] size:",p[22].size
print "char[28] size:",p[28].size
print "char[30] size:",p[30].size
print "char[38] size:",p[38].size
print "char[32] size:",p[32].size
print "char[42] size:",p[42].size

b=[p[10].size,p[12].size,p[2].size,p[6].size,p[8].size,p[4].size,p[16].size,p[18].size,p[22].size,p[28].size,p[30].size,p[38].size,p[32].size,p[42].size]
print "min:",min(b)
print "max:",max(b)
print p[22].shape


        

print len(p)    

cv2.waitKey(0)























