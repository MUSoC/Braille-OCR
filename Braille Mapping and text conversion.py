import cv2
import numpy as np
import math



img=cv2.imread('part7.jpg')
rows,cols,w = img.shape


#removing red pixels
for i in range(rows):
    for j in range(cols):
         pixel=img[i][j]
         if pixel[2]>pixel[1] and pixel[2]>pixel[0]:
             img[i][j]=[0,0,0]

cv2.imwrite('img7.jpg',img)


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
    p.append(hough_lines[sorted_m[i][1][1]:sorted_m[i][0][1],sorted_m[i][0][0]: sorted_m[i+1][0][0]])
    

for i in range(len(p)):
    if p[i].size in range(0,5000):
        
        RD= distance(sorted_m[i+1][0] ,sorted_m[i+2][0])
        LD=distance( sorted_m[i-1][0],sorted_m[i][0])
            
        h1, w1 = p[i].shape[:2]
        print h1,w1
        black_image = np.zeros((p[i].shape[0], 42-p[i].shape[1],3))
        h2, w2 = black_image.shape[:2]
        print h2,w2

        #create empty matrix
        vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
        if RD>LD:
            #combine 2 images
            vis[:h1, :w1,:3] = p[i]
            vis[:h2, w1:w1+w2,:3] = black_image
            resized_image=cv2.resize(vis,(40,50))
            p[i]=resized_image
            rows,cols,w=p[i].shape
            for x in range(rows):
                for y in range(cols):
                     pixel=p[i][x][y]
                     if pixel[2]>pixel[1] and pixel[2]>pixel[0]:
                         p[i][x][y]=[0,0,0]
            
        elif LD>RD:
            vis[:h2, :w2,:3] = black_image
            vis[:h1, w2:w1+w2,:3] = p[i]
            resized_image=cv2.resize(vis,(40,50))
            p[i]=resized_image
            rows,cols,w=p[i].shape
            for x in range(rows):
                for y in range(cols):
                     pixel=p[i][x][y]
                     if pixel[2]>pixel[1] and pixel[2]>pixel[0]:
                         p[i][x][y]=[0,0,0]
            
    else:
        resized_image=cv2.resize(p[i],(40,50))
        p[i]=resized_image

w=[]
cvt=[]
for i in range(len(p)):
    cv2.line(p[i],((p[i].shape[1])/2,0),(((p[i].shape[1])/2),p[i].shape[0]),(0,0,255),2)
    cv2.line(p[i],(0,(p[i].shape[0])/3),(40,(p[i].shape[0])/3),(0,0,255),2)
    cv2.line(p[i],(0,(2*(p[i].shape[0])/3)),(40,(2*(p[i].shape[0])/3)),(0,0,255),2)
    cv2.imwrite('character['+str(i)+'].jpg',p[i])

    #dividing each image into 6 equal rois
    w.append((p[i][0:(p[i].shape[0])/3,0:(p[i].shape[1])/2],     p[i][(p[i].shape[0])/3:2*((p[i].shape[0])/3),0:(p[i].shape[1])/2],        p[i][2*((p[i].shape[0])/3):p[i].shape[0],0:(p[i].shape[1])/2] ,     p[i][0:(p[i].shape[0])/3,(p[i].shape[1])/2:(p[i].shape[1])],     p[i][(p[i].shape[0])/3:2*((p[i].shape[0])/3),(p[i].shape[1])/2:(p[i].shape[1])] ,      p[i][2*(p[i].shape[0])/3:p[i].shape[0],(p[i].shape[1])/2:(p[i].shape[1])]))
    cvt.append([0,0,0,0,0,0])
q=0

for i in range(len(w)):
    for j in range(6):
        for x in range(w[i][j].shape[0]):
            for y in range(w[i][j].shape[1]):
                pixel=w[i][j][x][y]
                if pixel[0]>100 and pixel[1]>100 and pixel[2]>100:
                    q=1
                    cvt[i][j]=1
                    continue
for i in range(len(cvt)):
    
    a="".join(map(str,cvt[i]))
    cvt[i]=a





cv2.waitKey(0)
cv2.destroyAllWindows()
























