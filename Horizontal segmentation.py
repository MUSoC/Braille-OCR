import cv2
import numpy as np
import math

th2=cv2.imread('roi.jpg')
r,c,w=th2.shape
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (c+2000,13))
horizontal = cv2.dilate(th2, horizontalStructure, (-1, -1))
cv2.imwrite("horizontal2.jpg", horizontal)


img = cv2.imread('horizontal2.jpg')

#defining the edges
edges = cv2.Canny(img,50,150,apertureSize = 3)
cv2.imwrite('edges.jpg',edges)


#finding the end points of the hough lines
#lines = cv2.HoughLines(edges,1,np.pi/180,200)
m=[]

minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
                m.append(((x1,y1),(x2,y2)))
#print m
print "length of list m is:",len(m)

sorted_m=sorted(m, key=lambda x: x[0][1])

    
print "length of list sorted_m is:", len(sorted_m)

sorted_m.insert(0,((0,0),(c,0)))
#drawing line
for i in range (0,len(sorted_m)):
    cv2.line(th2,sorted_m[i][0],sorted_m[i][1],(0,0,255),3)
    
        
cv2.imwrite('hough_lines.png',th2)


s=cv2.imread('hough_lines.png')
p=[]
for i in range (0,len(sorted_m)):
    if i!=len(sorted_m)-1:
        p.append(th2[sorted_m[i][0][1]:sorted_m[i+1][0][1],sorted_m[i][0][0]:sorted_m[i][1][0]])
    else:
        p.append(th2[sorted_m[len(lines)-2][0][1]:r, sorted_m[len(lines)][0][0]:sorted_m[len(lines)][1][0]])

pix=[]

print(len(p))

for x in range(len(p)):
    def contains_white(img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY)
        h,w,l=img.shape
        for i in range(h):
            for j in range(w):
                if threshold[i][j]==255:
                    return True
            



    result= contains_white(p[x])
    if result== True:
        pix.append(p[x])

    
print(len(pix))
for i in range(len(pix)):
    cv2.imwrite('part' +str(i)+'.jpg',pix[i])

cv2.waitKey(0)
cv2.destroyAllWindows()
