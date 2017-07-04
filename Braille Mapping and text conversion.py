from __future__ import print_function
import cv2
import numpy as np
import math
import string


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

'''
#printing sorted_m
for i in range(len(sorted_m)):
    print "sorted_m[", i , "]:", sorted_m[i]
'''

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
    

print (p[1].shape)
print (p[3].shape)
print (p[7].shape)
print (p[9].shape)
print (p[11].shape)
print (p[13].shape)
print (p[15].shape)
print (p[17].shape)
print (p[24].shape)
print (p[26].shape)
print (p[28].shape)
print (p[30].shape)
print (p[32].shape)
print (p[36].shape)

print ("****************")
print (p[5].shape)
print (p[19].shape)
print (p[22].shape)
print (p[34].shape)


pxl=[]
for i in range(len(p)):
    if p[i].shape[1] in range(7,100):
        pxl.append(p[i])
        
for i in range(len(pxl)):
    cv2.imwrite('character['+str(i)+'].jpg',pxl[i])
        
for i in range(len(pxl)):
    if pxl[i].size in range(1300,5000):
        
        RD= distance(sorted_m[i+1][0] ,sorted_m[i+2][0])
        LD=distance( sorted_m[i-1][0],sorted_m[i][0])
            
        h1, w1 = pxl[i].shape[:2]
        black_image = np.zeros((pxl[i].shape[0], 42-pxl[i].shape[1],3))
        h2, w2 = black_image.shape[:2]
        

        #create empty matrix
        vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
        if RD>LD:
            #combine 2 images
            vis[:h1, :w1,:3] = pxl[i]
            vis[:h2, w1:w1+w2,:3] = black_image
            resized_image=cv2.resize(vis,(40,50))
            pxl[i]=resized_image
            rows,cols,w=pxl[i].shape
            for x in range(rows):
                for y in range(cols):
                     pixel=pxl[i][x][y]
                     if pixel[2]>pixel[1] and pixel[2]>pixel[0]:
                         pxl[i][x][y]=[0,0,0]
            
        elif LD>RD:
            vis[:h2, :w2,:3] = black_image
            vis[:h1, w2:w1+w2,:3] = pxl[i]
            resized_image=cv2.resize(vis,(40,50))
            pxl[i]=resized_image
            rows,cols,w=pxl[i].shape
            for x in range(rows):
                for y in range(cols):
                     pixel=pxl[i][x][y]
                     if pixel[2]>pixel[1] and pixel[2]>pixel[0]:
                         pxl[i][x][y]=[0,0,0]
            
    else:
        resized_image=cv2.resize(pxl[i],(40,50))
        pxl[i]=resized_image






w=[]
cvt=[]

for i in range(len(pxl)):
    cv2.line(pxl[i],((pxl[i].shape[1])/2,0),(((pxl[i].shape[1])/2),pxl[i].shape[0]),(0,0,255),2)
    cv2.line(pxl[i],(0,(pxl[i].shape[0])/3),(40,(pxl[i].shape[0])/3),(0,0,255),2)
    cv2.line(pxl[i],(0,(2*(pxl[i].shape[0])/3)),(40,(2*(pxl[i].shape[0])/3)),(0,0,255),2)
    cv2.imwrite('character['+str(i)+'].jpg',pxl[i])

    #dividing each image into 6 equal rois
    w.append((pxl[i][0:(pxl[i].shape[0])/3,0:(pxl[i].shape[1])/2],     pxl[i][(pxl[i].shape[0])/3:2*((pxl[i].shape[0])/3),0:(pxl[i].shape[1])/2],        pxl[i][2*((pxl[i].shape[0])/3):pxl[i].shape[0],0:(pxl[i].shape[1])/2] ,     pxl[i][0:(pxl[i].shape[0])/3,(pxl[i].shape[1])/2:(pxl[i].shape[1])],     pxl[i][(pxl[i].shape[0])/3:2*((pxl[i].shape[0])/3),(pxl[i].shape[1])/2:(pxl[i].shape[1])] ,      pxl[i][2*(pxl[i].shape[0])/3:pxl[i].shape[0],(pxl[i].shape[1])/2:(pxl[i].shape[1])]))
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



alphabets=str(100000110000100100100110100010110100110110110010010100010110101000111000101100101110101010111100111110111010011100011110101001111001010111101101101111101011)
d={}
alpha={}
for i in range(26):
    alpha[string.lowercase[i]]=(alphabets[i*6:(i+1)*6])

print (alpha)
print ("**************")
n={'0':'010110', '1':'100000', '2':'110000', '3':'100100' ,'4':'100110'  ,'5':'100010' ,'6':'110100' ,'7':'110110'   ,'8':'110010' ,'9': '010100'}
d['number']='001111'
d['capital']='000001'
d['decimal']='000101'
c={',':'010000', '.':'010011','!':'011010','?':'011001',';':'011000',' ':'000000'}


print (cvt)
print ("**************")
print (n)
print ("**************")
print (d)
print ("**************")
print (c)





for i in range(1,len(cvt)):
    if cvt[i-1]==d['number']:
            print (n.keys()[n.values().index(cvt[i])],end = '')
            continue
            

    elif cvt[i-1]==d['capital']:
            z=string.lowercase.index(alpha.keys()[alpha.values().index(cvt[i])])
            print (string.uppercase[z], end='')
            continue

    else:
            for ch,valu in alpha.iteritems():
                if cvt[i]==valu:
                    print (ch, end='')
                    continue

            for ch,valu in c.iteritems():
                if cvt[i]==valu:
                    print (ch, end='')
                    continue





cv2.waitKey(0)
cv2.destroyAllWindows()
























