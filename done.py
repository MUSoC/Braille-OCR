from __future__ import print_function
import cv2
import numpy as np
import math
import string
import sys



f=open("C:\Users\Mahe\Desktop\MUSOC.txt","w+")

for i in range(4):
    img=cv2.imread('part'+str(i)+'.jpg')
    rows,cols,w = img.shape

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray_image,200,255,cv2.THRESH_BINARY)


    #dilating
    kernel = np.ones((100,10),np.uint8)
    dilation = cv2.dilate(threshold,kernel,iterations = 1)

    cv2.imwrite('C:\Users\Mahe\Desktop\W3 characters\dilatedpart.jpg', dilation)
    k=cv2.imread('C:\Users\Mahe\Desktop\W3 characters\dilatedpart.jpg')

    #defining the edges
    edges = cv2.Canny(k,50,150,apertureSize = 3)
    cv2.imwrite('cannyEdge1.jpg',edges)

    m=[]
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
    for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                    m.append(((x1,y1),(x2,y2)))


    #sorting list m as per x coordinate in the tuple
    sorted_m=sorted(m, key=lambda x: x[0][0])


    #drawing lines
    for i in range(len(sorted_m)):
            cv2.line(img,sorted_m[i][0],sorted_m[i][1],(0,0,255),2)
    cv2.imwrite('C:\Users\Mahe\Desktop\W3 characters\hough1.jpg',img)


    #defining function distance
    def distance(f,g):
            if f==sorted_m[-1][0]:
                    return 0
            else:
                    dx=g[0]-f[0]
                    dy=g[1]-f[1]
                    d =math.sqrt(dx*dx+dy*dy)
                    return d

    def dis(f,g):
            dx=g[0]-f[0]
            dy=g[1]-f[1]
            d =math.sqrt(dx*dx+dy*dy)
            return d
            


    hough_lines=cv2.imread('C:\Users\Mahe\Desktop\W3 characters\hough1.jpg')
    p=[]
    sorted_m.append(((cols,rows),(cols,0)))

    #saving each character in list p
    for i in range(len(sorted_m)-1):
            p.append(hough_lines[sorted_m[i][1][1]:sorted_m[i][0][1],sorted_m[i][0][0]: sorted_m[i+1][0][0]])


    pxl=[]
    r=0
    def contains_space(img):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, threshold = cv2.threshold(gray_image,200,255,cv2.THRESH_BINARY)
            h,w,l=img.shape
            for g in range(h):
                for h in range(w):
                    if threshold[g][h]==255:
                            return 1
            return 0


            
    #removing the unnecesary space

    o=[]
    for i in range(len(p)):
            if contains_space(p[i])==1:
                    pxl.append(p[i])
                    o.append(p[i].shape[1])
            elif contains_space(p[i])==0:
                    if p[i].shape[1]>20:
                            pxl.append(p[i])
                            o.append(p[i].shape[1])
                            
            else:
                    pass
                            
                    

     
    #checking if p[i] ==img then return the width of adjecent space
    def xyz_r(img):
            for i in range(len(p)-1):
                    if img.shape == p[i].shape and not(np.bitwise_xor(img,p[i]).any()):
                            return p[i+1].shape[1]
                    
    def xyz_l(img):
            for i in range(len(p)):
                    if img.shape == p[i].shape and not(np.bitwise_xor(img,p[i]).any()):
                            return p[i-1].shape[1]


    zs={}
    for i in range(len(pxl)):
            zs[str(i)]=[]


    for i in range(len(pxl)):
            if pxl[i].shape[1] in range(min(o),(sum(o)/len(o))):
                    def contains_whitepx(img,x):
                            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            ret, threshold = cv2.threshold(gray_image,200,255,cv2.THRESH_BINARY)
                            h,w,l=img.shape
                            for g in range(h):
                                for h in range(w):
                                    if threshold[g][h]==255:
                                        zs[str(x)].append((h,g))     
                            return 0
                    contains_whitepx(pxl[i],i)
                    if len(zs[str(i)]) in range(0,20):
                            RD = xyz_r(pxl[i])
                            LD = xyz_l(pxl[i])
                                    
                            h1, w1 = pxl[i].shape[:2]
                            black_image = np.zeros((pxl[i].shape[0], 42-pxl[i].shape[1],3))
                            h2, w2 = black_image.shape[:2]
                            

                            #create empty matrix
                            vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
                            if RD>LD:
                                    #combine 2 images
                                    vis[:h1, :w1,:3] = pxl[i]
                                    vis[:h2, w1:w1+w2,:3] = black_image
                                    resized_image=cv2.resize(vis,(40,40))
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
                                    resized_image=cv2.resize(vis,(40,40))
                                    pxl[i]=resized_image
                                    rows,cols,w=pxl[i].shape
                                    for x in range(rows):
                                            for y in range(cols):
                                                     pixel=pxl[i][x][y]
                                                     if pixel[2]>pixel[1] and pixel[2]>pixel[0]:
                                                             pxl[i][x][y]=[0,0,0]
                                    
                    else:
                            resized_image=cv2.resize(pxl[i],(40,40))
                            pxl[i]=resized_image




    for i in range(len(pxl)):
            resized_image=cv2.resize(pxl[i],(40,50))
            pxl[i]=resized_image

    w=[]
    cvt=[]
    dic={}
    srt_dic={}

    #appending the coordinates of all the white pixels in the image in dic[str(i)]
    for i in range(len(pxl)):
            dic[str(i)]=[]
            srt_dic[str(i)]=[]
            

    def contains_white(img,x):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, threshold = cv2.threshold(gray_image,200,255,cv2.THRESH_BINARY)
            h,w,l=img.shape
            for i in range(h):
                for j in range(w):
                    if threshold[i][j]==255:
                        dic[str(x)].append((j,i))     
            return 0

    for x in range(len(pxl)):
            result= contains_white(pxl[x],x)
        
    for x in range(len(pxl)):
            dic[str(x)]=sorted(dic[str(x)], key=lambda tup: tup[1])
            
            



    #end_pts contain the coordinates of 1st and last white pixel in the image
    end_pts=[]
    for i in range (len(pxl)):
           if (len(dic[str(i)]) != 0) :
                   end_pts.append((dic[str(i)][0], (dic[str(i)][len(dic[str(i)])-1])))
           else:
                   end_pts.append(())


               
    for i in range(len(pxl)):
            
            cv2.line(pxl[i],((pxl[i].shape[1])/2,0),(((pxl[i].shape[1])/2),pxl[i].shape[0]),(0,0,255),2)
            cv2.line(pxl[i],(0,(pxl[i].shape[0])/3),(40,(pxl[i].shape[0])/3),(0,0,255),2)
            cv2.line(pxl[i],(0,(2*(pxl[i].shape[0])/3)),(40,(2*(pxl[i].shape[0])/3)),(0,0,255),2)
            cv2.imwrite('C:\Users\Mahe\Desktop\W3 characters\character['+str(i)+'].jpg',pxl[i])

            #dividing each image into 6 equal rois
            w.append((pxl[i][0:(pxl[i].shape[0])/3,0:(pxl[i].shape[1])/2],     pxl[i][(pxl[i].shape[0])/3:2*((pxl[i].shape[0])/3),0:(pxl[i].shape[1])/2],        pxl[i][2*((pxl[i].shape[0])/3):pxl[i].shape[0],0:(pxl[i].shape[1])/2] ,     pxl[i][0:(pxl[i].shape[0])/3,(pxl[i].shape[1])/2:(pxl[i].shape[1])],     pxl[i][(pxl[i].shape[0])/3:2*((pxl[i].shape[0])/3),(pxl[i].shape[1])/2:(pxl[i].shape[1])] ,      pxl[i][2*(pxl[i].shape[0])/3:pxl[i].shape[0],(pxl[i].shape[1])/2:(pxl[i].shape[1])]))
            cvt.append([0,0,0,0,0,0])


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


    d={'numbers': '001111' ,'capital': '000001' , 'decimal': '000101' }
    alpha={'a': '100000', 'c': '100100', 'b': '110000', 'e': '100010', 'd': '100110', 'g': '110110', 'f': '110100', 'i': '010100', 'h': '110010', 'k': '101000', 'j': '010110', 'm': '101100', 'l': '111000', 'o': '101010', 'n': '101110', 'q': '111110', 'p': '111100', 's': '011100', 'r': '111010', 'u': '101001', 't': '011110', 'w': '010111', 'v': '111001', 'y': '101111', 'x': '101101', 'z': '101011'}
    n={'0':'010110', '1':'100000', '2':'110000', '3':'100100' ,'4':'100110'  ,'5':'100010' ,'6':'110100' ,'7':'110110'   ,'8':'110010' ,'9': '010100'}
    c={',':'010000', '.':'010011','!':'011010','?':'011001',';':'011000',' ':'000000'}

    
    for ch,valu in alpha.iteritems():
            if cvt[0]==valu:
                    f.write (ch)
                    continue

    for ch,valu in c.iteritems():
            if cvt[0]==valu:
                    f.write (ch)
                    continue 


    for i in range(1,len(cvt)):
            try:
                    if cvt[i-1]==d['numbers']:
                                    f.write (n.keys()[n.values().index(cvt[i])])
                                    continue
                                    

                    elif cvt[i-1]==d['capital']:
                                    z=string.lowercase.index(alpha.keys()[alpha.values().index(cvt[i])])
                                    f.write (string.uppercase[z])
                                    continue

                    else:
                                    for ch,valu in alpha.iteritems():
                                            if cvt[i]==valu:
                                                    f.write (ch)
                                                    continue

                                    for ch,valu in c.iteritems():
                                            if cvt[i]==valu:
                                                    f.write (ch)
                                                    continue

            except ValueError:
                    pass

    
    f.write (' ')

f.close()







