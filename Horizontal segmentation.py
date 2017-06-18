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
       

k={}
for i in range (0,59):
    h=m['line_'+str(i)][1][1]
    if m['line_'+str(i+1)][1][1] in range(h,(h+213)):
        del m['line_'+str(i)]
        
    
#for i in range(0,len(m.keys())):

 #   k['line_'+str(i)]=(m.keys())[i]

#print k

for i in range (0,len( m.keys())):
    try:
        cv2.line(th2,m['line_'+str(i)][0],m['line_'+str(i)][1],(0,0,255),2)
    except KeyError:
        print("Key not available")
         



#for i in range(0,60):
    
 #   print ("line_"+str(i),':',m['line_'+str(i)])

#j=((m['line_'+str(1)][0][1]))
#print j
#print m['line_'+str(0)][0][1]
#g= img[m['line_'+str(0)][0][1]:j,m['line_'+str(0)][0][0]:m['line_'+str(0)][1][0]]



#for i in range(0,59):
    #k['var_'+str(i)]= img[m['line_'+str(i)][0][1]:m['line_'+str(i+1)][0][1],m['line_'+str(i)][0][0]:m['line_'+str(i)][1][0]]
    #k['var_'+str(i)]= img[m['line_'+str(i)][0][1]:m['line_'+str(i+1)][1][1],m['line_'+str(i)][0][0]:m['line_'+str(i+1)][1][0]]
   
#cv2.imshow('part 12', g)

#cv2.imshow('part 1', k['var_'+str(0)])
#cv2.imshow('new', img)
cv2.imwrite('hough_lines.jpg',th2)





cv2.waitKey(0)
cv2.destroyAllWindows()

