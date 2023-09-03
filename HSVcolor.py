import cv2 
import numpy as np
evt = 0
def mClick(events,x,y,flags,params):
    global evt,xPos,yPos
    if events == cv2.EVENT_LBUTTONUP:
        evt = 4
        xPos = x
        yPos = y
matS = np.zeros([256,720,3],dtype=np.uint8)
matV = np.zeros([256,720,3], dtype = np.uint8)
clr = np.zeros([200,200,3], dtype = np.uint8)
for rows in range(0,256):
    for cols in range(0,720):
        matS[rows,cols] = (int(cols/4),rows,255)
for rows in range(0,256):
    for cols in range(0,720):
        matV[rows,cols] = (int(cols/4),255,rows)
matS = cv2.cvtColor(matS,cv2.COLOR_HSV2BGR)
matV = cv2.cvtColor(matV,cv2.COLOR_HSV2BGR)
cv2.namedWindow("S_color")
cv2.setMouseCallback("S_color",mClick)
while True:
    if evt == 4:
        clr[:,:] = matS[yPos][xPos]
        cv2.putText(clr,str(clr[0][0]),(0,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)
        cv2.imshow("color",clr)
        evt = 0 
    cv2.imshow("S_color",matS)
    cv2.imshow("V_Color",matV)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break