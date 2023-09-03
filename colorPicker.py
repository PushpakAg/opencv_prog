import cv2
import numpy as np
evt = 0
def mClick(event,x,y,flags,params):
    global evt,xPos,yPos
    if event == cv2.EVENT_LBUTTONUP:
        evt = event
        xPos = x
        yPos = y
    if evt == cv2.EVENT_RBUTTONUP:
        evt = event
        xPos = x
        yPos = y
myCam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
myCam.set(cv2.CAP_PROP_FRAME_WIDTH,320)
myCam.set(cv2.CAP_PROP_FRAME_HEIGHT,150)
myCam.set(cv2.CAP_PROP_FPS,30)
myCam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
cv2.namedWindow("demo")
cv2.setMouseCallback("demo",mClick)
while True:
    ignore, frames = myCam.read()
    if evt == 4:
        cube = np.zeros([200,200,3],dtype = np.uint8)
        clr = frames[yPos][xPos]
        cube[:,:] = clr
        color = cv2.cvtColor(cube,cv2.COLOR_BGR2HSV)
        cv2.putText(color,"BGR:"+str(clr),(0,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)
        cv2.putText(color,"HSV:"+str(color[0][0]),(0,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)
        cv2.imshow("color",color)
        cv2.moveWindow("color",670,0)
        cv2.resizeWindow("color",200,200)
        evt = 0
    cv2.imshow("demo", frames)
    cv2.moveWindow("demo",0,0)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
myCam.release()
