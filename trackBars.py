import cv2
xPos,yPos = 0,0
evt = 0
def mClick(event,x,y,flags,params):
    global evt 
    if event == cv2.EVENT_LBUTTONUP:
        evt = event
    if event == cv2.EVENT_RBUTTONUP:
        evt = event 
def winMoveX(val):
    global xPos
    xPos = val
    cv2.moveWindow("demo",xPos,yPos)
def winMoveY(val):
    global yPos
    yPos = val
    cv2.moveWindow("demo",xPos,yPos)
myCam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
myCam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
myCam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
myCam.set(cv2.CAP_PROP_FPS,30)
myCam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
cv2.namedWindow("demo")
cv2.setMouseCallback("demo",mClick)
count = 1
while True:
    ignore, frames = myCam.read()
    if evt == 4:
        if count == 1:
            cv2.namedWindow("Setup")
            cv2.moveWindow("Setup",670,0)
            cv2.resizeWindow("Setup",400,130)
            cv2.createTrackbar("WinX","Setup",0,1920,winMoveX)
            cv2.createTrackbar("WinY","Setup",0,1080,winMoveY)
            evt = 0
            count = 0
    if evt == 5:
        cv2.destroyWindow("Setup")
        evt = 0
        count = 1
    cv2.putText(frames,"L button click for Setup menu",(0,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
   # cv2.putText(frames,"R button click to close Setup menu",(0,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow("demo", frames)
    #cv2.resizeWindow("demo",1920,1080)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
myCam.release()