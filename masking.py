import cv2
import numpy as np
x1= 0 
y1= 0
evt = 0
def call1(val):
    global HueL
    HueL = val
def call2(val):
    global HueH
    HueH= val
def call3(val):
    global SatL
    SatL = val
def call4(val):
    global SatH
    SatH = val
def call5(val):
    global ValL
    ValL = val
def call6(val):
    global ValH
    ValH = val
def call11(val):
    global hL
    hL = val
def call22(val):
    global hH
    hH = val 
def mClick(event,xPos,yPos,flags,params):
    global evt,x,y
    if event == cv2.EVENT_LBUTTONUP:
        evt = event
        x = xPos
        y = yPos
myCam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
myCam.set(cv2.CAP_PROP_FRAME_WIDTH,320)
myCam.set(cv2.CAP_PROP_FRAME_HEIGHT,160)
myCam.set(cv2.CAP_PROP_FPS,30)
myCam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
cv2.namedWindow("Setup")
cv2.resizeWindow("Setup",640,500)
cv2.moveWindow("Setup",800,800)
cv2.namedWindow("demo")
cv2.setMouseCallback("demo",mClick)
cv2.createTrackbar("HueL","Setup",10,179,call1)
cv2.createTrackbar("HueH","Setup",20,179,call2)
cv2.createTrackbar("hL","Setup",10,179,call11)
cv2.createTrackbar("hH","Setup",20,179,call22)
cv2.createTrackbar("SatL","Setup",10,255,call3)
cv2.createTrackbar("SatH","Setup",250,255,call4)
cv2.createTrackbar("ValL","Setup",10,255,call5)
cv2.createTrackbar("ValH","Setup",250,255,call6)

while True:
    ignore, frames = myCam.read()
    framesHSV = cv2.cvtColor(frames,cv2.COLOR_BGR2HSV)
    if evt == 4:
        print(framesHSV[x][y])
        evt = 0
    lBound = np.array([HueL,SatL,ValL]) 
    hBound = np.array([HueH,SatH,ValH])
    lB = np.array([hL,SatL,ValL])
    hB = np.array([hH,SatH,ValH])
    maskC1 = cv2.inRange(framesHSV,lBound,hBound)
    maskC1dis = cv2.resize(maskC1,[150,100])
    maskC2 = cv2.inRange(framesHSV,lB,hB)
    mainMask = maskC1 | maskC2 
    contours, ignored = cv2.findContours(mainMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    #cv2.drawContours(frames,contours,-1,(0,0,255),2)
    #obj_ofInt = cv2.bitwise_and(frames,frames, mask = mainMask)
    for i in contours:
        if cv2.contourArea(i) > 100:
            x,y,w,h = cv2.boundingRect(i)
            #print(cv2.boundingRect(i))
            cv2.rectangle(frames,(x,y),(x+w,y+h),(255,0,0),2)
            #x1 = int((x/640) * 1920)
            #y1 = int((y/480) * 1080)
    cv2.imshow("demo", frames)
    cv2.moveWindow("demo",0,0)
    cv2.imshow("maskC1", maskC1)
   # cv2.imshow("maskC2",maskC2)
    #cv2.imshow("main",obj_ofInt)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
myCam.release()