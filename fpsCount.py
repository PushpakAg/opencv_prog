import cv2
import time
t1 = 0
myCam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
myCam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
myCam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
myCam.set(cv2.CAP_PROP_FPS,30)
myCam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
t2 = 1000
while True:
    ignore, frames = myCam.read()
    frames = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)
    cv2.imshow("demo", frames)
    cv2.moveWindow("demo",0,0)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
myCam.release()
