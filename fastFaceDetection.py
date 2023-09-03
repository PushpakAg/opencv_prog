import cv2
import time
import mediapipe as mp 
myCam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
myCam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
myCam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
myCam.set(cv2.CAP_PROP_FPS,30)    
myCam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
findFace = mp.solutions.face_detection.FaceDetection()
drawFace = mp.solutions.drawing_utils
t1 = 0.0
while True:
    t2 = time.time()
    dt = t2-t1
    fps = int(1/dt)
    rFps = int( 0.9*30 + 0.1*fps ) 
    t1 = time.time()
    ignore, frames = myCam.read()
    cv2.putText(frames,"fps: " + str(rFps),(20,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    framesRGB = cv2.cvtColor(frames,cv2.COLOR_BGR2RGB)
    results = findFace.process(framesRGB)
    # print(results.detections.location_data)
    if results.detections != None:
        for face in results.detections:
            
            drawFace.draw_detection(frames,face)
    cv2.imshow("demo", frames)
    cv2.moveWindow("demo",0,0)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
myCam.release()