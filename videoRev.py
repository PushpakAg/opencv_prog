import cv2
import time
vid  = cv2.VideoCapture("D:/bts-part-6-robomaster-build-ft-dreadnoughtrobotics-shorts-robotics-team-1280-ytshorts.savetube.me.mp4")
vid.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
index = frames - 1
t1 = 0.0
while True:
    t2 = time.time()
    dt = t2-t1
    fps = int(1/dt)
    t1 = time.time()
    vid.set(cv2.CAP_PROP_POS_FRAMES,index)
    ignore, video = vid.read()
    cv2.putText(video, "fps: " + str(fps), (20,20), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    cv2.imshow("video", video)
    index-=1
    if cv2.waitKey(10) & 0xff == ord('q'):
        break
vid.release()
