import cv2
import face_recognition as fr
import pickle
#rtsp_url = "rtsp://admin:L2ZXQZCO@192.168.29.197:554/cam/realmonitor?channel=1&subtype=0"
myCam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
myCam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
myCam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
myCam.set(cv2.CAP_PROP_FPS,30)
myCam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

with open("trainedData.pkl","rb") as data:
    names = pickle.load(data)
    knownEncode = pickle.load(data)
print(names)
while True:
    ignore,frames = myCam.read()
    framesRGB = cv2.cvtColor(frames,cv2.COLOR_RGB2BGR)
    faceLocs = fr.face_locations(frames)
    unknownEncode = fr.face_encodings(framesRGB)
    for i,j in zip(faceLocs,unknownEncode):
        t,l,b,r = i
        name = "Unknown"
        cv2.rectangle(frames,(l,t),(r,b),(255,0,0),2)
        matches = fr.compare_faces(knownEncode,j)
        print(matches) 
        if True in matches:
            matchIndex = matches.index(True)
            name = names[matchIndex]
            print(name)
        cv2.putText(frames,name,(r,t),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.75,(105,60,0),2)
    cv2.imshow("LiveVideo",frames)
    cv2.moveWindow("LiveVideo",0,0)
    if cv2.waitKey(1) & 0xff == ord('q'):
         break
myCam.release()
