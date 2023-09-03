import cv2
myCam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
myCam.set(cv2.CAP_PROP_FRAME_WIDTH,320)
myCam.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
#print(myCam.get(cv2.CAP_PROP_FRAME_HEIGHT))
myCam.set(cv2.CAP_PROP_FPS,30)
myCam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
faceCascade = cv2.CascadeClassifier("haarFiles\data\haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarFiles\data\haarcascade_eye.xml")
bodyCascade = cv2.CascadeClassifier("haarFiles\data\haarcascade_fullbody.xml")
while True:
    ignore, frames = myCam.read()
    framesG = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(framesG,1.1,5)
    bodies = bodyCascade.detectMultiScale(framesG,1.1,5)
    for face in faces:
        x,y,w,h = face
        eyesROI = framesG[y:y+h,x:x+w]
        eyes = eyeCascade.detectMultiScale(eyesROI)
        for eye in eyes:
            x1,y1,w1,h1 = eye
            xc = int(x1+w1/2)
            yc = int(y1+h1/2)
            #cv2.rectangle(frames[y:y+h,x:x+w],(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
            cv2.circle(frames[y:y+h,x:x+w],(xc,yc),15,(255,0,0),2)
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2)
    for body in bodies:
        xb,yb,wb,hb = body
        cv2.rectangle(frames,(xb,yb),(xb+wb,yb+hb),(0,0,255),2)
    cv2.imshow("demo", frames)
    cv2.moveWindow("demo",0,0)
    if cv2.waitKey(1) & 0xff == ord('q'):
         break
myCam.release()
