import cv2
import face_recognition as fr
font = cv2.FONT_HERSHEY_SIMPLEX
donFace = fr.load_image_file("C:/Users/pushp/Documents/pythonAI/demoImages/known/Donald Trump.jpg")
facePos = fr.face_locations(donFace)[0]
donEncode = fr.face_encodings(donFace)[0]

nancyFace = fr.load_image_file("C:/Users/pushp/Documents/pythonAI/demoImages/known/Nancy Pelosi.jpg")
facePos = fr.face_locations(nancyFace)[0]
nancyEncode = fr.face_encodings(nancyFace)[0]
print(nancyEncode)
jijiFace = fr.load_image_file("C:/Users/pushp/Documents/pythonAI/demoImages/known/Jiji.jpg")
facePos = fr.face_locations(jijiFace)[0]
jijiEncode = fr.face_encodings(jijiFace)[0]

knownEncode = [donEncode,nancyEncode,jijiEncode]
names = ["donaldT","nancyP","jiji"]

unknownFace = fr.load_image_file("C:/Users/pushp/Documents /pythonAI/demoImages/unknown/u00.jpg")
unknownFaceBGR = cv2.cvtColor(unknownFace,cv2.COLOR_RGB2BGR)
faceLocs = fr.face_locations(unknownFace)
unEncode = fr.face_encodings(unknownFace,faceLocs)

for i,j in zip(faceLocs,unEncode):
    t,l,b,r = i
    cv2.rectangle(unknownFaceBGR,(l,t),(r,b),(0,0,255),2)
    name = "unknown"
    matches = fr.compare_faces(knownEncode,j)
    print(matches)
    if True in matches:
        matchIn = matches.index(True)
        name = names[matchIn]
    cv2.putText(unknownFaceBGR,name,(r,t),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.55,(105,60,0),2)
cv2.imshow("faceRecognition",unknownFaceBGR)
cv2.moveWindow("faceRecognition",0,0)
cv2.waitKey(10000)  
