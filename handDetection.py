import mediapipe as mp
import cv2
myCam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
myCam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
myCam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
myCam.set(cv2.CAP_PROP_FPS,30)
myCam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
hands = mp.solutions.hands.Hands(False,2,.7,.7)
mpDraw = mp.solutions.drawing_utils
def handDraw(results,output):
    # added mpDraw for some future modifications. 
    handsArr = []
    if results.multi_hand_landmarks != None:
        for handLandMarks in results.multi_hand_landmarks:
            handsPos = [] 
            for landmark in handLandMarks.landmark:
                x = int(landmark.x*640)
                y = int(landmark.y*480)
                handsPos.append((x,y))
                if output == "single":
                    return handsPos
            
            if output == "multi" :
                handsArr.append(handsPos)   
                return handsArr         
while True:
    # handsArr = [] 
    ignore, frames = myCam.read()
    framesRGB = cv2.cvtColor(frames,cv2.COLOR_BGR2RGB)
    results = hands.process(framesRGB)
    handsPos = handDraw(results,output="single")
    if handsPos != None:
        cv2.circle(frames,handsPos[20],10,(255,0,255),-1)
    # if results.multi_hand_landmarks != None:
    #     for handLandMarks in results.multi_hand_landmarks:
    #         count+=1
    #         handsPos = []
    #         #mpDraw.draw_landmarks(frames,handLandMarks,mp.solutions.hands.HAND_CONNECTIONS)
    #         # handLandMarks.landmark has 21 values.
    #         for landmark in handLandMarks.landmark:
    #             x = int(landmark.x*640)
    #             y = int(landmark.y*480)
    #             handsPos.append((x,y))
    #             print("1")
    #         print("2")
    #         cv2.circle(frames,handsPos[2],15,(255,255,0),-1)
    #         handsArr.append(handsPos)
    #         print(handsArr)
    #         print("count in: ", count)
    #         print("")
    # print("count out: ",count)
    cv2.imshow("demo", frames)
    cv2.moveWindow("demo",0,0)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
myCam.release()
