import os
import pickle
import cv2
import time
import face_recognition as fr
mainDir = "C:\\Users\pushp\Documents\pythonAI\demoImages\known"
encodings = []
names = []
bar_width = 30
for root,dirs,files in os.walk(mainDir):
    counts = len(files)
    count = 0 
    print("Training Started")
    for file in files:
        name = str(file)[0:len(str(file)) - 4]
        filePath = str(root) + "\x5c" + file
        face = fr.load_image_file(filePath)
        encoding = fr.face_encodings(face)[0]
        encodings.append(encoding)
        names.append(name)
        count+=1
        progress = count * bar_width // counts
        rem = bar_width - progress
        filled = "\u001b[31;1m"+ "\u2588" * progress
        empty = "\u2591" * rem
        reset_code = "\u001b[0m"
        bar = filled+reset_code+empty
        percentage = count/counts *100
        if count == counts:
            print(f"[{bar}] {percentage:.1f}%")
        else:
            print(f"[{bar}] {percentage:.1f}%", end = "\r")
print("Training compeleted, Model is ready!!")
with open("trainedData.pkl","wb") as model:
    pickle.dump(names,model)
    pickle.dump(encodings,model)