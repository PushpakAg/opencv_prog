import os 
mainDir = "C:\\Users\pushp\Documents\pythonAI\demoImages"
for root,dirs,files in os.walk(mainDir):
    print("root folder: ", root) 
    print("dirs in root: ",dirs)
    print("files in root: ",files)
    for file in files:
        name = str(file)[0:len(str(file))-4]
        print("fileName: ", name )
        fullFilePath = str(root) + "\x5c"+file
        print(fullFilePath)