import tkinter as tk
from tkinter import Message, Text
import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
from pathlib import Path
 
window = tk.Tk()
window.title("Face_Recognizer")
window.configure(background ='white')
window.grid_rowconfigure(0, weight = 1)
window.grid_columnconfigure(0, weight = 1)
message = tk.Label(
    window, text ="Face-Recognition-System",
    bg ="green", fg = "white", width = 50,
    height = 3, font = ('times', 30, 'bold'))
     
message.place(x = 200, y = 20)
 
lbl = tk.Label(window, text = "No.",
width = 20, height = 2, fg ="green",
bg = "white", font = ('times', 15, ' bold ') )
lbl.place(x = 400, y = 200)
 
txt = tk.Entry(window,
width = 20, bg ="white",
fg ="green", font = ('times', 15, ' bold '))
txt.place(x = 700, y = 215)
 
lbl2 = tk.Label(window, text ="Name",
width = 20, fg ="green", bg ="white",
height = 2, font =('times', 15, ' bold '))
lbl2.place(x = 400, y = 300)
 
txt2 = tk.Entry(window, width = 20,
bg ="white", fg ="green",
font = ('times', 15, ' bold ')  )
txt2.place(x = 700, y = 315)
 
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
  
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
  
    return False

def TakeImages():       
     
   
    Id =(txt.get())
    name =(txt2.get())
     
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = r"C:\Users\Hp\AppData\Local\Programs\Python\Python39\haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             
            faces = detector.detectMultiScale(gray, 1.3, 5)
             
           
            for (x, y, w, h) in faces:
                
                cv2.rectangle(img, (x, y), (
                    x + w, y + h), (255, 0, 0), 2)
                sampleNum = sampleNum + 1
                
                cv2.imwrite(
                    r"C:\Users\Hp\AppData\Local\Programs\Python\Python39\TRAIN\TrainingImageTrainingImage\ "+name +"."+Id +'.'+ str(
                        sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                
                cv2.imshow('frame', img)
           
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
            elif sampleNum>60:
                break
       
        cam.release()
       
        cv2.destroyAllWindows()
        
        res = "Images Saved for ID : " + Id +" Name : "+ name 
       
        row = [Id, name]
        with open(r'C:\Users\Hp\AppData\Local\Programs\Python\Python39\TRAIN\UserDetails\UserDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            
            writer.writerow(row)
        csvFile.close()
        message.configure(text = res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text = res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text = res)
             

def TrainImages():
    
    recognizer = cv2.face.LBPHFaceRecognizer_create() 
    
    harcascadePath = r"C:\Users\Hp\AppData\Local\Programs\Python\Python39\haarcascade_frontalface_default.xml"
    
    detector = cv2.CascadeClassifier(harcascadePath)
    path = (r"C:\Users\Hp\AppData\Local\Programs\Python\Python39\TRAIN\TrainingImage")
    
    faces, Id = getImagesAndLabels(r"C:\Users\Hp\AppData\Local\Programs\Python\Python39\TRAIN\TrainingImage")
    
    recognizer.train(faces, np.array(Id))    
    recognizer.save(r"C:\Users\Hp\AppData\Local\Programs\Python\Python39\TRAIN\TrainingImageLabel\Trainner.yml")
  
    res = "Image Trained"
    message.configure(text = res)
 
def getImagesAndLabels(path):
    
    imagePaths =[os.path.join(path, f) for f in os.listdir(path)]
    faces =[]
    
    Ids =[]
    
    for imagePath in imagePaths:
      
        pilImage = Image.open(imagePath).convert('L')
        
        imageNp = np.array(pilImage, 'uint8')
        
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
       
        faces.append(imageNp)
        Ids.append(Id)       
    return faces, Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    recognizer.read(r"C:\Users\Hp\AppData\Local\Programs\Python\Python39\TRAIN\TrainingImageLabel\Trainner.yml")
    harcascadePath = r"C:\Users\Hp\AppData\Local\Programs\Python\Python39\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)

    df = pd.read_csv(r"C:\Users\Hp\AppData\Local\Programs\Python\Python39\TRAIN\UserDetails\UserDetails.csv") 
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX       
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)   
        for(x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])                                  
            if(conf < 50):
                aa = df.loc[df['Id'] == Id]['Name'].values
                tt = str(Id)+"-"+aa   
            else:
                Id ='Unknown'               
                tt = str(Id) 
            if(conf > 75):
                noOfFile = len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+
                str(noOfFile) + ".jpg", im[y:y + h, x:x + w])           
            cv2.putText(im, str(tt), (x, y + h),
            font, 1, (255, 255, 255), 2)       
        cv2.imshow('im', im)
        if (cv2.waitKey(1)== ord('q')):
            break
    cam.release()
    cv2.destroyAllWindows()
   
   
takeImg = tk.Button(window, text ="Sample",
command = TakeImages, fg ="white", bg ="green",
width = 20, height = 3, activebackground = "Red",
font =('times', 15, ' bold '))
takeImg.place(x = 200, y = 500)
trainImg = tk.Button(window, text ="Training",
command = TrainImages, fg ="white", bg ="green",
width = 20, height = 3, activebackground = "Red",
font =('times', 15, ' bold '))
trainImg.place(x = 500, y = 500)
trackImg = tk.Button(window, text ="Testing",
command = TrackImages, fg ="white", bg ="green",
width = 20, height = 3, activebackground = "Red",
font =('times', 15, ' bold '))
trackImg.place(x = 800, y = 500)
quitWindow = tk.Button(window, text ="Quit",
command = window.destroy, fg ="white", bg ="green",
width = 20, height = 3, activebackground = "Red",
font =('times', 15, ' bold '))
quitWindow.place(x = 1100, y = 500)
 
  
window.mainloop()