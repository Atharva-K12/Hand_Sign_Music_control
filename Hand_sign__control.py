import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from selenium import webdriver 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.common.keys import Keys
import time
def prediction(img):#For use of trained model to  give prediction
    test_img = torch.from_numpy(np.expand_dims(np.expand_dims(cv2.resize(img, (28, 28)), 0), 0))
    model.eval()
    y=model.forward(test_img)
    result=F.softmax(y,dim=1)
    max_val=0
    index=0
    max_val,index=torch.max(result,axis=1)
    if max_val>0.75:# keeping a threshold to avoid functioning at lower probablities
        sign_dict = {0: "A", 1: "B", 2: "D", 3: "G", 4: "O", 5: "P",6:"V",7:"W",8:"Y"}
        return sign_dict[index.item()]
def close():
    print("Turning off camera.")
    webcam.release()
    print("Camera off.")
    print("Program ended.")
    cv2.destroyAllWindows()
class HandSignModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img=nn.Sequential(
            nn.Conv2d(1,64,3,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,128,3,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(5*5*128,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,9),
            nn.BatchNorm1d(9),
        )

    def forward(self,X):
        return self.img(X.float())
model=HandSignModel()
model.load_state_dict(torch.load('trained.pth'))#Loading pretrained state_dictionary
model.eval()
key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
flag=0#flag is 1 if browser was opened else 0
while True:
    try:
        ret,frame = webcam.read()
        frame=cv2.flip(frame,1)
        key = cv2.waitKey(1)
        if key == ord('q'):
            if flag==1:
                driver.close()
            close()
            break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.rectangle(frame,(x:=100,y:=100),(x_:=x+200,y_:=y+200),(255,0,0),1)
        hand=gray[y:y_,x:x_]#limiting the image size as a 100*100*1
        cv2.imshow("Capturing",hand)
        ch=prediction(hand)
        if ch != None:
            print(ch)
        if ch=='O':#O is for opening/closing of browser
            if flag==0:
                flag=1
                driver_location='./geckodriver.exe'
                driver = webdriver.Firefox(executable_path=driver_location)
                driver.get("https://www.jiosaavn.com/album/guitar/hk5Nf1gQ76I_")#selected a random playlist can be made userinput
                play=driver.find_element_by_xpath('//button[@class="play"]')
                driver.minimize_window()
                play.click()#plays the selected playlist
                WebDriverWait(driver,600)#unconditonal wait for raise of exceptions
                play_pause=1# 1 is for play 0 is for pause
                time.sleep(1)
            else:
                driver.close()
                close()
        elif ch=='D' and flag==1:#D is for volume to level 1
            vol_1=driver.find_element_by_xpath('//span[@class="slider ui-draggable"]')
            driver.execute_script("arguments[0].style.top='66px';return arguments[0];",vol_1)
            time.sleep(2)
        elif ch=='V' and flag==1:#V is for play and pause
            if play_pause==0:
                el = WebDriverWait(driver,600).until(lambda d: d.find_element_by_xpath('//button[@class="controls"][@id="play"]'))
                pause=driver.find_element_by_xpath('//button[@class="controls"][@id="play"]')
                pause.click()
                play_pause=1
            else:
                el = WebDriverWait(driver,600).until(lambda d: d.find_element_by_xpath('//button[@class="controls"][@id="pause"]'))
                play=driver.find_element_by_xpath('//button[@class="controls"][@id="pause"]')
                play.click()
                play_pause=0
            time.sleep(2)
        elif ch=='G' and flag==1:#G is for next song
            next_song=driver.find_element_by_xpath('//button[@id="fwd"]')
            next_song.click()
            time.sleep(1)
        elif ch=='P' and flag==1:#P is for prevoius song
            previous_song=driver.find_element_by_xpath('//button[@id="rew"]')
            previous_song.click()
            time.sleep(2)
        elif ch=='B' and flag==1:#B is for volume to level0 
            vol_0=driver.find_element_by_xpath('//span[@class="slider ui-draggable"]')
            driver.execute_script("arguments[0].style.top='100px';return arguments[0];",vol_0)
            time.sleep(2)
        elif ch=='A' and flag==1:#A is for volume to level 2
            vol_2=driver.find_element_by_xpath('//span[@class="slider ui-draggable"]')
            driver.execute_script("arguments[0].style.top='33px';return arguments[0];",vol_2)
            time.sleep(2)
        elif ch=='W' and flag==1:#W is for volumw to max
            vol_3=driver.find_element_by_xpath('//span[@class="slider ui-draggable"]')
            driver.execute_script("arguments[0].style.top='0px';return arguments[0];",vol_3)
            time.sleep(2)
        elif ch=='Y' and flag==1:#Y is to turn shuffle on/off
            shuffle=driver.find_element_by_xpath('//button[@id="shuffle"]')
            shuffle.click()
            time.sleep(2)
        else:
            pass
    except(KeyboardInterrupt):
        close()
        driver.close()
        break