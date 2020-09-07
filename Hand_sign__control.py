import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from selenium import webdriver 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.common.keys import Keys 
import pickle
import time
def prediction(img):
    test_img = torch.from_numpy(np.expand_dims(np.expand_dims(cv2.resize(img, (28, 28)), 0), 0))
    model.eval()
    y=model.forward(test_img)
    result=F.softmax(y,dim=1)
    max_val=0
    index=0
    max_val,index=torch.max(result,axis=1)
    if max_val>0.75:
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
model.load_state_dict(torch.load('trained.pth'))
model.eval()
key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
flag=0
ch_='_'
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
        hand=gray[y:y_,x:x_]
        cv2.imshow("Capturing",hand)
        ch=prediction(hand)
        if ch != None:
            print(ch)
        if ch=='O':
            if flag==0:
                flag=1
                driver_location='./geckodriver.exe'
                driver = webdriver.Firefox(executable_path=driver_location)
                driver.get("https://www.jiosaavn.com/album/guitar/hk5Nf1gQ76I_")
                play=driver.find_element_by_xpath('//a[@class="c-btn c-btn--primary"]')
                driver.minimize_window()
                play.click()
                WebDriverWait(driver,600)
                play_pause=1
                time.sleep(1)
            else:
                driver.close()
                close()
        elif ch=='D' and flag==1:
            vol_1=driver.find_element_by_xpath('//div[@class="c-slider__level"]')
            driver.execute_script("arguments[0].style.height='33%';return arguments[0];",vol_1)
            time.sleep(2)
        elif ch=='V' and flag==1:
            if play_pause==0:
                el = WebDriverWait(driver,600).until(lambda d: d.find_element_by_xpath('//span[@class="o-icon-play o-icon--xlarge"]'))
                pause=driver.find_element_by_xpath('//span[@class="o-icon-play o-icon--xlarge"]')
                pause.click()
                play_pause=1
            else:
                el = WebDriverWait(driver,600).until(lambda d: d.find_element_by_xpath('//span[@class="o-icon-pause o-icon--xlarge"]'))
                play=driver.find_element_by_xpath('//span[@class="o-icon-pause o-icon--xlarge"]')
                play.click()
                play_pause=0
            time.sleep(2)
        elif ch=='G' and flag==1:
            next_song=driver.find_element_by_xpath('//span[@class="o-icon-next o-icon--xlarge"]')
            next_song.click()
            time.sleep(1)
        elif ch=='P' and flag==1:
            previous_song=driver.find_element_by_xpath('//span[@class="o-icon-previous o-icon--xlarge"]')
            previous_song.click()
            time.sleep(2)
        elif ch=='B' and flag==1:
            vol_0=driver.find_element_by_xpath('//div[@class="c-slider__level"]')
            driver.execute_script("arguments[0].style.height='7%';return arguments[0];",vol_0)
            time.sleep(2)
        elif ch=='A' and flag==1:
            vol_2=driver.find_element_by_xpath('//div[@class="c-slider__level"]')
            driver.execute_script("arguments[0].style.height='66%';return arguments[0];",vol_2)
            time.sleep(2)
        elif ch=='W' and flag==1:
            time.sleep(1)
            vol_3=driver.find_element_by_xpath('//div[@class="c-slider__level"]')
            driver.execute_script("arguments[0].style.height='94%';return arguments[0];",vol_3)
        elif ch=='Y' and flag==1:
            time.sleep(1)
            try:
                shuffle=driver.find_element_by_xpath('//span[@class="u-color-js-green o-icon-shuffle o-icon--xlarge"]')
                shuffle.click()
            except:
                shuffle=driver.find_element_by_xpath('//span[@class="o-icon-shuffle o-icon--xlarge"]')
                shuffle.click()
        else:
            pass
    except(KeyboardInterrupt):
        close()
        driver.close()
        break
