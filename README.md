# Hand_Sign_Music_control
Most of the time while working we dont have space on screen for app to control the background activities like music.
This project is an example of use of handsigns as input for web-automation.

---

#### Requirements for usage
* Python 3.8 or above

    Python libraries:
    * [Pytorch](https://pytorch.org/)
    * CV2
      > pip install opencv-python
    * Selenium
      > pip install selenium
* Geckodriver
(present in the repository)
* Webcam

---

#### Requirements for training CNN model
* Python 3.8 or above

    Python libraries:
    * [Pytorch](https://pytorch.org/)
    * CV2
      > pip install opencv-python
    * Matplotlib
      > pip install matplotlib
      
1. [Usage](#usage)
2. [Training Stats](#training-stats)

---


### Usage
Run the code using command line
###### use of plane background is recommended

A webcam window will open.

![Instructions](https://drive.google.com/uc?export=view&id=1CpgfY4xAcSS2LlmuNEFIMoi_GkoPmGUY)

Make these handsigns in the visible region of output of webcam.

Until the open browser sign is made all instructions will be ignored.
[Demo Video](https://drive.google.com/file/d/1qaKtV5Gom2dx8soXgI0vn_Ne4RITov_0/view?usp=sharing)

---

### Training Stats
Input image size=28x28x1

DataSet:

    Train Set = 9000 Images
    Dev Set = 1800 Images
Architecture used:

    (FC-Fully Connected Dense Layer,F-Features,S-Strides,C-Channels)
    Conv Layer - F(3x3) S(1x1) C(1,64)
    BatchNorm 
    Max Pool - F(2x2) S(2x2)
    Conv Layer - F(3x3) S(1x1) C(64,128)
    BatchNorm
    Max Pool - F(2x2) S(2x2)
    FC(5*5*128,128)
    BatchNorm
    FC(128,9)
Optimizer Used: ADAM

    Learning Rate(alpha)=0.01
    Decay rate=(0.95^i)    {i is epoch number}
Number of Epochs = 10

Mini Batch Size = 1000

![Cost_stats](https://drive.google.com/uc?export=view&id=1cEbbeFhMenBc-5gItIvp-NSXY7pVVfOt) --time in seconds


![Cost Graph](https://drive.google.com/uc?export=view&id=1VrM64E5dJISVl9goOi5Pc8chZYsb4QVj)

#### Train Set Accuracy = 100%

#### Dev Set Accuracy = 99.88% 


---


## Contributors
**[Atharva Kathale](https://github.com/Atharva-K12)**

