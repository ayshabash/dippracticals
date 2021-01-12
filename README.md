# dippracticals

### 1. Develop a program to display grayscale image using read and write operation.

Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. 
It varies between complete black and complete white.
Importance of grayscaling –

Dimension reduction: For e.g. In RGB images there are three color channels and has three dimensions while grayscaled images are single dimensional.
Reduces model complexity: Consider training neural article on RGB images of 10x10x3 pixel.The input layer will have 300 input nodes. On the other hand, the same neural network will need only 100 input node for grayscaled images.
For other algorithms to work: There are many algorithms that are customized to work only on grayscaled images e.g. Canny edge detection function pre-implemented in OpenCV library works on Grayscaled images only.


#### Below is the code to Grayscale an image-

import cv2
import numpy as np
image = cv2.imread('img20.jpg')
image = cv2.resize(image, (0, 0), None, .25, .25)
grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
numpy_horizontal = np.hstack((image, grey_3_channel))
numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
cv2.imshow('flower', numpy_horizontal_concat)
cv2.waitKey()

#### OUTPUT:

![image](https://user-images.githubusercontent.com/73472521/104284579-43143900-54d8-11eb-9564-8516e6f615e7.png)


### 2.Develop a program to perform linear transformation on image. (Scaling and rotation) 

Scaling an Image :-

Scaling operation increases/reduces size of an image.

import cv2 as c
img=c.imread("img3.jpg")
c.imshow('image',img)
nimg=c.resize(img,(0,0),fx=0.50,fy=0.50)
c.imshow("Result",nimg)
c.waitKey(0)

OUTPUT:

![image](https://user-images.githubusercontent.com/73472521/104284658-5d4e1700-54d8-11eb-9742-e9304c9bf9ef.png)



Rotating an image :-

Images can be rotated to any degree clockwise or otherwise.
We just need to define rotation matrix listing rotation point, degree of rotation and the scaling factor.

import cv2 
import numpy as np 
img = cv2.imread('img22.jfif') 
(rows, cols) = img.shape[:2] 
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 120, 1) 
res = cv2.warpAffine(img, M, (cols, rows)) 
cv2.imshow('image', img)
cv2.waitKey(0) 
cv2.imshow('result',res) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

OUTPUT:

![image](https://user-images.githubusercontent.com/73472521/104284702-7656c800-54d8-11eb-98b8-433645de5bc1.png)

### 4.Write a program to convert color image into gray scale and binary image.

A binary image is the type of image where each pixel is black or white. You can also say the pixels as 0 or 1 value. Here 0 represents black and 1 represents a white pixel.
Thresholding is used to create a binary image from a grayscale image. It is the simplest way to segment objects from a background.
Images are composed of Pixels and in Binary image every pixel value is either 0 or 1 i.e either black or white. it is called bi-level or two level image
while in gray scale ; image can have any value between 0 to 255 for 8-bit color(every pixel is represented by 8 bits) i.e it can have transition between pure black or pure white . It only have intensity value.

So, Gray Scale image can have shades of grey varying between Black and white while Binary image can either of two extreme for a pixel value either white or black

Thresholding is used to create a binary image from a grayscale image. It is the simplest way to segment objects from a background.

import cv2
image=cv2.imread("img19.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(tresh,blackAndWhiteImage)=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow("gray",gray)
cv2.imshow("BINARY",blackAndWhiteImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

Where 127 is the threshold, 255 is the max value and cv2.THRESH_BINARY indicates binary thresholding. You can feed it cv2.THRESH_BINARY_INV instead if you need it inverted.

#### OUTPUT:


### 5.Write a program to convert color image into different color space.

Color spaces are a way to represent the color channels present in the image that gives the image that particular hue. There are several different color spaces and each has its own significance.
Some of the popular color spaces are RGB (Red, Green, Blue), CMYK (Cyan, Magenta, Yellow, Black), HSV (Hue, Saturation, Value), etc.

BGR color space: OpenCV’s default color space is RGB. However, it actually stores color in the BGR format. It is an additive color model where the different intensities of Blue, Green and Red give different shades of color.

HSV color space: It stores color information in a cylindrical representation of RGB color points. It attempts to depict the colors as perceived by the human eye. Hue value varies from 0-179, Saturation value varies from 0-255 and Value value varies from 0-255. It is mostly used for color segmentation purpose.

CMYK color space: Unlike, RGB it is a subtractive color space. The CMYK model works by partially or entirely masking colors on a lighter, usually white, background. The ink reduces the light that would otherwise be reflected. Such a model is called subtractive because inks “subtract” the colors red, green and blue from white light. White light minus red leaves cyan, white light minus green leaves magenta, and white light minus blue leaves yellow.

LAB color space :
L – Represents Lightness.
A – Color component ranging from Green to Magenta.
B – Color component ranging from Blue to Yellow.

YUV: Even though RGB is good for many purposes, it tends to be very limited for many real life applications. People started thinking about different methods to separate the intensity information from the color information. Hence, they came up with the YUV color space. Y refers to the luminance or intensity, and U/V channels represent color information. This works well in many applications because the human visual system perceives intensity information very differently from color information.


import cv2
image=cv2.imread("img20.jpg")
cv2.imshow("old",image)
cv2.waitKey()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV",hsv)
cv2.waitKey(0)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("LAB",lab)
cv2.waitKey(0)
hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
cv2.imshow("HLS",hls)
cv2.waitKey(0)
yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
cv2.imshow("YUV",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()

#### OUTPUT:

![image](https://user-images.githubusercontent.com/73472521/104285151-317f6100-54d9-11eb-84e8-dbe3648aa530.png)



# 6.Develop a program to create an image from 2D array.

NumPy Or numeric python is a popular library for array manipulation. Since images are just an array of pixels carrying various color codes. NumPy can be used to convert an array into image. Apart from NumPy we will be using PIL or Python Image Library also known as Pillow to manipulate and save arrays.

Numpy zeros np.zeros() function in python is used to get an array of given shape and type filled with zeros. You can pass three parameters inside function np.zeros shape, dtype and order. Numpy zeros function returns an array of the given shape.

Getting back the image from converted Numpy Array
Image.fromarray() function helps to get back the image from converted numpy array. We get back the pixels also same after converting back and forth. Hence, this is very much efficient

import numpy as np
from PIL import Image
import cv2 as c 
array = np.zeros([100, 200, 3], dtype=np.uint8)
array[:,:100] = [150, 128, 0] #Orange left side
array[:,100:] = [0, 0, 255]   #Blue right side
img = Image.fromarray(array)
img.save('flower.jpg')
img.show()
c.waitKey(0)

#### OUTPUT:

![image](https://user-images.githubusercontent.com/73472521/104286190-b6b74580-54da-11eb-8665-1c30e21edb2f.png)


