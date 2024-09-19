
import keras
from keras.models import Sequential #sequential API allows you to create models layer-by-layer
from keras.layers import Conv2D #Conv2D class is used to determine whether a bias vector will be added to the convolutional layer.
from keras.layers import MaxPooling2D #Max pooling is a sample-based discretization process
from keras.layers import Flatten #Reshape the input data into a format suitable for the convolutional layers, using X_train. ...
from keras.layers import Dense #A dense layer thus is used to change the dimensions of your vector.

import matplotlib.pyplot as plt
import numpy as np
from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import cv2
from PIL import Image
#from matplotlib import pyplot as plt

#from skimage.feature import greycomatrix, greycoprops


main = tkinter.Tk()
#main.title("Lung Cancer Nodule Feature Extraction Using Digital Image Processing System" )
main.geometry("650x400")
main.configure(bg = 'LightSkyBlue4')

global filename

def uploadImage():
    global filename
    filename = askopenfilename(initialdir = "LUNG")
    imagepath.config(text=filename)

def watershed():
    img = cv2.imread(filename)
    b,g,r = cv2.split(img)
    rgb_img = cv2.merge([r,g,b])

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
    kernel = np.ones((2,2),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

# sure background area
    sure_bg = cv2.dilate(closing,kernel,iterations=3)

# Finding sure foreground area
    dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)

# Threshold
    ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

# Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

# Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    count = 0
    if  count >= 0:
        kmeansbutton['state'] = "normal"






        plt.subplot(421),plt.imshow(rgb_img)
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(422),plt.imshow(thresh, 'gray')
        plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])

        plt.subplot(423),plt.imshow(closing, 'gray')
        plt.title("morphologyEx:Closing:2x2"), plt.xticks([]), plt.yticks([])
        plt.subplot(424),plt.imshow(sure_bg, 'gray')
        plt.title("Dilation"), plt.xticks([]), plt.yticks([])

        plt.subplot(425),plt.imshow(dist_transform, 'gray')
        plt.title("Distance Transform"), plt.xticks([]), plt.yticks([])
        plt.subplot(426),plt.imshow(sure_fg, 'gray')
        plt.title("Thresholding"), plt.xticks([]), plt.yticks([])

        plt.subplot(427),plt.imshow(unknown, 'gray')
        plt.title("detect part"), plt.xticks([]), plt.yticks([])

        plt.subplot(428),plt.imshow(img, 'gray')
        plt.title("Result from Watershed"), plt.xticks([]), plt.yticks([])

        plt.tight_layout()
        plt.show()
    else:
        kmeansbutton['state'] = "disabled"






def prediction():
    global filename
    classifier = Sequential()
# Input layer
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu')) #ReLU stands for rectified linear unit, and is a type of activation function





# In[3]:


    classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Hidden layer 1
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Hidden layer 2

    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Hidden layer 3
    classifier.add(Flatten())

# Hidden layer 4
    classifier.add(Dense(activation = 'relu',units=128))
    classifier.add(Dense(activation = 'sigmoid',units=1))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    classifier.summary()


# In[4]:


    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('D:/Implementation-of-Lung-Cancer-Nodule-Detection-master/5. Source Code and screen shot/Lung_Cancer/Dataset/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

    test_set = test_datagen.flow_from_directory('D:/Implementation-of-Lung-Cancer-Nodule-Detection-master/5. Source Code and screen shot/Lung_Cancer/Dataset/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# In[7]:



    classifier.fit_generator(training_set, steps_per_epoch=None, epochs=100, verbose=1, callbacks=None, validation_data=test_set, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)






# In[8]:


    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img(filename , target_size = (64, 64))
    test_image


# In[10]:


    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis = 0)
    test_image


# In[11]:


    result = classifier.predict(test_image)
    result


# In[12]:


    training_set.class_indices






# In[13]:


    if result[0][0] == 0:
        messagebox.showinfo("Detected Cancer is", "Benign Cancer")

    else:
        messagebox.showinfo("Detected Cancer is", "Malignant Cancer")





def GLCM():

    img = cv2.imread(filename,0)
    img = cv2.medianBlur(img,5)

    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    count = 0

    titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]

    if count >= 0:
        histobutton['state'] = "normal"
        for i in range(4):

            plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')

            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
            plt.show()
    else:
         histobutton['state'] = "disabled"





def histogram():
    image = cv2.imread(filename)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    histr = cv2.calcHist([image],[0],None,[256],[0,256])
    cv2.imwrite("test.jpg",image)
    i = Image.open("test.jpg")
    pixels = i.load() # this is not a list, nor is it list()'able
    width, height = i.size
    img = np.zeros((height,width,1),np.uint8)
    count = 0
    for x in range(width):
        for y in range(height):
            cpixel = pixels[x, y]
            if cpixel > 230:
                img[y,x,0] = cpixel
                count = count + 1
            else:
                img[y,x,0] = 0
    if count > 0:
        messagebox.showinfo("Changes detected in LUNG","Change detected in LUNG. LUNG is abnormal. Run SVM to view abnormal patches")
        predbutton['state'] = "normal"
        plt.plot(histr)
        plt.show()
    else:
        predbutton['state'] = "disabled"
        messagebox.showinfo("No change detected in LUNG","No change detected in LUNG. LUNG is normal")










def exit():
    global main
    main.destroy()

#font1 = ('times', 14, 'bold')

font1 = ('times', 14, 'bold')
title = Label(main, text="Lung Cancer Nodule Feature Extraction Using Digital Image Processing System", justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')
#title.config(font=font)
title.config(height=3, width=120)
title.place(x=50,y=5)
title.pack()
uploadbutton = Button(main, text="Upload Lung CT Image", command=uploadImage)
uploadbutton.place(x=50,y=100)
uploadbutton.config(font=font1)

imagepath = Label(main)
imagepath.place(x=250,y=50)
imagepath.config(font=font1)

waterbutton = Button(main, text="Image Processing And Segmentation", command=watershed)
waterbutton.place(x=50,y=150)
waterbutton.config(font=font1)

kmeansbutton = Button(main, text="Feature Extraction", command=GLCM)
kmeansbutton.place(x=50,y=200)
kmeansbutton.config(font=font1)
kmeansbutton['state'] = "disabled"

histobutton = Button(main, text="Histogram", command=histogram)
histobutton.place(x=50,y=250)
histobutton.config(font=font1)
histobutton['state'] = "disabled"



predbutton = Button(main, text="Classification", command=prediction)
predbutton.place(x=50,y=300)
predbutton.config(font=font1)
predbutton['state'] = "disabled"


exitbutton = Button(main, text="Exit Application", command=exit)
exitbutton.place(x=50,y=350)
exitbutton.config(font=font1)

main.mainloop()
