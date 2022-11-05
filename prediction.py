import pandas as pd
import cv2
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps

X,Y = fetch_openml('mnist_784', version = 1, return_X_y = True )
print(pd.Series(Y).value_counts())

classes = ['A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
n_classes = len(classes)

x_train, x_test, y_train, y_test = train_test_split(X,Y, random_state = 9, train_size = 7500, test_size = 2500)
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0
clf = LogisticRegression(solver = 'saga', multi_class= 'multinomial').fit(x_train_scaled, y_train)
y_pred = clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred) 
print(accuracy)




def get_prediction(image):
        #pillow - image processing
        im_pil = Image.open(Image)
        image_bw = im_pil.convert("L")
        image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)

        pixel_filter = 20
        min_pixel = np.percentage(image_bw_resized_inverted,pixel_filter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
        max_pixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_predict = clf.predict(test_sample)
        return test_predict[0]
    