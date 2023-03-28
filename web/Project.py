import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras
import io
import tensorflow as tf
from PIL import Image
from keras.models import load_model
from tensorflow.keras.preprocessing import image
model = load_model('Fruits_Classification_1.model')
img_test = 'testing/1004.jpg'
#Xử lí ảnh trước khi phân loại
categories = ["Apple Red" , "Banana", "Cherry Rainier" ,"Eggplant", "Kiwi", "Lemon", "Mango", "Orange", "Peach", "Raspberry"]
def prepare(filepath):
  IMG_SIZE = 100
  img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
  new_array = cv2.resize(img_array, (100,100))
  new_array = new_array/255.0
  return new_array.reshape(-1,100,100,1)

#Hiển thị ảnh đã qua xử lí
#plt.imshow(np.squeeze(prepare(img_test)),cmap='gray')
#plt.show()
#print('Hình ảnh đã qua xử lí')


img1 = image.load_img(img_test, target_size=(100, 100))
img1_tensor = image.img_to_array(img1)
img1_tensor = np.expand_dims(img1_tensor, axis=0)
img1_tensor /= 255.0


#+ str(categories[int(np.argmax(prediction))]))
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    image_pred = Image.open(io.BytesIO(bytes_data))
    st.image(image_pred)
    image= cv2.cvtColor(np.array(image_pred), cv2.COLOR_RGB2BGR)
    cv2.imwrite("ima.png",image)
    img_test ="ima.png"
    prediction = model.predict([prepare(img_test)])
    #print(prediction)
    #print('Giá trị dự đoán: ' + str(np.argmax(prediction)))
    #print('')
    st.write('Tên loại quả: ' + str(categories[int(np.argmax(prediction))]))