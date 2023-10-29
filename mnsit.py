import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, utils
from tensorflow import keras
"""
(imgTrain, lblTrain),(imgTest,lblTest) = datasets.mnist.load_data()

imgTrain = utils.normalize(imgTrain,axis=1)
imgTest = utils.normalize(imgTest,axis=1)

model = models.Sequential()
model.add(layers.Flatten(input_shape = (28,28)))
model.add(layers.Dense(units=128, activation='relu'))
model.add(layers.Dense(units=128, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))

model.compile(
optimizer= keras.optimizers.Adam(), 
loss= keras.losses.SparseCategoricalCrossentropy(from_logits= True), 
metrics= ['accuracy'])

model.fit(imgTrain, lblTrain, epochs=3)

loss , accuracy = model.evaluate(imgTrain,lblTrain)
print(accuracy)
print(loss)


model.save('digits_Classifier')
"""
model = keras.models.load_model('digits_Classifier')
for x in range(1,10):
    img = cv.imread(f"{x}.png")[:,:,0]
    img = np.invert(np.array([img]))
    predication = model.predict(img)
    print(f"Actual : {x} | Predicted : {np.argmax(predication)}")
    #plt.imshow(img[0],cmap=plt.cm.binary)
    #plt.show()
