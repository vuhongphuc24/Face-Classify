from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV3Small,imagenet_utils

# Load image
filepath = ['https://github.com/vuhongphuc24/Face-Classify/tree/main/data/gates',
            'https://github.com/vuhongphuc24/Face-Classify/tree/main/data/jack',
            'https://github.com/vuhongphuc24/Face-Classify/tree/main/data/modi',
            'https://github.com/vuhongphuc24/Face-Classify/tree/main/data/musk',
            'https://github.com/vuhongphuc24/Face-Classify/tree/main/data/trump']
label = ['gates','jack','modi','musk','trump']
new_size = (224,224)
def load_image(filepath,label,new_size):
    x = []
    y=  []
    for path in filepath:
        y_label = label[filepath.index(path)]
        for filename in os.listdir(path):
            if filename.endswith('.jpg'):
                image_path = os.path.join(path, filename)
                image = cv2.imread(image_path)
                resized_image = cv2.resize(image, new_size)
                image = imagenet_utils.preprocess_input(resized_image)
                x.append(image)
                y.append(y_label)
    x = np.array(x)
    y = np.array(y)
    return x,y

x,y = load_image(filepath,label,new_size)
y = LabelEncoder().fit_transform(y)

# Model feature extract
model_feature_extract = MobileNetV3Small(weights='imagenet', include_top=False)

# Feature Extract
features = model_feature_extract.predict(x)
features = features.reshape((features.shape[0], 576*7*7))

# Split
X_train, X_test, y_train, y_test = train_test_split(features,y,train_size=0.8,random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test,y_test,test_size=0.8,random_state=42)

# Model
model = LogisticRegression(solver='lbfgs',multi_class='multinomial')
model.fit(X_train,y_train)

# Evaluate
pred = model.predict(X_test)
print(classification_report(y_test, pred))
