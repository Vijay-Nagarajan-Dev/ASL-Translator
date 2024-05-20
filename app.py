#imports (steelers on top)
from flask import Flask, render_template, Response
import cv2
from cv2 import VideoWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from cv2 import VideoWriter_fourcc 

data = pd.read_csv("sign_mnist_train.csv")
letterData = data["label"].values
data.drop("label", axis=1, inplace= True)

imageData = data.values
imageData = np.array([np.reshape(i, (28,28)) for i in imageData])
imageData = np.array(([i.flatten() for i in imageData]))

binarizer = LabelBinarizer()

letterData = binarizer.fit_transform(letterData)

x_train, x_test, y_train, y_test = train_test_split(imageData, letterData, test_size=0.82, random_state=101)

y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

# Initialize the Support Vector Classifier (SVC) model
model = SVC(kernel='rbf', C=1.0, random_state=42)

# Train the model using the training data
model.fit(x_train, y_train)

# Make predictions on the test data
y_prediction = model.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_prediction)


# Print the accuracy
print(f"Accuracy of the SVC model: {accuracy:.2f}")


def convertNumberToLetter(result):
  labelsToLetters = {
    0:"A",
    1:"B",
    2:"C",
    3:"D",
    4:"E",
    5:"F",
    6:"G",
    7:"H",
    8:"I",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
  }

  res = int(result)
  return labelsToLetters[res]


app = Flask(__name__)

camera = cv2.VideoCapture(0)
def gen_frames():  
    while True:
        #reads camera frame loop
        success, frame = camera.read()
        #if cant read
        if not success:
            break
        #if can read, continuously run frame loop, converts frames into images
        else:
             # Resize frame to (28, 28)
           # Define the region of interest (ROI) coordinates
            roi_x, roi_y, roi_width, roi_height = 100, 100, 200, 200

            # Extract the ROI from the frame
            roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]


            # Convert ROI to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            gray_roi = cv2.resize(gray_roi, (28, 28))

            gray_roi = np.array(gray_roi)
            

            # Predict using the trained model
            prediction = model.predict(np.array([gray_roi.flatten()]))
            print(prediction)
            prediction = convertNumberToLetter(prediction[0])

            # Display the prediction on the frame
            cv2.putText(roi, f"Letter: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if cv2.waitKey(33) == ord('a'):
            cv2.destroyAllWindows() 
            camera.release()
            break


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)