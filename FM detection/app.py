from flask import Flask, render_template, Response
import cv2
import pickle

app = Flask(__name__)

# Load your PCA and SVM models here
with open('pca_model.pkl', 'rb') as pca_file:
    pca = pickle.load(pca_file)

with open('svm_model.pkl', 'rb') as svm_file:
    svm = pickle.load(svm_file)

names = {0 : 'Mask', 1 : 'No Mask'} # Replace with your actual names

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX

def generate_frames():
    while True:
        flag, img = capture.read()
        if flag:
            faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 4)
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (50, 50))
                face = face.reshape(1, -1)
                face = pca.transform(face)
                pred = svm.predict(face)
                n = names[int(pred)]
                cv2.putText(img, n, (x, y), font, 1, (244, 250, 250), 2)
            
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)