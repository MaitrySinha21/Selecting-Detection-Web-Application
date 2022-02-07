from flask import Flask, render_template, Response, request
from flask_cors import cross_origin
import cv2
import numpy as np

size = 300
cthres = 0.5
nthres = 0.2
classNames = []
classFile = 'coco.names'

with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

classNames = ['background'] + classNames
configPath = 'SSDmobilenet_coco.pbtxt'
weightPath = 'frozenGraph.pb'
net = cv2.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(size,size)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

path = 1
name = 'person'
app = Flask(__name__)
@app.route('/',methods=['GET'])
@cross_origin()
def index():
    return render_template('index.html')


@app.route("/view",methods=["GET","POST"])
@cross_origin()
def view():
    global path,name
    if request.method == 'POST':
        path = request.form['path']
        name = request.form['name']
        return render_template('index.html')

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(path)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        frame = cv2.resize(frame,(880,560))
        try:
            classIds, conf, bbox = net.detect(frame, confThreshold=cthres)
            bbox = list(bbox)
            conf = list(np.array(conf.reshape(1, -1)[0]))
            conf = list(map(float, conf))
            indices = cv2.dnn.NMSBoxes(bbox, conf, score_threshold=cthres, nms_threshold=nthres)
            for i in indices:
                i = i[0]
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                text = classNames[classIds[i][0]]  # +" "+str(round(conf[0]*100,2))+"%"
                if (text == name):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 255), thickness=1)
                    cv2.rectangle(frame, (x, y), (x + 60, y - 14), (0, 255, 255), cv2.FILLED)
                    cv2.putText(frame, text.upper(), (x + 2, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1)


            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        except:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()


def gen(camera):
    Camera = VideoCamera()
    while True:
        frame = Camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host="0.0.0.0",port="5000")
