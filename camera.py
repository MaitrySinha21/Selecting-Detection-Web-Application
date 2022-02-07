import cv2
import numpy as np

path = 'video\street.mp4'
#path = 0
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

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(path)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        frame = cv2.resize(frame,(900,520))
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
                #if (text == 'car'):
                cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 255), thickness=1)
                cv2.putText(frame, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 255), 1)

            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        except:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

