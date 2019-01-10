import numpy as np  
import sys,os  
import cv2
from math import sqrt
caffe_root = '/home/joy_hsiao/Haitec/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  
import time
GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)


net_file= 'example/8_class_prototxt/MobileNetSSD_deploy.prototxt'  
caffe_model='MobileNetSSD_8_classes_deploy.caffemodel'
#caffe_model='snapshot/mobilenet_iter_107000.caffemodel' # not merge_bn

#net_file= 'MobileNetv2SSDLite_deploy.prototxt'  
#caffe_model='MobileNetv2SSDLite_deploy.caffemodel'

test_dir = "images"

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  
"""
CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
"""

CLASSES = ['background',
           'bike', 'bus', 'car', 'motorbike', 'person', 'truck', 'van']


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)



#front
path = '/home/joy_hsiao/Haitec/video/'
fileName = 'AV1-20181116_172502_back.avi'
cap = cv2.VideoCapture(path+fileName) 
#cap = cv2.VideoCapture('/home/joy_hsiao/Video/train33_30fps.mp4') #start:30000
#cap = cv2.VideoCapture('/home/joy_hsiao/Video/AV1-20181116_162151_right.avi') 
#cap = cv2.VideoCapture('/home/joy_hsiao/Video/left.mp4') 
#cap = cv2.VideoCapture('/home/ubuntu/Videos/YDXJ0349.mp4')
#cap = cv2.VideoCapture('/media/ubuntu/Database1/night_view/100GOPRO-20180617/h.264/front/front2.mp4')

#back
#cap = cv2.VideoCapture('/media/ubuntu/Database1/night_view/100GOPRO-20180617/h.264/back/back2.mp4') #start:40000

#left
#cap = cv2.VideoCapture('/media/ubuntu/Database1/night_view/100GOPRO-20180617/h.264/left/left9.mp4') #start:0

#right
#cap = cv2.VideoCapture('/media/ubuntu/Database1/night_view/100GOPRO-20180617/h.264/right/right5.mp4')

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fontScale = 1
thickness = 3
accelerationFrame=0 # five frames update acceleration
preDis = -99.99 # count 1's frame Distance
preSec = -99.99 # count 1's sec Distance
V = -99.99

def detect(ori_img):
    global nearDistance
    global accelerationFrame
    global preDis
    global preSec
    global V
    img = preprocess(ori_img)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(ori_img, out)

    for i in range(len(box)):
        #p1 = (box[i][0], box[i][1])
        #p2 = (box[i][2], box[i][3])
        xmin = box[i][0]
        ymin = box[i][1]
        xmax = box[i][2]
        ymax = box[i][3]
        #cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), (255,255,255))
        p3 = (max(xmin, 15), max(ymin, 15))
        #title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        title = "%s" % (CLASSES[int(cls[i])])
        #coordinate = "(%d,%d) " % (xmin, ymin)

        color_picker = {
        'person' : (255, 0, 0),
        'car' : (0, 255, 0),
        'motorbike' : (0, 0, 255),
        'bus' : (0, 255, 255),
        'bike' : (255, 255, 0),
        'truck' : (128, 255, 0),
        'van' : (255, 128, 0)
        }

        color = color_picker[CLASSES[int(cls[i])]]
        
        if fileName.find('right') > -1:
            slashDistance = pow(Width-xmax,2) + pow(Height-ymax,2)
            slashDistance = sqrt(slashDistance)
            pre_ymax=ymax
            ymax=int(478-slashDistance)
            if ymax<240 and ymax>0: #slash distance adjustment
                #ymax=ymax+((CloseDis/pow(ymax,2))*1000)*2
                ymax=(pre_ymax+ymax)/2 
        if fileName.find('left') > -1:
            #cv2.circle(ori_img, (xmin, ymax), 5, (255, 0, 255), -1)
            slashDistance = pow(xmin-Width,2) + pow(Height-ymax,2)
            slashDistance = sqrt(slashDistance)
            pre_ymax=ymax
            ymax=int(478-slashDistance)
            if ymax<240 and ymax>0: #slash distance adjustment
                #ymax=ymax+((CloseDis/pow(ymax,2))*1000)*2
                ymax=(pre_ymax+ymax)/2
            distance=0
            if ymax>190:
                distance = LambdaBack*(1.0/(ymax-Level)-1.0/(CloseDis-Level))+5.5
            #print pre_ymax, distance, ymax, slashDistance

        if ymax > 190:
            if fileName.find('back') > -1:
                pre_ymax=ymax
                distance = LambdaBack*(1.0/(ymax-Level)-1.0/(CloseDis-Level))+5.5
            if fileName.find('right') > -1:
                distance = LambdaRight*(1.0/(ymax-Level)-1.0/(CloseDis-Level))+2.7
            #print ("2 ", ymax, slashDistance, distance)
            if fileName.find('left') > -1:
                distance = LambdaLeft*(1.0/(ymax-Level)-1.0/(CloseDis-Level))+3.2
            if distance < 0.0:
                distance=0.0
            if(nearDistance > distance):
                nearDistance = distance
            if(distance<90):
                distanceShow = "%.2f " % distance

                if(preDis<-99):
                    preDis = distance
                Sec = time.time()
                S = Sec - preSec
                V = ((distance-preDis)/1000.0)/(S/60.0/60.0) # m/sec to km/hr
                #print(preDis, distance, distance-preDis, preSec, Sec, S)
                preSec = Sec
                preDis = distance
                show = "%.2f" % V

                cv2.rectangle(ori_img, (xmin, ymin), (xmax, pre_ymax), color, 2)
                cv2.putText(ori_img, distanceShow + show, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)
            


#    if(accelerationFrame==0):
#        preDis = distance
#        preSec = time.time()
#        print("==", accelerationFrame, preDis, distance, distance-preDis)
#        accelerationFrame+=1
#    elif(accelerationFrame<5):
#        print("**", accelerationFrame, preDis, distance, distance-preDis)
#        accelerationFrame+=1
#    else:
#        Sec = time.time()
#        S = Sec - preSec
#        V = (distance-preDis)*100/S
#        print(preSec, Sec, S, V)
#        print("##",accelerationFrame, preDis, distance, distance-preDis)
#        accelerationFrame=0
    return ori_img

#        Level   5M  10M 15M
#back    190     270 235 225
#        Level   2.7M   8M      13M     18M
#right   190     330    250     234     228
#left    190     316    235     216     210
Level = 190
BackClosestDis=270
RightClosestDis=330
LeftClosestDis=316
LambdaBack = (15-5)/(1.0/(225-Level)-1.0/(BackClosestDis-Level))
LambdaRight = (18-2.7)/(1.0/(228-Level)-1.0/(RightClosestDis-Level))
LambdaLeft = (18-2.7)/(1.0/(210-Level)-1.0/(LeftClosestDis-Level))

#cap.set(cv2.CAP_PROP_POS_FRAMES, 3000)

out = cv2.VideoWriter('/home/joy_hsiao/Haitec/video/Out/'+fileName, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (718,478))
#out = cv2.VideoWriter('/home/joy_hsiao/Haitec/video/Out/'+fileName, cv2.VideoWriter_fourcc('M','P','4','V'), 30, (718,478))
while(cap.isOpened()):
    ret, frame = cap.read()
    #print("size =", (cap.get(cv2.CAP_PROP_FRAME_WIDTH)), (cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #src_frame = cv2.resize(frame, (718, 478))
    #frame = cv2.resize(frame, (718, 478))

    framePos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    timer = Timer()
    timer.tic()

    if fileName.find('left') > -1:
        roiImg = frame[0:478,150:718]
        Height = roiImg.shape[0]
        Width = 0
        CloseDis=LeftClosestDis
    elif fileName.find('right') > -1:
        roiImg = frame[0:478,0:580]
        Height = roiImg.shape[0]
        Width = roiImg.shape[1]
        CloseDis=RightClosestDis
    else:
        roiImg = frame[0:478,0:718]
        Height = roiImg.shape[0]
        Width = roiImg.shape[1]
        CloseDis=BackClosestDis

    #try:
    #    roiImg
    #except NameError:
    #    roiImg = frame[0:478,0:718]
    
    nearDistance=99.00 # closest car distance
    dst_img = detect(roiImg)
    if fileName.find('left') > -1:
        frame[0:478,150:718]=roiImg
    elif fileName.find('right') > -1:
        frame[0:478,0:580]=roiImg
    else:
        frame[0:478,0:718]=roiImg
    
    timer.toc()
    if(nearDistance<5):
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
    elif(nearDistance<10):
        cv2.circle(frame, (30, 30), 10, (0, 255, 255), -1)
    else:
        cv2.circle(frame, (30, 30), 10, (0, 255, 0), -1)
    cv2.putText(frame, "FPS : " + str(int(1 / timer.total_time)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 0, 255), 4, cv2.LINE_AA)
    #cv2.line(frame, (580,0), (580, 479), (0,255,0), 1)
    #cv2.line(dst_img, (0,210), (718, 210), (0,255,0), 1)
    cv2.imshow("result", frame)
    #cv2.waitKey(-1)
    #cv2.imwrite("front2/src/"+ 'ft2_' + str(int(framePos)) + '.jpg', src_frame)
    cv2.imwrite('result_front2/' + 'ft2_' + str(int(framePos)) + '.jpg', frame)
    out.write(frame)
    #print("detection time :", timer.total_time)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#for f in os.listdir(test_dir):
#    if detect(test_dir + "/" + f) == False:
#       break
