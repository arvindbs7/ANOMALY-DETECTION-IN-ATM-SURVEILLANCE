import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import time
#################################################################################
import warnings
import os
import tensorflow as tf

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Adjust TensorFlow for compatibility warnings (if needed)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#################################################################################

from openalpr_ocr import ocr
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess

import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3
from warnings import filterwarnings
filterwarnings('ignore')

import telepot
bot = telepot.Bot("7571085950:AAG97USNilVsH2uYooeKdGoGdbYa4SNqTJY")
ch_id="1662818653"

Count_Helmet =0
Helmet_Detected =0


second_pass =False
class YOLO(object):
    _defaults = {
            "model_path": 'Unusual_Txt/trained_weights_final.h5',
           # "model_path": 'E:/P_2020/Dataset/OIDv4_ToolKit/Waste_Manage/trained_weights_final_waste.h5',
            "anchors_path": 'model_data/yolo_anchors.txt',
            #"classes_path": 'E:/P_2020/Dataset/OIDv4_ToolKit/Waste_Manage/4_CLASS_test_classes_waste.txt',
            "classes_path": 'Unusual_Txt/4_CLASS_test_classes.txt',
            "score" : 0.3,
            "iou" : 0.45,
            "model_image_size" : (416, 416),
            "text_size" : 1,
    }
  
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class1()
        self.anchors = self._get_anchors1()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate1()

    def _get_class1(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors1(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate1(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = image_preporcess(np.copy(image), tuple(reversed(self.model_image_size)))
            image_data = boxed_image

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],#[image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        global second_pass
        if second_pass == True:
            yolo = YOLO()

        thickness = (image.shape[0] + image.shape[1]) // 600
        fontScale=1
        ObjectsList = []

        if len(out_classes):
            print(out_classes)
            clss = str(out_classes)
            c_person = clss.count('5')
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            #label = '{}'.format(predicted_class)
            scores = '{:.2f}'.format(score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            mid_h = (bottom-top)/2+top
            mid_v = (right-left)/2+left
            
            if predicted_class in ['Axe','Handgun','Knife','Missile','Rifle']:
##                print('weapon detected')
                
                cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)

                # get text size
                (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)

                # put text rectangle
                cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)

                # put text above rectangle
                cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)

                # add everything to list
                ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])
                cv2.imwrite("out.jpg",image)
                myobj = gTTS(text=f"{predicted_class} Detected",lang='en', slow =False)
                myobj.save("voice.mp3")


                
                print('\n------------Playing--------------\n')
                song = MP3("voice.mp3")
                pygame.mixer.init()
                pygame.mixer.music.load('voice.mp3')
                pygame.mixer.music.play()
                time.sleep(song.info.length)
                pygame.quit()
                print('\n------------Playing Done--------------\n')


                
                print('\n------------Message Sending--------------\n')
                bot.sendMessage(ch_id, f"{predicted_class} Detected")
                bot.sendPhoto(ch_id, photo=open("out.jpg", "rb"))
                print('\n------------Message Sent--------------\n')

            elif predicted_class == 'Helmet' and  float(scores) > 0.40 :
##                print('Helmet Detected with more 70 % Accuracy')
                global Helmet_Detected
                Helmet_Detected = Helmet_Detected + 1

                if Helmet_Detected > 1 :
##                    print('Helmet_Detected at 2')
                    
                    cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)

                    # get text size
                    (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)

                    # put text rectangle
                    cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)

                    # put text above rectangle
                    cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)

                    # add everything to list
                    ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])
                    cv2.imwrite("out.jpg",image)
                    myobj = gTTS(text=f"{Helmet_Detected} Helmet Detected",lang='en', slow =False)
                    myobj.save("voice.mp3")
                    print('\n------------Playing--------------\n')
                    song = MP3("voice.mp3")
                    pygame.mixer.init()
                    pygame.mixer.music.load('voice.mp3')
                    pygame.mixer.music.play()
                    time.sleep(song.info.length)
                    pygame.quit()
                    print('\n------------Playing Done--------------\n')
                    print('\n------------Message Sending--------------\n')

                    
                    bot.sendMessage(ch_id, f"{Helmet_Detected} Helmet Detected")
                    bot.sendPhoto(ch_id, photo=open("out.jpg", "rb"))
                    print('\n------------Message Sent--------------\n')
                    Helmet_Detected =0


            elif predicted_class == 'Person':
                cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)

                # get text size
                (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)

                # put text rectangle
                cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)

                # put text above rectangle
                cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)

                # add everything to list
                ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])
                if c_person > 1 :
##                    print('Helmet_Detected at 2')
                    
                    
                    cv2.imwrite("out.jpg",image)
                    myobj = gTTS(text=f"{c_person} Person Detected",lang='en', slow =False)
                    myobj.save("voice.mp3")
                    print('\n------------Playing--------------\n')
                    song = MP3("voice.mp3")
                    pygame.mixer.init()
                    pygame.mixer.music.load('voice.mp3')
                    pygame.mixer.music.play()
                    time.sleep(song.info.length)
                    pygame.quit()
                    print('\n------------Playing Done--------------\n')
                    print('\n------------Message Sending--------------\n')

                    
                    bot.sendMessage(ch_id, f"{c_person} Person Detected")
                    bot.sendPhoto(ch_id, photo=open("out.jpg", "rb"))
                    print('\n------------Message Sent--------------\n')
                    

                        ## Detecting no. of faces in the frames .

            else :
##                print('lllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll')
                if predicted_class == 'Helmet' and float(scores) < 0.511 :
                     label = 'Unknown'
##                     print('Prediction ..............')
  

                     cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)


                     (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)


                     cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)


                     cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)


                     ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])

                cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)

                # get text size
                (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)

                # put text rectangle
                cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)

                # put text above rectangle
                cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)

                # add everything to list
                ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])

            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)

                # get text size
            (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)

                # put text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)

                # put text above rectangle
            cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)

                # add everything to list
            ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])

                

        return image, ObjectsList

    def close_session(self):
        self.sess.close()

    def detect_img(self, image):
        #image = cv2.imread(image, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        r_image, ObjectsList = self.detect_image(original_image_color)
        return r_image, ObjectsList
##############################################################################################################################################################################
##########################################################################################################################################################################

if __name__=="__main__":
    yolo = YOLO()

    # set start time to current time
    start_time = time.time()
    # displays the frame rate every 2 second
    display_time = 2
    # Set primarry FPS to 0
    fps = 0

    # we create the video capture object cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("We cannot open webcam")

    while True:
        ret, frame = cap.read()
        # resize our captured frame if we need
        frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)

        # detect object on our frame
        r_image, ObjectsList = yolo.detect_img(frame)

        

        # show us frame with detection
        cv2.imshow("ATM CAMERA", r_image)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        # calculate FPS
        fps += 1
        TIME = time.time() - start_time
        if TIME > display_time:
##            print("FPS:", fps / TIME)
            fps = 0 
            start_time = time.time()


    cap.release()
    cv2.destroyAllWindows()
    yolo.close_session()



