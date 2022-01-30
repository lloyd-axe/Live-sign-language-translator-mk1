import cv2 as cv
import numpy as np
from keras.models import load_model
from MediaPipeHelper import Tracker

class ASLTranslator: #ASL Translator
    def __init__ (
        self, 
        model, 
        words):
        #Required model input shape will be (x, 258)
        self.model = load_model(model)
        self.words = words
        self.tracker = Tracker()
        
    def RunVideo(
        self, 
        vid = "",
        draw = [True, True, True, True], 
        threshold = 0.75, 
        interval = 10, 
        wordCount = 5, 
        displayText = True, 
        textColor= (0,0,255), 
        fontSize = 1):
        #internal parameters
        sentence, sequence = [], []
        frame_no = 0
        
        #Start video
        cap = cv.VideoCapture(0) if vid == "" else cv.VideoCapture(vid)
            
        while cap.isOpened():
            isTrue, frame = cap.read()
            if not isTrue:
                break
            
            #get data
            results = self.tracker.Detect(frame)
            keyPoints = self.tracker.GetKeypointsData(results, noFace = True)
            frame_no += 1
            sequence.insert(0, keyPoints)
            sequence = sequence[:interval]
            
            #predict
            if frame_no >= interval:
                frame_no = 0
                prediction = self.model.predict(np.array(sequence))[0]
                if prediction[np.argmax(prediction)] >= threshold:
                    #form a sentence
                    word = self.words[np.argmax(prediction)]
                    if len(sentence) > 0:
                        if word != sentence[-1]:
                            sentence.append(word)
                    else:
                        sentence.append(word)
                if len(sentence) > wordCount:
                    sentence = sentence[-wordCount:]
                   
            #visualization
            self.tracker.DrawLandmarks(frame, results, draw[0], draw[1], draw[2], draw[3])
            if displayText:
                cv.putText(
                    frame, 
                    ' '.join(sentence), 
                    (10,50), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    fontSize, 
                    textColor, 
                    2, 
                    cv.LINE_AA)
            cv.imshow('Capture', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
            
        
        