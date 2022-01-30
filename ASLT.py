import os
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
        sentence, sequences = [], []
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
            sequences.insert(0, keyPoints)
            sequences = sequences[:interval]
            
            #predict
            if frame_no >= interval:
                frame_no = 0
                prediction = self.model.predict(np.array(sequences))[0]
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

    def GetDataFromPath(
        self, 
        data_path, 
        saveNpy = True):
        #each word should be on separate folders
        label_map = {label: num for num, label in enumerate(self.words)}
        sequences, labels = [], []
        for word in self.words:
            word_path = os.path.join(data_path,word)
            print(f'Obtaining data for "{word}"...')
            if os.path.isdir(word_path):
                for file_name in os.listdir(word_path):
                    file_path = os.path.join(word_path, file_name)
                    
                    #check if numpy file already exists
                    npy_path = f"{file_path.split('.')[0]}.npy"
                    vid_path = f"{file_path.split('.')[0]}_np"
                    if os.path.isfile(npy_path): #load numpy
                        keyPoints = np.load(npy_path)
                        sequences.append(keyPoints)
                        labels.append(label_map[word])
                    elif os.path.isdir(vid_path): #load numpy for vid
                        for npy_path in os.listdir(vid_path):
                            npy_path = os.path.join(vid_path,npy_path)
                            keyPoints = np.load(npy_path)
                            sequences.append(keyPoints)
                            labels.append(label_map[word])
                    else: #load file
                        if os.path.isfile(file_path):
                            #identify file
                            file_ext = file_name.split('.')[1]
                            if file_ext in ['jpg', 'png', 'jpeg']: #Image file
                                frame = cv.imread(file_path)
                                #add to sequences
                                keyPoints = self.tracker.GetKeypointsData(self.tracker.Detect(frame), noFace = True)
                                sequences.append(keyPoints)
                                labels.append(label_map[word])
                                #save to numpy
                                if saveNpy:
                                    np.save(npy_path, keyPoints)
                            elif file_ext in ['mp4', 'mkv']: #Video file
                                cap = cv.VideoCapture(file_path)
                                vid_frame = 0
                                if not os.path.isdir(vid_path):
                                    os.makedirs(vid_path)
                                while cap.isOpened():
                                    isTrue, frame = cap.read()
                                    if not isTrue:
                                        break
                                    #add to sequences
                                    keyPoints = self.tracker.GetKeypointsData(self.tracker.Detect(frame), noFace = True)
                                    sequences.append(keyPoints) 
                                    labels.append(label_map[word])
                                    npy_path = os.path.join(vid_path,f'{vid_frame}.npy')
                                    vid_frame += 1
                                    if saveNpy:
                                        np.save(npy_path, keyPoints)
                                    
                            else:
                                print('File is not recognized')
                                return None
                        elif os.path.isdir(file_path):
                            pass
                        else:
                            print(f'{file_name} does not exists!')
            else:
                print(f'{word_path} does not exists!')
                return None
        return sequences, labels      
        
        