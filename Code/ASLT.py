import os
import cv2 as cv
import numpy as np
from keras.models import load_model

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from MediaPipeHelper import Tracker #MediaPipeHelper module should be in the same directory

class ASLTranslator: #ASL Translator
    def __init__ (
        self, 
        model = '', 
        words = [], 
        interval = 18):
        
        self.words = words
        
        #Required model input shape will be (x, interval, 138)
        #load model
        if model != '':
            self.model = self._loadModel(model, interval)
        self.tracker = Tracker()
        
    def _loadModel(self, model_name, interval):
        model = self.CreateModel(interval)
        model.load_weights(model_name)
        return model
        
    def CreateModel(self, interval = 18):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(interval,138)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(self.words), activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model
    
    def SplitData(self, x_raw, y_raw):
        x = np.array(x_raw)
        y = to_categorical(y_raw).astype(int)
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.05)
        return x_train, x_test, y_train, y_test
    
    def TrainModel(self, model, x, y, epochs = 30, model_name = ''):
        model.fit(x, y, epochs = epochs)
        if model_name != '':
            model.save(model_name)
        return model
        
    def StartCapture(self, 
        vid = "",
        draw = [True, True, True, True], 
        threshold = 0.75, 
        interval = 18, 
        wordCount = 5, 
        displayText = True, 
        textColor= (0,0,255), 
        fontSize = 1):
        
        sentence, sequences = [], []
        frame_no = 0
        
        #start video
        cap = cv.VideoCapture(0) if vid == "" else cv.VideoCapture(vid)
        while cap.isOpened():
            isTrue, frame = cap.read()
            if not isTrue:
                break
                
            #get data
            results = self.tracker.Detect(frame)
            keyPoints = self.tracker.GetKeypointsData(
                results, 
                noFace = True, 
                posePoints = [11,0,12])
            
            #form sequences
            sequences.append(keyPoints)
            sequences = sequences[1:] if len(sequences) > interval else sequences #limit sequences
            
            frame_no += 1
            
            #predict
            if frame_no == interval:
                frame_no = 0
                prediction = self.model.predict(np.expand_dims(sequences, axis=0))[0]
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
        
    def GetNpyDataFromPath(self, data_path):
        if os.path.isdir(data_path): #check if path exists
            with open(os.path.join(data_path, 'word_list.txt')) as wordlist: #get list of words
                words = [w.rstrip() for w in wordlist]
            
            label_map = {label: num for num, label in enumerate(words)}
            
            sequences, labels = [], []
            for word in words:
                word_path = os.path.join(data_path, word)
                for samp in os.listdir(word_path): #get each sample data
                    sq = []
                    sample_path = os.path.join(word_path, samp)
                    for f in os.listdir(sample_path): #get all frame data in each sample
                        file = os.path.join(sample_path, f)
                        print(f'Loading data for {word}-{str(samp)}-{f} .....')
                        sq.append(np.load(file))
                    sequences.append(sq)
                    labels.append(label_map[word])
        return sequences, labels
    
    def CollectDataforWord(
        self, 
        word, 
        data_path, 
        sampleCount = 20, 
        frameCount = 18, 
        isDraw = True, 
        collectReverse = True):
        
        word_list_path = os.path.join(data_path, 'word_list.txt')
        currentWords = []
        if os.path.isfile(word_list_path):
            with open(word_list_path) as wordlist: #get list of words
                    currentWords = [w.rstrip() for w in wordlist]
        with open(os.path.join(data_path, 'word_list.txt'), 'a') as wordlist:  #create word_list.txt
            if word not in currentWords:
                wordlist.write(word + '\n')
            else:
                print(f'Overriding data for {word}')
        
        #create paths
        word_path = os.path.join(data_path, word)
        if not os.path.isdir(word_path):
            os.makedirs(word_path)
        for s in range(sampleCount):
            s_path = os.path.join(word_path, str(s))
            sr_path = os.path.join(word_path, f'{s}_r')
            if not os.path.isdir(s_path):
                os.makedirs(s_path)
            if collectReverse and not os.path.isdir(sr_path):
                os.makedirs(sr_path)
        
        #collect
        cap = cv.VideoCapture(0)
        for s in range(sampleCount): #collect samples
            if s == 0:
                _, frame = cap.read()
                cv.imshow('Capture', frame) #starting frame
                cv.waitKey(2000)
                
            for fn in range(frameCount): #collect each frame data for word sample
                _, frame = cap.read()
                results = self.tracker.Detect(frame)
                self.tracker.DrawLandmarks(frame, results, face = False)
                screenText = f'STARTING COLLECTION for {word}'
                if fn > 0:
                    screenText = f'{word} - vid#{s}'
                    
                cv.putText(
                    frame, 
                    screenText, 
                    (12,25), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0,255,0), 
                    2, 
                    cv.LINE_AA)
                cv.imshow('Capture', frame)

                if fn == 0:
                    cv.waitKey(1000)

                keys = self.tracker.GetKeypointsData(results, noFace = True , posePoints = [11,0,12]) #138
                print(f'saving {word}-{str(s)}-{fn}.npy .....')
                np.save(os.path.join(data_path, word, str(s), f'{fn}.npy'), keys)
                
                #collect reverse
                if collectReverse:
                    revFrame = cv.flip(frame, 1)
                    revResults = self.tracker.Detect(revFrame)
                    revKeys = self.tracker.GetKeypointsData(revResults, noFace = True , posePoints = [11,0,12]) #138
                    print('saving reverse ...')
                    np.save(os.path.join(data_path, word, f'{s}_r', f'{fn}_r.npy'), revKeys)
                             
                if cv.waitKey(10) & 0xFF == ord('q'):
                        break
        cap.release()
        cv.destroyAllWindows()
        
     
        
