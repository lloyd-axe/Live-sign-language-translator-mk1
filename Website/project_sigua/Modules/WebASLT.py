import os
import threading
import cv2 as cv
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from Modules.MediaPipeHelper import Tracker

'''
This is just a web version of the ASLT module ;)
'''

class WebASLTranslator: #ASL Translator
    def __init__ (
        self, 
        model, 
        words, 
        interval = 18):
        #Required model input shape will be (x, interval, 138)
        self.words = words
        self.model = self._loadModel(model, interval)
        self.tracker = Tracker()

        #streaming
        self.capture = cv.VideoCapture(0)

    def __del__(self):
        self.capture.release()

    def _loadModel(
        self, 
        model_name, 
        interval):
        model = self.CreateModel(interval)
        model.load_weights(model_name)
        return model
        
    def CreateModel(
        self, 
        interval = 18):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(interval,138)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(self.words), activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

    def _getData(
        self, 
        frame, 
        sequences):
        results = self.tracker.Detect(frame)
        keyPoints = self.tracker.get_keypoints(
                results, 
                pose = True, 
                pose_positions = [11,0,12])
        sequences.append(keyPoints)
        return sequences, results

    def _predict(
        self,
        frame_no, 
        interval, 
        sequences, 
        threshold, 
        sentence):
        if frame_no == interval:
            prediction = self.model.predict(np.expand_dims(sequences, axis=0))[0]
            if prediction[np.argmax(prediction)] >= threshold:
                #form a sentence
                word = self.words[np.argmax(prediction)]
                if len(sentence) > 0:
                    if word != sentence[-1]:
                        sentence.append(word)
                else:
                    sentence.append(word)
        return sentence

    def _runScript(
        self,
        frame,
        frame_no, 
        sentence, 
        sequences,
        interval,
        threshold):

        sequences, results = self._getData(frame, sequences)
        sequences = sequences[1:] if len(sequences) > interval else sequences #limit sequences

        #predict
        sentence = self._predict(
            frame_no, 
            interval, 
            sequences, 
            threshold, 
            sentence)
        return frame, sentence, sequences            

    def Stream(
        self,
        frame_no, 
        sentence, 
        sequences,
        draw = [True, True, True, True], 
        threshold = 0.75, 
        interval = 10):
        #Start video
        self.frame_no = 0
        isCapture, frame = self.capture.read()
        #resize frame 
        frame = cv.resize(frame, (640,480), interpolation = cv.INTER_AREA)

        self.frame, sentence, sequences = self._runScript(
            frame,
            frame_no, 
            sentence, 
            sequences,
            interval,
            threshold)

        _, stream = cv.imencode('.jpg', self.frame)

        return stream.tobytes(), sentence, sequences
        
        