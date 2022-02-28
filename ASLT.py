import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

from MediaPipeHelper import Tracker #MediaPipeHelper module should be in the same directory

'''
------------------------------------------------------------
This module focuses on using MediaPipe to classify hand-shoulder gestures.
A beginner's attempt to make a live American Sign Language (ASL) Translator.

What should be in the same directory:
1. your_model.h5
2. word_list.txt

This module also contains methods that can Collect specific numpy data, Create & Train
models.

This is one of the first porjects I made while studying 
Artificial Intelligence so I apologize if code is a bit trashy...
------------------------------------------------------------
'''

class ASLTranslator:
    def __init__ (
        self, 
        model_path, 
        word_list_path, 
        interval = 18):
        '''
        You can set the model to None if you don't have saved model.
        In other cases, model and the list of words are required 
        when initializing the object
        
        interval is the number of frames that will be taken into account when predicting
        '''
        if word_list_path:
            self.word_list_path = word_list_path
            with open(self.word_list_path) as file:
                self.words  = [line.rstrip() for line in file]
        if model_path:
            self.model = self._load_model(model_path, interval)
        self.tracker = Tracker() #From MediaPipeHelper module
        
    def _load_model(self, 
        model_path, 
        interval):
        model = self.create_model(len(self.words), interval)
        model.load_weights(model_path)
        return model
    #------------------------------------------------------------
    # Deep learing related tasks
    def create_model(self, 
        no_words,
        interval = 18):
        #Basic sequential model
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(interval,138)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(no_words, activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model
    
    def split_data(self, 
        x_raw, 
        y_raw, 
        validate = True):
        x = np.array(x_raw)
        y = to_categorical(y_raw).astype(int)
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.05)
        
        if validate:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)
            return x_train, x_test, x_val, y_train, y_test, y_val
        return x_train, x_test, y_train, y_test
    
    def train_model(self, 
        model, 
        train_data,
        val_data, 
        epochs = 30, 
        model_name = '', 
        plot = False,
        early_stop = True):

        x = train_data[0]
        y = train_data[1]

        #Save checkpoints
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
            print('checkpoints folder made')
        check_model_path = 'checkpoints/mdl-{epoch:02d}-{val_categorical_accuracy:2f}.hdf5'
        model_check = ModelCheckpoint(check_model_path, 
                                      monitor='val_categorical_accuracy', 
                                      verbose = 1, 
                                      save_best_only = True, 
                                      mode='max')
        cb = [model_check]
        if early_stop: #Will stop if model is not improving after 5 iterations
            cb.append(EarlyStopping(monitor='val_loss', patience = 5, verbose=3))

        history = model.fit(x, y, epochs = epochs, callbacks = cb, validation_data = val_data)
        if plot:
            plt.figure(figsize=(16,10))
            val = plt.plot(history.epoch, 
                    history.history['val_categorical_accuracy'],
                    '--', 
                        label='Val')
            plt.plot(history.epoch, 
                    history.history['categorical_accuracy'], 
                    color=val[0].get_color(), 
                    label='Train')
            plt.plot(history.epoch, 
                    history.history['loss'], 
                    label='loss')
            plt.plot(history.epoch, 
                    history.history['val_loss'], 
                    label='val_loss')
            plt.xlabel('Epochs')
            plt.ylabel('categorical_accuracy')
            plt.legend()
            plt.xlim([0,max(history.epoch)])        
        if model_name != '':
            model.save(model_name)
        return model

    #------------------------------------------------------------
    # Data preparation related tasks
    def collect_word_data(
        self, 
        word, 
        data_path, 
        samp_count = 20, 
        frame_count = 18,
        collect_reverse = True):
        '''
        This method will uses OpenCV to collect specific data for each {word}.

        Modify samp_count to set the number of samples that will be collected.
        Modify samp_count to set the number of franes that will be collected in each sample.
        By dafault, collect_reverse is set to True. This will mirror every frames collected.
        To turn it off, set collect_reverse to False.

        The final results will be a new folder named "{word}". 
        It will contain {samp_count} folders each have {frame_count} .npy files.

        The word_list.txt will also be created if it does not exists.
        The {word} will be appended to word_list.txt
        '''
        
        word_list_path = os.path.join(data_path, 'word_list.txt')
        current_words = []
        if os.path.isfile(word_list_path):
            with open(word_list_path) as word_list: #get list of words
                    current_words = [w.rstrip() for w in word_list]
        with open(os.path.join(data_path, 'word_list.txt'), 'a') as word_list:  #create word_list.txt
            if word not in current_words:
                word_list.write(word + '\n')
            else:
                print(f'Overriding data for {word}')
        
        #Create folders
        word_path = os.path.join(data_path, word)
        if not os.path.isdir(word_path):
            os.makedirs(word_path)
        for s in range(samp_count):
            s_path = os.path.join(word_path, str(s))
            sr_path = os.path.join(word_path, f'{s}_r')
            if not os.path.isdir(s_path):
                os.makedirs(s_path)
            if collect_reverse and not os.path.isdir(sr_path):
                os.makedirs(sr_path)
        
        #Start collecting data
        cap = cv.VideoCapture(0)
        for s in range(samp_count):
            #Show starting frame
            if s == 0:
                _, frame = cap.read()
                cv.imshow('Capture', frame) 
                cv.waitKey(2000)
            #Collect each frame data for word sample  
            for fn in range(frame_count): 
                _, frame = cap.read()
                results = self.tracker.Detect(frame)
                self.tracker.draw_landmarks(frame, results, face = False)
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

                #Initial 1 second pause on each sample collection
                if fn == 0:
                    cv.waitKey(1000)

                keys = self.tracker.get_keypoints(results, pose = True, pose_positions = [11,0,12]) #138 points will be collected
                print(f'saving {word}-{str(s)}-{fn}.npy .....')
                np.save(os.path.join(data_path, word, str(s), f'{fn}.npy'), keys)
                
                #Collect data for mirror of frame
                if collect_reverse:
                    rev_frame = cv.flip(frame, 1)
                    rev_results = self.tracker.Detect(rev_frame)
                    rev_keys = self.tracker.get_keypoints(rev_results, pose = True, pose_positions = [11,0,12]) 
                    print('saving reverse ...')
                    np.save(os.path.join(data_path, word, f'{s}_r', f'{fn}_r.npy'), rev_keys)
                             
                if cv.waitKey(10) & 0xFF == ord('q'):
                        break
        cap.release()
        cv.destroyAllWindows()

    def get_npy_from_directory(self, data_path):
        '''
        This will load the existing numpy data per word.
        Will return a numpy array that contains the sequence of each sample
        and a list of labels that corresponds to that seuqence

        word_list.txt should exists in the data directory
        Data directory should be like the following:
         - word (directory)
         --- sample (directory)
         ------ frame (.npy)
        '''
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
        else:
            print('Error: Path does not exists!')
        return sequences, labels
    #------------------------------------------------------------
    
    def start_capture(self, 
        vid = "",
        draw = [True, True, True, True], 
        threshold = 0.75, 
        interval = 18, 
        word_count = 5, 
        display_text = True, 
        color= (0,0,255), 
        size = 1):
        '''
        Use this method to run the whole application
        Leave {vid} blank if you want to capture live footage.
        If not, you can input the video file path.

        {draw} will determine if the mediapipe lines will be displayed in the video.
        pose = draw[0]
        face = draw[1]
        left = draw[2]
        right = draw[3]

        The model will only decide that it predicted something if the resulting
        probabilty is greater or equal to the {threshold}.

        {word_count} How many words will be displayed in the screen
        '''

        #Initial states
        sentence, sequences = [], []
        frame_no = 0

        cap = cv.VideoCapture(0) if vid == "" else cv.VideoCapture(vid)
        print('Start video Capture')
        print('Press Q to exit')

        while cap.isOpened():
            isTrue, frame = cap.read()
            if not isTrue:
                break
                
            #Collect live data per frame
            results = self.tracker.Detect(frame)
            keyPoints = self.tracker.get_keypoints(results, pose = True, pose_positions = [11,0,12])
            
            
            sequences.append(keyPoints) #Form a sequences
            sequences = sequences[1:] if len(sequences) > interval else sequences #Limit the sequences length
            frame_no += 1
            
            #Predict
            if frame_no == interval:
                frame_no = 0
                prediction = self.model.predict(np.expand_dims(sequences, axis=0))[0]
                if prediction[np.argmax(prediction)] >= threshold:
                    word = self.words[np.argmax(prediction)] #Form a sentence
                    if len(sentence) > 0:
                        if word != sentence[-1]:
                            sentence.append(word)
                    else:
                        sentence.append(word)
                if len(sentence) > word_count:
                    sentence = sentence[-word_count:]

            #Visualize
            self.tracker.draw_landmarks(frame, results, draw[0], draw[1], draw[2], draw[3])
            if display_text:
                cv.putText(
                    frame, 
                    ' '.join(sentence), 
                    (10,50), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    size, 
                    color, 
                    2, 
                    cv.LINE_AA)
            cv.imshow('Capture', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
        
    
        
     
        
