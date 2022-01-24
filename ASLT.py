import cv2 as cv
import numpy as np
import mediapipe as mp

class ASLT: #ASL Translator
    def __init__ (self):
        #keypoints, models and stuff
        self.holistic = mp.solutions.holistic
        self.drawing = mp.solutions.drawing_utils
        
    def run(self, 
            model, 
            actions,
            isDraw = [True, True, True], threshold = 0):
        cap = cv.VideoCapture(0)
        sequence = []
        sentence = []
        if threshold == 0:
            threshold = ((1/(len(actions))) * (len(actions) - 1))*0.9
        pred = np.zeros(1*3)
        with self.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic_model:
            while cap.isOpened():
                isTrue, frame = cap.read()
                results = self.detect(frame, holistic_model) #get landmarks in frame
                self.drawLandmarks(frame, results, isDraw[0], isDraw[1], isDraw[2])               
                keyPoints = self.extractKeypoints(results)
                sequence.insert(0, keyPoints)
                sequence = sequence[:30] #limit to 30
                
                #predict
                if len(sequence) == 30:
                    pred = model.predict(np.expand_dims(sequence, axis=0))[0]
                    
                #VISUALIZE
                if pred[np.argmax(pred)] > threshold:
                    action = actions[np.argmax(pred)]
                    if len(sentence) > 0:
                        if action != sentence[-1]:
                            sentence.append(action)
                    else:
                        sentence.append(action)

                if len(sentence) > 5:
                    sentence = sentence[-5:]
                    
                cv.putText(frame,
                           ' '.join(sentence),
                           (10,50),
                           cv.FONT_HERSHEY_SIMPLEX,
                           1,
                           (0, 0,255),
                           2,
                           cv.LINE_AA)
                
                cv.imshow('ASL Translate', frame)
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv.destroyAllWindows()      
    
    def detect(self, frame, model):
        return model.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    
    def drawLandmarks(self,
                      frame,
                      results,
                      head = True,
                      pose = True,
                      hands = True):
        if head:
            self.drawing.draw_landmarks(frame,
                                      results.face_landmarks,
                                      self.holistic.FACEMESH_TESSELATION,
                                      self.drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                      self.drawing.DrawingSpec(color=(80, 255, 121), thickness=1, circle_radius=1))
        if pose:
            self.drawing.draw_landmarks(frame, 
                                      results.pose_landmarks, 
                                      self.holistic.POSE_CONNECTIONS)
        if hands:
            self.drawing.draw_landmarks(frame, 
                                      results.left_hand_landmarks, 
                                      self.holistic.HAND_CONNECTIONS)
            self.drawing.draw_landmarks(frame, 
                                      results.right_hand_landmarks, 
                                      self.holistic.HAND_CONNECTIONS)
        
    def extractKeypoints(self,
                          results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])