import cv2 as cv
import numpy as np
import mediapipe as mp
'''
------------------------------------------------------------
This module contains methods that implements the commonly used
task when using mediapipe.

1. Dectection
2. Key points collection
3. Drawing
------------------------------------------------------------
'''


class Tracker:
    def __init__(self, 
        detect_conf = 0.5, 
        tracking_conf = 0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        #Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic_model = self.mp_holistic.Holistic(
            min_detection_confidence = detect_conf, 
            min_tracking_confidence = tracking_conf)
        
    #Detection - returns mediapipe holistic model landmarks
    def Detect(self, 
        frame, 
        color = cv.COLOR_BGR2RGB):
        return self.holistic_model.process(cv.cvtColor(frame, color))
    
    #Reformat holistic model landmarks into a dictionary for easy access
    def get_positions(self, 
        results):
        landmarks = {'pose': [], 'face': [], 'left': [], 'right': []}
        landmarks['pose'] = [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark] if results.pose_landmarks else np.zeros((33,4))
        landmarks['face'] = [[res.x, res.y, res.z] for res in results.face_landmarks.landmark] if results.face_landmarks else np.zeros((468,3))
        landmarks['left'] = [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else np.zeros((21,3))
        landmarks['right'] = [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else np.zeros((21,3))
        return landmarks
    
    #Returns a flat array of specified keypoints from landmarks
    def get_keypoints(self, 
        results, 
        face = False, 
        pose = False, 
        pose_positions = []):
        landmarks = self.get_positions(results)
        if not face:
            landmarks.pop('face')
        if not pose:
            landmarks.pop('pose')
        if pose_positions != []:
            landmarks['pose'] = [landmarks['pose'][p] for p in pose_positions]
        return np.concatenate([np.array(part).flatten() for part in landmarks.values()])
    
    #VISUALS
    def draw_landmarks(
        self, 
        frame, 
        results,
        pose = True, 
        face = True, 
        leftHand = True, 
        rightHand = True, 
        color = ((0,0,255), (255,255,255)),
        thickness = (1,1),
        radius = (1,1)):
        if pose:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                color = color[0], 
                thickness = thickness[0], 
                circle_radius = radius[0]),
                self.mp_drawing.DrawingSpec(
                color = color[1], 
                thickness = thickness[1], 
                circle_radius = radius[1]))
        if face:
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                self.mp_drawing.DrawingSpec(
                color = color[0], 
                thickness = thickness[0], 
                circle_radius = radius[0]),
                self.mp_drawing.DrawingSpec(
                color = color[1], 
                thickness = thickness[1], 
                circle_radius = radius[1]))
        if leftHand:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                color = color[0], 
                thickness = thickness[0], 
                circle_radius = radius[0]),
                self.mp_drawing.DrawingSpec(
                color = color[1], 
                thickness = thickness[1], 
                circle_radius = radius[1]))
        if rightHand:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                color = color[0], 
                thickness = thickness[0], 
                circle_radius = radius[0]),
                self.mp_drawing.DrawingSpec(
                color = color[1], 
                thickness = thickness[1], 
                circle_radius = radius[1]))
            
    def check_position(
        self, 
        frame, 
        positions, 
        size = 10, 
        color = (0,255,0)):
        for pos in positions:
            cv.circle(frame, (pos[0],pos[1]), size, color, cv.FILLED)

            
