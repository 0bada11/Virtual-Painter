"""
handTrackingModule.py

This module provides a `handDetector` class that uses MediaPipe and OpenCV
to detect hands, extract hand landmarks, and display real-time FPS.

Author: [Obada Mansour]
Date: [7/31/2025]
"""

import cv2 as cv
import mediapipe as mp
import time

class handDetector:
    """
    A class for detecting hands using MediaPipe and OpenCV.

    Attributes:
        mode (bool): Whether to treat the input images as a batch of static images or a video stream.
        maxHands (int): Maximum number of hands to detect.
        minDetectionCon (float): Minimum detection confidence threshold.
        minTrackingCon (float): Minimum tracking confidence threshold.
    """

    def __init__(self, mode=False, maxHands=2, minDetectionCon=0.5, minTrackingCon=0.5):
        """
        Initializes the handDetector with the given parameters.

        Args:
            mode (bool): Static image mode or video stream.
            maxHands (int): Max number of hands to detect.
            minDetectionCon (float): Minimum detection confidence.
            minTrackingCon (float): Minimum tracking confidence.
        """
        # Attributes for .Hands Method
        self.mode = mode
        self.maxHands = maxHands
        self.minDetectionCon = minDetectionCon
        self.minTrackingCon = minTrackingCon

        # Attributes for prepare Drawing + Hand object
        self.results = None
        self.mpDraw = mp.solutions.drawing_utils # type: ignore
        self.mpHands = mp.solutions.hands   # type: ignore
        self.hands = self.mpHands.Hands( 
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackingCon
        )

        # Attributes for showing FPS
        self.pTime = 0
        self.cTime = 0
        # for fingersUp
        self.tipsId = [4, 8, 12, 16, 20]

    def findHands(self, frame, draw=True):
        """
        Detects hands in the given frame and optionally draws landmarks.

        Args:
            frame (np.ndarray): BGR frame from webcam.
            draw (bool): Whether to draw the hand landmarks.

        Returns:
            np.ndarray: The frame with or without landmarks drawn.
        """
        # convert frame from BGR to RGB
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Detecting Hands in the frame
        self.results = self.hands.process(frameRGB)

        # if there are landmarks :
        if self.results.multi_hand_landmarks: 
            # for every landmark in the result array
            for handLms in self.results.multi_hand_landmarks: 
                # draw the landmark
                if draw: 
                    self.mpDraw.draw_landmarks(
                        frame, 
                        handLms, 
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),  # landmark dots
                        self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2) )      #  lines

        return frame

    def findPosition(self, frame, handNo=0, draw=False):
        """
        Finds positions (landmark coordinates) of the detected hand.

        Args:
            frame (np.ndarray): The frame in which hands were detected.
            handNo (int): Index of the hand to analyze (default: 0).
            draw (bool): Whether to draw circles on landmarks.

        Returns:
            list: A list of lists, each containing [id, x, y] for a landmark.
        """
        lmList = []
        # if there are landmarks :
        if self.results.multi_hand_landmarks: # type: ignore
            myHand = self.results.multi_hand_landmarks[handNo] # type: ignore
            # setting hand landmarks positions in (x,y,z)
            for id, lm in enumerate(myHand.landmark):
                # converting from (x,y,z) to (X_Pixel , Y_Pixel) 
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # add position to lmList []
                lmList.append([id, cx, cy])
                # if you want to draw circle around landmark
                if draw:
                    cv.circle(frame, (cx, cy), 20, (255, 255, 255), cv.FILLED)

        return lmList

    def fingersUp(self,lmList):
        """
            Check which fingers are raised.

            Args:
                lmList (list): List of hand landmarks, where each item is [id, x, y].

            Returns:
                list: Boolean values for each finger [Thumb, Index, Middle, Ring, Pinky].
                      True  → finger is up
                      False → finger is down

            Note:
                - Thumb is checked by x position.
                - Other fingers are checked by y position.
            """
        fingers = []
        # working with Thumb
        if lmList[self.tipsId[0]][1] < lmList[self.tipsId[0] - 1][1]:
            fingers.append(True)
        else:
            fingers.append(False)

        # working for 4 fingers
        # lmList[finger_index][0,1,2]
        for id in range(1, 5):
            if lmList[self.tipsId[id]][2] < lmList[self.tipsId[id] - 2][2]:
                fingers.append(True)
            else:
                fingers.append(False)
        return fingers

    def showFPS(self, frame, org =(10,50), fontScale=2  , color=(255, 255, 255),thickness=4 ):
        """
        Calculates and displays FPS on the frame.

        Args:
            frame (np.ndarray): The current video frame.

        Returns:
            np.ndarray: The frame with FPS displayed.
        """
        # Calculate FPS 
        self.cTime = time.time()
        fps = 1 / (self.cTime - self.pTime) if (self.cTime - self.pTime) > 0 else 0
        self.pTime = self.cTime
        # return frame with FPS
        cv.putText(frame, f"FPS : {str(int(fps))}", org,
                   cv.FONT_HERSHEY_PLAIN, fontScale, color, thickness)
        return frame
