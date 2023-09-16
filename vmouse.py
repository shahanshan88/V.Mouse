import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Enum for gesture labels
class Gest:
    PALM = 0
    FIST = 1
    INDEX = 2
    PINCH_MAJOR = 3
    PINCH_MINOR = 4
    MID = 5
    

# Enum for hand labels
class HLabel:
    MAJOR = "Major"
    MINOR = "Minor"

# Hand Recognition class
class HandRecog:
    
    def _init_(self, hand_label):
        self.hand_label = hand_label
        self.hand_result = None
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.finger = 0
    
    # Update the hand result with the latest hand landmarks
    def update_hand_result(self, hand_landmarks):
        self.hand_result = hand_landmarks
    
    # Calculate the Euclidean distance between two landmarks
    def get_dist(self, landmarks):
        x1, y1, z1 = self.hand_result.landmark[landmarks[0]].x, self.hand_result.landmark[landmarks[0]].y, self.hand_result.landmark[landmarks[0]].z
        x2, y2, z2 = self.hand_result.landmark[landmarks[1]].x, self.hand_result.landmark[landmarks[1]].y, self.hand_result.landmark[landmarks[1]].z
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
    
    # Set the finger state based on the hand landmarks
    def set_finger_state(self):
        if self.hand_result.landmark[mp_hands.HandLandmark.THUMB_TIP].x < self.hand_result.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x:
            self.finger += 16
        if self.hand_result.landmark[mp_hands.HandLandmark.PINKY_TIP].y > self.hand_result.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y:
            self.finger += 8
        if self.hand_result.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > self.hand_result.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y:
            self.finger += 4
        if self.hand_result.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > self.hand_result.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y:
            self.finger += 2
        if self.hand_result.landmark[mp_hands.HandLandmark.THUMB_TIP].y > self.hand_result.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y:
            self.finger += 1
    
    # This will return the stable gesture if we are detecting the same gesture
    # for last 5 frames.
    def get_stable_gesture(self):
        if self.ori_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        if self.frame_count >= 5:
            return self.ori_gesture
        else:
            return None
    
    # Recognize the hand gesture
    def get_gesture(self):
        if self.hand_result is None:
            return Gest.PALM
        
        # Set finger state
        self.set_finger_state()
        
        # Calculate the distance between landmarks
        pinch_dist = self.get_dist([mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP])
        index_palm_dist = self.get_dist([mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP])
        mid_last3_dist = self.get_dist([mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP])
        
        # Determine the gesture based on finger state and distances
        if self.finger == 0:
            return Gest.PALM
        elif self.finger == 17:
            return Gest.FIST
        elif pinch_dist < 0.02:
            if index_palm_dist < 0.06:
                return Gest.PINCH_MAJOR
            else:
                return Gest.PINCH_MINOR
        elif pinch_dist > 0.05:
            if mid_last3_dist < 0.04:
                return Gest.MID
            else:
                return Gest.LAST3
        elif self.finger == 16:
            return Gest.INDEX
        elif self.finger == 1:
            return Gest.PINKY
        else:
            return Gest.V_GEST

# Gesture Controller class
class Controller:
    
    def _init_(self):
        self.hand_recog = HandRecog(HLabel.MAJOR)
        self.hand_label = HLabel.MAJOR
    
    # Perform left-click
    def left_click(self):
        pyautogui.click(button='left')
    
    # Perform right-click
    def right_click(self):
        pyautogui.click(button='right')
    
    # Perform double-click
    def double_click(self):
        pyautogui.doubleClick()
    
    # Move the cursor to the specified coordinates
    def move_cursor(self, x, y):
        screen_width, screen_height = pyautogui.size()
        target_x = int(x * screen_width)
        target_y = int(y * screen_height)
        pyautogui.moveTo(target_x, target_y)
    
    # Run the gesture controller
    def run(self):
        cap = cv2.VideoCapture(0)
        with mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.hand_recog.update_hand_result(hand_landmarks)
                        gesture = self.hand_recog.get_gesture()
                        mp_drawing.draw_landmarks(
                            frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        if gesture == Gest.INDEX:
                            self.left_click()
                        elif gesture == Gest.PINKY:
                            self.right_click()
                        elif gesture == Gest.FIST:
                            self.double_click()
                        elif gesture == Gest.V_GEST:
                            self.hand_label = HLabel.MINOR if self.hand_label == HLabel.MAJOR else HLabel.MAJOR
                        elif gesture == Gest.PALM:
                            stable_gesture = self.hand_recog.get_stable_gesture()
                            if stable_gesture == Gest.MID:
                                pyautogui.scroll(1)
                            elif stable_gesture == Gest.LAST3:
                                pyautogui.scroll(-1)
                cv2.imshow('Gesture Controller', frame_rgb)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()

# Create the controller instance and run the gesture controller
controller = Controller()
controller.run()