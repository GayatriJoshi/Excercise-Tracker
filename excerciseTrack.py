import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class LateralRaisesAnalyzer(VideoTransformerBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.lateral_raise_count = 0
        self.correctness = "Unknown"
        self.frame_debounce = 5
        self.frame_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(rgb_frame)

        if results.pose_landmarks:
            points = {}
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                points[id] = (cx, cy)

            # Check if both shoulders and elbows are detected
            if all(i in points for i in [11, 13, 12, 14]):
                left_angle = np.degrees(np.arctan2(points[13][1] - points[11][1], points[13][0] - points[11][0]))
                right_angle = np.degrees(np.arctan2(points[14][1] - points[12][1], points[14][0] - points[12][0]))

                # Check if both arms are raised
                both_arms_up = left_angle < 90 and right_angle < 90

                if both_arms_up:
                    if self.frame_count >= self.frame_debounce:
                        self.lateral_raise_count += 1
                        self.frame_count = 0
                else:
                    self.frame_count += 1


            # Display lateral raise count and correctness on frame
            cv2.putText(img, f"Lateral Raise Count: {self.lateral_raise_count}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            

        return img
    

class HammerCurlsAnalyzer(VideoTransformerBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hammer_curls_count = 0
        self.correctness = "Unknown"
        self.frame_debounce = 5
        self.frame_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
        results = self.mp_pose.process(rgb_frame)

        if results.pose_landmarks:
            points = {}
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                points[id] = (cx, cy)

            # Check if both elbows and hands are detected
            if all(i in points for i in [11, 13, 15, 12, 14, 16]):
               
               left_angle = np.degrees(np.arctan2(points[13][1] - points[15][1], points[13][0] - points[15][0]))
               right_angle = np.degrees(np.arctan2(points[14][1] - points[16][1], points[14][0] - points[16][0]))
               hammer_curls_position = left_angle < 30 and right_angle < 30
               if hammer_curls_position:
                    if self.frame_count >= self.frame_debounce:
                        self.hammer_curls_count += 1
                        self.frame_count = 0
                    else:
                        self.frame_count += 1

        
            # Display hammer curls count and correctness on frame
            cv2.putText(img, f"Hammer Curls Count: {self.hammer_curls_count}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return img




class SquatsAnalyzer(VideoTransformerBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.squats_count = 0
        self.correctness = "Unknown"
        self.frame_debounce = 5
        self.frame_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert frame to RGB format
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe Pose
        results = self.mp_pose.process(rgb_frame)

        if results.pose_landmarks:
            points = {}
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                points[id] = (cx, cy)

            # Check if ankles, knees, and hips are detected
            if all(i in points for i in [23, 25, 27, 24, 26, 28]):
                # Calculate angles between ankles, knees, and hips
                left_knee_angle = calculate_angle(points[23], points[25], points[27])
                right_knee_angle = calculate_angle(points[24], points[26], points[28])

                # Check if knees are bent at least to a certain angle for both legs
                knees_bent = left_knee_angle > 160 and right_knee_angle > 160

                if knees_bent:
                    if self.frame_count >= self.frame_debounce:
                        self.squats_count += 1
                        self.frame_count = 0
                else:
                    self.frame_count += 1

            

            # Display squats count and correctness on frame
            cv2.putText(img, f"Squats Count: {self.squats_count}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            

        return img

def calculate_angle(a, b, c):
    """Calculate the angle between three points"""
    angle = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0]))
    return angle + 360 if angle < 0 else angle



class PushUpsAnalyzer(VideoTransformerBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.push_ups_count = 0
        self.correctness = "Unknown"
        self.frame_debounce = 5
        self.frame_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert frame to RGB format
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe Pose
        results = self.mp_pose.process(rgb_frame)

        if results.pose_landmarks:
            points = {}
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                points[id] = (cx, cy)

            # Check if shoulders, elbows, and wrists are detected
            if all(i in points for i in [11, 13, 15, 12, 14, 16]):
                # Calculate angle between shoulders, elbows, and wrists for left arm
                left_elbow_angle = calculate_angle(points[11], points[13], points[15])
                # Calculate angle between shoulders, elbows, and wrists for right arm
                right_elbow_angle = calculate_angle(points[12], points[14], points[16])

                # Check if arms are in a push-up position
                arms_in_push_up_position = left_elbow_angle > 160 and right_elbow_angle > 160

                if arms_in_push_up_position:
                    if self.frame_count >= self.frame_debounce:
                        self.push_ups_count += 1
                        self.frame_count = 0
                else:
                    self.frame_count += 1

           

            # Display push-ups count and correctness on frame
            cv2.putText(img, f"Push-Ups Count: {self.push_ups_count}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
           

        return img




class BicepCurlsAnalyzer(VideoTransformerBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.bicep_curls_count = 0
        self.correctness = "Unknown"
        self.frame_debounce = 5
        self.frame_count = 0
        self.previous_hand_positions = {"left": None, "right": None}

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert frame to RGB format
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe Pose
        results = self.mp_pose.process(rgb_frame)

        if results.pose_landmarks:
            points = {}
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                points[id] = (cx, cy)

            # Check if elbows, wrists, and hands are detected
            if all(i in points for i in [13, 15, 17, 14, 16, 18]):
                # Get hand positions
                left_hand_pos = points[17]
                right_hand_pos = points[18]

                # Check if both hands are moving upwards (indicative of a bicep curl)
                left_moving_up = self.is_moving_up(left_hand_pos, "left")
                right_moving_up = self.is_moving_up(right_hand_pos, "right")

                if left_moving_up and right_moving_up:
                    if self.frame_count >= self.frame_debounce:
                        self.bicep_curls_count += 1
                        self.frame_count = 0
                else:
                    self.frame_count += 1

            # Display bicep curls count and correctness on frame
            cv2.putText(img, f"Bicep Curls Count: {self.bicep_curls_count}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return img

    def is_moving_up(self, hand_pos, hand):
        """
        Check if the hand is moving upwards by comparing its current position with its previous position.
        """
        if self.previous_hand_positions[hand] is not None:
            if hand_pos[1] < self.previous_hand_positions[hand][1]:
                self.previous_hand_positions[hand] = hand_pos
                return True
        self.previous_hand_positions[hand] = hand_pos
        return False