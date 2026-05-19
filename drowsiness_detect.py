'''This script detects if a person is drowsy or not,using dlib and eye aspect ratio
calculations. Uses webcam video feed as input.'''

#Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame #For playing sound
import time
import dlib
import cv2

#Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

#Minimum threshold of eye aspect ratio below which alarm is triggerd
EYE_ASPECT_RATIO_THRESHOLD = 0.3

#Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50

#Minimum threshold of mouth aspect ratio above which yawn is detected
MOUTH_ASPECT_RATIO_THRESHOLD = 0.6

#Minimum consecutive frames for which mouth ratio is above threshold for yawn to be detected
MOUTH_OPEN_CONSEC_FRAMES = 15

#COunts no. of consecutuve frames below threshold value
COUNTER = 0

#Counts consecutive frames for yawn detection
YAWN_COUNTER = 0

#This function calculates and return eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear


#This function calculates and return mouth aspect ratio
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # points 50-58
    B = distance.euclidean(mouth[4], mouth[8])   # points 52-56
    C = distance.euclidean(mouth[0], mouth[6])   # points 48-54

    mar = (A + B) / (2.0 * C)
    return mar


def drowsiness_score(ear, mar, pitch, ear_thresh, mar_thresh):
    score = 0
    if ear is not None and ear < ear_thresh:
        score += 50
    if mar is not None and mar > mar_thresh:
        score += 30
    if pitch is not None and pitch < -10:
        score += 20
    return min(score, 100)

#Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Extract indexes of facial landmarks for the mouth
(mStart, mEnd) = (48, 68)

# 6 canonical 3D face points for head pose estimation
model_points = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip (30)
        (0.0, -330.0, -65.0),     # Chin (8)
        (-225.0, 170.0, -135.0),  # Left eye left corner (36)
        (225.0, 170.0, -135.0),   # Right eye right corner (45)
        (-150.0, -150.0, -125.0), # Left Mouth corner (48)
        (150.0, -150.0, -125.0),  # Right mouth corner (54)
    ],
    dtype=np.float64,
)

#Start webcam video capture
video_capture = cv2.VideoCapture(1)

#Give some time for camera to initialize(not required)
time.sleep(2)

#For basic FPS measurement
_prev_frame_time = time.time()

WINDOW_NAME = 'Video'

#Mouse debug overlay state
_mouse_x = -1
_mouse_y = -1


def _on_mouse(event, x, y, flags, param):
    global _mouse_x, _mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        _mouse_x, _mouse_y = x, y


cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, _on_mouse)

_alarm_playing = False

while(True):
    #Read each frame and flip it, and convert to grayscale
    ret, frame = video_capture.read()
    if not ret or frame is None:
        print("Không đọc được camera, thoát...")
        break
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect facial points through detector function
    faces = detector(gray, 0)

    eyeAspectRatio = None
    mar = None
    pitch = None

    #Detect facial points
    for face in faces:

        #Draw a rectangle around the dlib face
        cv2.rectangle(
            frame,
            (face.left(), face.top()),
            (face.right(), face.bottom()),
            (255, 0, 0),
            2,
        )

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #Calculate aspect ratio of both eyes
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        # Mouth aspect ratio (yawn detection)
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)

        if mar > MOUTH_ASPECT_RATIO_THRESHOLD:
            YAWN_COUNTER += 1
            if YAWN_COUNTER >= MOUTH_OPEN_CONSEC_FRAMES:
                cv2.putText(
                    frame,
                    "NGAP!",
                    (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    2,
                )
        else:
            YAWN_COUNTER = 0

        # Head pose estimation (pitch/yaw)
        h, w = frame.shape[:2]
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        image_points = np.array(
            [
                shape[30],
                shape[8],
                shape[36],
                shape[45],
                shape[48],
                shape[54],
            ],
            dtype=np.float64,
        )

        try:
            solvepnp_result = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                np.zeros((4, 1)),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if isinstance(solvepnp_result, tuple) and len(solvepnp_result) >= 3:
                success, rvec, tvec = solvepnp_result[:3]
            else:
                success, rvec, tvec = False, None, None

            if success:
                rmat, _ = cv2.Rodrigues(rvec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                pitch = float(angles[0])
                yaw = float(angles[1])

                if pitch < -10:
                    cv2.putText(
                        frame,
                        "GAT DAU!",
                        (10, 350),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 165, 255),
                        2,
                    )
        except cv2.error:
            pitch = None

        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Detect if eye aspect ratio is less than threshold
        if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
            COUNTER += 1
            #If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                if not _alarm_playing:
                    pygame.mixer.music.play(-1)
                    _alarm_playing = True
                cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        else:
            if _alarm_playing:
                pygame.mixer.music.stop()
                _alarm_playing = False
            COUNTER = 0

    else:
        # Chạy khi faces rỗng (không detect được mặt)
        if len(faces) == 0:
            COUNTER = 0
            YAWN_COUNTER = 0
            if _alarm_playing:
                pygame.mixer.music.stop()
                _alarm_playing = False

    #Overlay metrics (always show, even if no face is detected)
    ear_text = "--" if eyeAspectRatio is None else f"{eyeAspectRatio:.3f}"
    cv2.putText(
        frame,
        f"EAR: {ear_text}  TH: {EYE_ASPECT_RATIO_THRESHOLD:.2f}  CNT: {COUNTER}/{EYE_ASPECT_RATIO_CONSEC_FRAMES}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # Drowsiness dashboard (0-100)
    score = drowsiness_score(
        eyeAspectRatio,
        mar,
        pitch,
        EYE_ASPECT_RATIO_THRESHOLD,
        MOUTH_ASPECT_RATIO_THRESHOLD,
    )
    color = (0, 255, 0) if score < 40 else (0, 165, 255) if score < 70 else (0, 0, 255)
    cv2.putText(
        frame,
        f"Buon ngu: {score}%",
        (10, 395),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
    cv2.rectangle(frame, (10, 400), (10 + int(score) * 3, 420), color, -1)
    cv2.putText(
        frame,
        f"Faces (dlib): {len(faces)}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    #Mouse debug overlay similar to "x,y,R,G,B" style
    if 0 <= _mouse_x < frame.shape[1] and 0 <= _mouse_y < frame.shape[0]:
        b, g, r = frame[_mouse_y, _mouse_x]
        cv2.putText(
            frame,
            f"x={_mouse_x}, y={_mouse_y}  R:{int(r)} G:{int(g)} B:{int(b)}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    #FPS overlay (outside face loop)
    _now = time.time()
    _dt = _now - _prev_frame_time
    _prev_frame_time = _now
    if _dt > 0:
        fps = 1.0 / _dt
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    #Show video feed
    cv2.imshow(WINDOW_NAME, frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

#Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()
