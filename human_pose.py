import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import time

def get_frame_landmarks(results):
    size_landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark[:23]]).flatten() if results.pose_landmarks else np.zeros(4*23)
    return size_landmarks

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def get_scaled_landmarks(landmarks, dimenson):
    landmarks_2d = []
    if dimenson == '2d':
        for landmark in landmarks:
            x, y = int(landmark.x*1280), int(landmark.y*720)
            landmarks_2d.append([x, y])
        return landmarks_2d

def draw_landmark_on_image( results, image):
    lmks = results.pose_landmarks.landmark

    pose_landmarks = [lmks[0], lmks[1], lmks[2], lmks[3], lmks[4], lmks[5], lmks[6], lmks[7], lmks[8], lmks[9], lmks[10], lmks[11], lmks[12], lmks[13], lmks[14], lmks[15], lmks[16], lmks[17], lmks[18], lmks[19], lmks[20], lmks[21], lmks[22], lmks[23], lmks[24]]
    pose_landmarks = get_scaled_landmarks(pose_landmarks, '2d')

    cv2.line(image, tuple(pose_landmarks[0]), tuple(pose_landmarks[4]), (0, 144, 255), 2)
    cv2.line(image, tuple(pose_landmarks[0]), tuple(pose_landmarks[1]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[4]), tuple(pose_landmarks[6]), (0, 144, 255), 2)
    cv2.line(image, tuple(pose_landmarks[6]), tuple(pose_landmarks[8]), (0, 144, 255), 2)
    cv2.line(image, tuple(pose_landmarks[1]), tuple(pose_landmarks[3]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[3]), tuple(pose_landmarks[7]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[9]), tuple(pose_landmarks[10]), (255, 255, 255), 2)

    cv2.line(image, tuple(pose_landmarks[11]), tuple(pose_landmarks[12]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[11]), tuple(pose_landmarks[13]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[11]), tuple(pose_landmarks[23]), (255, 205, 0), 2)

    cv2.line(image, tuple(pose_landmarks[24]), tuple(pose_landmarks[23]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[24]), tuple(pose_landmarks[12]), (0, 144, 255), 2)

    cv2.line(image, tuple(pose_landmarks[14]), tuple(pose_landmarks[16]), (0, 144, 255), 2)
    cv2.line(image, tuple(pose_landmarks[14]), tuple(pose_landmarks[12]), (0, 144, 255), 2)

    cv2.line(image, tuple(pose_landmarks[16]), tuple(pose_landmarks[22]), (0, 144, 255), 2)
    cv2.line(image, tuple(pose_landmarks[16]), tuple(pose_landmarks[20]), (0, 144, 255), 2)
    cv2.line(image, tuple(pose_landmarks[16]), tuple(pose_landmarks[18]), (0, 144, 255), 2)
    cv2.line(image, tuple(pose_landmarks[18]), tuple(pose_landmarks[20]), (0, 144, 255), 2)

    cv2.line(image, tuple(pose_landmarks[15]), tuple(pose_landmarks[13]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[15]), tuple(pose_landmarks[21]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[15]), tuple(pose_landmarks[19]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[15]), tuple(pose_landmarks[17]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[19]), tuple(pose_landmarks[17]), (255, 205, 0), 2)


    for lm in pose_landmarks:

        cv2.circle(image, (int(lm[0]), int(lm[1])), 5, (255,255,255), 4)
        cv2.circle(image, (int(lm[0]), int(lm[1])), 4, (255,255,141), -1)

def show_fps_body(image, prev_frame_time):
    new_frame_time = time.time()
    fps = int(1/(new_frame_time-prev_frame_time))
    cv2.putText(image, f"fps: {fps}", (1000, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 2, cv2.LINE_AA)
    return new_frame_time

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print(results)
    if results[0][0] > 0.5:
        label = "cheat"
    else:
        label = "non_cheat"
    return label