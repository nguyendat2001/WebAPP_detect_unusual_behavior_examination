from flask import Flask, redirect, url_for, render_template, request, flash, Response
import cv2
import time
import mediapipe as mp
import tensorflow as tf
import threading
import numpy as np
import os

from face_mesh import  face_detect, show_fps_face
from human_pose import mediapipe_detection, draw_landmark_on_image, get_frame_landmarks, draw_class_on_image, show_fps_body


app=Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


camera1 = cv2.VideoCapture(0)
iw, ih =  720, 580

def get_mouth_movement(upper_mouth, bottom_mouth):
    distance = abs(int(upper_mouth.y * ih) - int(bottom_mouth.y * ih))
    if distance > 8:
        return 1
    else:
        return 0

def warning(warnings):
    count = 0
    for warning in warnings:
        if warning == 1:
            count = count + 1
    if count >= 10:
        return "warning: Talking"
    else :
        return ""

def generate_frames1():
    mouth_movements = []
#     iw, ih =  720, 580
    warn_mouth = ""
    re_time = 0
    with mp_face_mesh.FaceMesh(

            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

            while camera1.isOpened():
                success, image = camera1.read()

                if not success:
                    print("Ignoring empty camera frame.")
                  # If loading a video, use 'break' instead of 'continue'.
                    continue

                image = cv2.resize(image,(720,580))

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:

                        lmks = face_landmarks.landmark

                        mouth_movement = get_mouth_movement(lmks[13], lmks[14])
                        mouth_movements.append(mouth_movement)

                        if len(mouth_movements) == 16:
                            warn_mouth = warning(mouth_movements)
                            mouth_movements = []


                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())

                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())

                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())

                # Flip the image horizontally for a selfie-view display.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1)
                if warn_mouth != "":
                    image = cv2.putText(image, str(warn_mouth), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2,cv2.LINE_AA)
                frame = face_detect(image , results)

        #         cv2.imshow('Head Pose Estimation', image)

    #             cv2.imshow('MediaPipe Face Mesh',image)

                ret,buffer=cv2.imencode('.jpg',frame)
                frame = buffer.tobytes()

                yield(b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def predict_pose(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print(results)
    if results[0][0] > 0.5:
        label = "cheat"
    else:
        label = ""
    return label

camera2 = cv2.VideoCapture('test.mp4')


label = "Warmup...."

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
model = tf.keras.models.load_model("./model_sigmoid_no_angles.h5")

def generate_frames2():
    warmup_frames = 100
    n_time_steps = 10
    lm_list = []
    list_image = []
    frame_number = 0
    prev_frame_time = 0
    i = 0

    while True:

        ## read the camera frame
        success,frame=camera2.read()
        frame = cv2.resize(frame,(1280,720))
        if not success:
            break
        else:
            frame.flags.writeable = False
            image, results = mediapipe_detection(frame, pose)

            frame.flags.writeable = True
            i = i + 1
            if i > warmup_frames:
                print("Start detect....")
                if results.pose_landmarks:
                    # Ghi nhận thông số khung xương
                    # Vẽ khung xương lên ảnh
                    draw_landmark_on_image( results, frame)

                    lm = get_frame_landmarks(results)
                    list_image.append(frame)

                    lm_list.append(lm)
                    print(lm)
                    if len(lm_list) == n_time_steps:
                        t1 = threading.Thread(target=predict_pose, args=(model, lm_list,))
                        t1.start()
                        lm_list = []
                        if(label == "cheat"):
                            for t in range (0 ,n_time_steps):
                                cv2.imwrite("./static/save_frame_cheating/"+"frame%d.jpg" % frame_number, list_image[t])
                                frame_number += 1
                        list_image = []
                        print(lm)

                frame = draw_class_on_image(label, frame)
                prev_frame_time = show_fps_body(frame, prev_frame_time)

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')



@app.route('/detect')
def detect():
    return render_template('index_1.html')


picFolder = os.path.join('static', 'save_frame_cheating')
app.config['UPLOAD_FOLDER'] = picFolder


@app.route("/displaycheat")
def displaycheat():
    imageList = os.listdir('static/save_frame_cheating')
    imagelist = ['save_frame_cheating/' + image for image in imageList]
    return render_template("inner-page.html", imagelist=imagelist)


@app.route('/video1')
def video1():
    return Response(generate_frames1(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video2')
def video2():
    return Response(generate_frames2(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=True)