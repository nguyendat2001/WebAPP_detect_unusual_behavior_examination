import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf

from human_pose import mediapipe_detection, draw_landmark_on_image, get_frame_landmarks, detect, draw_class_on_image, show_fps
from face_mesh import  face_mesh_detection

# face mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
iw, ih = 1280, 720
re_time = 0

