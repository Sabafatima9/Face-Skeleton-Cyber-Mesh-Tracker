"""
Face Skeleton Tracker using MediaPipe and Tkinter.
Detects 468/478 3D facial landmarks in real time and renders an interactive skeleton & cyber mesh overlay.

Compatible with both modern MediaPipe Tasks (mediapipe >= 1.0.0)
and legacy MediaPipe Solutions (mediapipe < 1.0.0).
"""

import os
import sys
import time
import math
import urllib.request
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Enable High-DPI scaling on Windows if available
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# ==============================================================================
# MODEL DEFINITION & DOWNLOAD URL
# ==============================================================================
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_FILENAME = "face_landmarker.task"

# ==============================================================================
# COLOR PALETTES (BGR format for OpenCV drawing)
# ==============================================================================
COLOR_THEMES = {
    "Cyberpunk Neon": {
        "mesh": (90, 60, 20),           # Subtle cyber blue wireframe
        "oval": (255, 230, 0),          # Bright Cyan
        "lips": (255, 0, 180),          # Neon Hot Pink
        "left_eye": (0, 255, 255),      # Electric Yellow
        "right_eye": (0, 255, 255),     # Electric Yellow
        "left_eyebrow": (255, 140, 0),  # Azure Blue
        "right_eyebrow": (255, 140, 0), # Azure Blue
        "nose": (0, 255, 128),          # Neon Mint
        "left_iris": (0, 240, 255),     # Glowing Amber
        "right_iris": (0, 240, 255),    # Glowing Amber
        "joint_halo": (0, 240, 255),
        "joint_core": (255, 255, 255),
        "reticle": (0, 255, 255),       # Targeting HUD
        "badge": "#38bdf8",
    },
    "Matrix Emerald": {
        "mesh": (20, 60, 20),
        "oval": (80, 255, 80),
        "lips": (120, 255, 120),
        "left_eye": (50, 255, 150),
        "right_eye": (50, 255, 150),
        "left_eyebrow": (40, 220, 40),
        "right_eyebrow": (40, 220, 40),
        "nose": (100, 255, 100),
        "left_iris": (0, 255, 0),
        "right_iris": (0, 255, 0),
        "joint_halo": (0, 255, 0),
        "joint_core": (240, 255, 240),
        "reticle": (80, 255, 180),
        "badge": "#10b981",
    },
    "Electric Sunset": {
        "mesh": (25, 35, 80),
        "oval": (0, 140, 255),          # Sunset Orange
        "lips": (80, 80, 255),          # Coral Red
        "left_eye": (0, 215, 255),      # Bright Gold
        "right_eye": (0, 215, 255),     # Bright Gold
        "left_eyebrow": (180, 50, 255), # Rose Magenta
        "right_eyebrow": (180, 50, 255),
        "nose": (0, 180, 255),          # Amber
        "left_iris": (255, 255, 255),   # Pure Light
        "right_iris": (255, 255, 255),
        "joint_halo": (0, 165, 255),
        "joint_core": (255, 255, 255),
        "reticle": (0, 200, 255),
        "badge": "#f59e0b",
    },
    "Sci-Fi Cyan": {
        "mesh": (60, 50, 20),
        "oval": (255, 220, 0),
        "lips": (255, 180, 0),
        "left_eye": (255, 235, 100),
        "right_eye": (255, 235, 100),
        "left_eyebrow": (255, 170, 0),
        "right_eyebrow": (255, 170, 0),
        "nose": (240, 200, 50),
        "left_iris": (255, 255, 150),
        "right_iris": (255, 255, 150),
        "joint_halo": (255, 200, 0),
        "joint_core": (255, 255, 255),
        "reticle": (255, 200, 0),
        "badge": "#06b6d4",
    },
}

# ==============================================================================
# FACIAL CONNECTIONS & TOPOLOGY LOADER
# ==============================================================================
def _load_face_connections():
    """Extracts MediaPipe face connections dynamically across versions."""
    def to_tuples(conn_iterable):
        out = []
        if not conn_iterable:
            return out
        for c in conn_iterable:
            if hasattr(c, "start") and hasattr(c, "end"):
                out.append((int(c.start), int(c.end)))
            elif isinstance(c, (tuple, list)) and len(c) == 2:
                out.append((int(c[0]), int(c[1])))
        return out

    # 1. Try modern mediapipe.tasks API
    try:
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as FLC
        return {
            "oval": to_tuples(FLC.FACE_LANDMARKS_FACE_OVAL),
            "lips": to_tuples(FLC.FACE_LANDMARKS_LIPS),
            "left_eye": to_tuples(FLC.FACE_LANDMARKS_LEFT_EYE),
            "right_eye": to_tuples(FLC.FACE_LANDMARKS_RIGHT_EYE),
            "left_eyebrow": to_tuples(FLC.FACE_LANDMARKS_LEFT_EYEBROW),
            "right_eyebrow": to_tuples(FLC.FACE_LANDMARKS_RIGHT_EYEBROW),
            "left_iris": to_tuples(FLC.FACE_LANDMARKS_LEFT_IRIS),
            "right_iris": to_tuples(FLC.FACE_LANDMARKS_RIGHT_IRIS),
            "nose": to_tuples(getattr(FLC, "FACE_LANDMARKS_NOSE", [])),
            "tesselation": to_tuples(FLC.FACE_LANDMARKS_TESSELATION),
        }
    except Exception:
        pass

    # 2. Try legacy mediapipe.solutions API
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            fm = mp.solutions.face_mesh
            return {
                "oval": to_tuples(fm.FACEMESH_FACE_OVAL),
                "lips": to_tuples(fm.FACEMESH_LIPS),
                "left_eye": to_tuples(fm.FACEMESH_LEFT_EYE),
                "right_eye": to_tuples(fm.FACEMESH_RIGHT_EYE),
                "left_eyebrow": to_tuples(fm.FACEMESH_LEFT_EYEBROW),
                "right_eyebrow": to_tuples(fm.FACEMESH_RIGHT_EYEBROW),
                "left_iris": to_tuples(getattr(fm, "FACEMESH_LEFT_IRIS", getattr(fm, "FACEMESH_IRISES", []))),
                "right_iris": to_tuples(getattr(fm, "FACEMESH_RIGHT_IRIS", [])),
                "nose": to_tuples(getattr(fm, "FACEMESH_NOSE", [])),
                "tesselation": to_tuples(fm.FACEMESH_TESSELATION),
            }
    except Exception:
        pass

    # 3. Minimal Fallback Contours
    return {
        "oval": [(10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10)],
        "lips": [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291)],
        "left_eye": [(263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362)],
        "right_eye": [(33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)],
        "left_eyebrow": [(276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)],
        "right_eyebrow": [(46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)],
        "left_iris": [(474, 475), (475, 476), (476, 477), (477, 474)],
        "right_iris": [(469, 470), (470, 471), (471, 472), (472, 469)],
        "nose": [(168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 1), (1, 19), (19, 94), (94, 2)],
        "tesselation": [],
    }

FACE_CONNECTIONS = _load_face_connections()
FACE_CONTOUR_GROUPS = ["oval", "lips", "left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "nose"]

# Canonical 3D head model landmarks for Head Pose Estimation (PnP)
HEAD_POSE_3D = np.array([
    [0.0, 0.0, 0.0],          # 1: Nose tip
    [0.0, -330.0, -65.0],      # 152: Chin
    [-225.0, 170.0, -135.0],   # 33: Left eye left corner
    [225.0, 170.0, -135.0],    # 263: Right eye right corner
    [-150.0, -150.0, -125.0],  # 61: Left mouth corner
    [150.0, -150.0, -125.0]    # 291: Right mouth corner
], dtype=np.float64)

KEY_POSE_INDICES = [1, 152, 33, 263, 61, 291]


# ==============================================================================
# MEDIAPIPE FACE DETECTOR (Dual compatibility: Tasks API & Solutions API)
# ==============================================================================
class FaceDetector:
    """Wrapper providing unified 468/478-point 3D Face Landmarker across MediaPipe versions."""

    def __init__(self, confidence=0.5, max_faces=2):
        self.confidence = confidence
        self.max_faces = max_faces
        self.use_tasks = False
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        import mediapipe as mp

        # Check if legacy mp.solutions is available
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.use_tasks = False
            self.detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.max_faces,
                refine_landmarks=True,
                min_detection_confidence=self.confidence,
                min_tracking_confidence=self.confidence,
            )
        else:
            # Modern mediapipe.tasks API (v1.0+)
            self.use_tasks = True
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
            if not os.path.exists(model_path):
                print(f"Model file not found. Downloading {MODEL_FILENAME} from Google MediaPipe...")
                urllib.request.urlretrieve(MODEL_URL, model_path)
                print("Model downloaded successfully!")

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=self.max_faces,
                min_face_detection_confidence=self.confidence,
                min_face_presence_confidence=self.confidence,
                min_tracking_confidence=self.confidence,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr):
        """Processes a BGR image and returns detected faces with 3D landmarks & telemetry."""
        import mediapipe as mp
        h, w, _ = frame_bgr.shape
        faces_list = []

        if self.use_tasks:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.detector.detect(mp_image)

            if result and result.face_landmarks:
                for idx, face_lms in enumerate(result.face_landmarks):
                    pts = []
                    for lm in face_lms:
                        pts.append((int(lm.x * w), int(lm.y * h), lm.z))

                    blendshapes = {}
                    if result.face_blendshapes and idx < len(result.face_blendshapes):
                        for b in result.face_blendshapes[idx]:
                            blendshapes[b.category_name] = b.score

                    faces_list.append({
                        "landmarks": pts,
                        "raw_landmarks": face_lms,
                        "blendshapes": blendshapes,
                    })
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self.detector.process(frame_rgb)

            if result and result.multi_face_landmarks:
                for face_lms in result.multi_face_landmarks:
                    pts = []
                    for lm in face_lms.landmark:
                        pts.append((int(lm.x * w), int(lm.y * h), lm.z))

                    faces_list.append({
                        "landmarks": pts,
                        "raw_landmarks": face_lms.landmark,
                        "blendshapes": {},
                    })

        return faces_list


# ==============================================================================
# FACIAL TELEMETRY & 3D HEAD POSE ESTIMATION
# ==============================================================================
def estimate_head_pose(landmarks_2d, img_w, img_h):
    """Estimates head pose (pitch, yaw, roll) using cv2.solvePnP with canonical face model."""
    try:
        pts_2d = np.array([
            landmarks_2d[idx][:2] for idx in KEY_POSE_INDICES
        ], dtype=np.float64)

        focal_length = float(img_w)
        center = (float(img_w) / 2.0, float(img_h) / 2.0)
        cam_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            HEAD_POSE_3D, pts_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = angles[0], angles[1], angles[2]

        # Project 3D coordinate axes from nose tip (Red=X, Green=Y, Blue=Z)
        axis_length = 65.0
        axis_3d = np.array([
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, -axis_length, 0.0],
            [0.0, 0.0, -axis_length]
        ], dtype=np.float64)

        imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, cam_matrix, dist_coeffs)

        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "nose_pt": (int(pts_2d[0][0]), int(pts_2d[0][1])),
            "axis_points": [
                (int(imgpts[0].ravel()[0]), int(imgpts[0].ravel()[1])),
                (int(imgpts[1].ravel()[0]), int(imgpts[1].ravel()[1])),
                (int(imgpts[2].ravel()[0]), int(imgpts[2].ravel()[1])),
                (int(imgpts[3].ravel()[0]), int(imgpts[3].ravel()[1])),
            ]
        }
    except Exception:
        return None


def calculate_facial_metrics(pts, blendshapes=None):
    """Computes Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and expression status."""
    blendshapes = blendshapes or {}

    # Geometric Eye Aspect Ratio (EAR)
    # Left eye: top 159, bottom 145, outer 33, inner 133
    v_left = math.hypot(pts[159][0] - pts[145][0], pts[159][1] - pts[145][1])
    h_left = math.hypot(pts[33][0] - pts[133][0], pts[33][1] - pts[133][1])
    ear_left = v_left / (h_left + 1e-6)

    # Right eye: top 386, bottom 374, outer 362, inner 263
    v_right = math.hypot(pts[386][0] - pts[374][0], pts[386][1] - pts[374][1])
    h_right = math.hypot(pts[362][0] - pts[263][0], pts[362][1] - pts[263][1])
    ear_right = v_right / (h_right + 1e-6)

    # Mouth Aspect Ratio (MAR): top 13, bottom 14, left 61, right 291
    v_mouth = math.hypot(pts[13][0] - pts[14][0], pts[13][1] - pts[14][1])
    h_mouth = math.hypot(pts[61][0] - pts[291][0], pts[61][1] - pts[291][1])
    mar = v_mouth / (h_mouth + 1e-6)

    # Blendshapes or geometric heuristics
    blink_l = blendshapes.get("eyeBlinkLeft", 0.0)
    blink_r = blendshapes.get("eyeBlinkRight", 0.0)
    smile_l = blendshapes.get("mouthSmileLeft", 0.0)
    smile_r = blendshapes.get("mouthSmileRight", 0.0)
    jaw_open = blendshapes.get("jawOpen", 0.0)

    if blendshapes:
        left_blinking = blink_l > 0.45
        right_blinking = blink_r > 0.45
        is_smiling = (smile_l + smile_r) / 2.0 > 0.35
        mouth_open = jaw_open > 0.35
    else:
        left_blinking = ear_left < 0.19
        right_blinking = ear_right < 0.19
        eye_dist = math.hypot(pts[33][0] - pts[263][0], pts[33][1] - pts[263][1])
        smile_ratio = h_mouth / (eye_dist + 1e-6)
        is_smiling = smile_ratio > 0.65
        mouth_open = mar > 0.32

    # Status summary
    if left_blinking and right_blinking:
        expression = "Blinking (Both Eyes)"
    elif left_blinking:
        expression = "Winking (Left Eye)"
    elif right_blinking:
        expression = "Winking (Right Eye)"
    elif is_smiling:
        expression = "Smiling"
    elif mouth_open:
        expression = "Mouth Open"
    else:
        expression = "Neutral Pose"

    return {
        "ear_left": ear_left,
        "ear_right": ear_right,
        "mar": mar,
        "left_blinking": left_blinking,
        "right_blinking": right_blinking,
        "is_smiling": is_smiling,
        "mouth_open": mouth_open,
        "expression": expression,
    }


# ==============================================================================
# FACE SKELETON RENDERER
# ==============================================================================
def draw_face_skeleton(
    canvas_img,
    faces,
    theme_name="Cyberpunk Neon",
    detail_mode="full_mesh",  # "full_mesh", "contours_only", "iris_focus"
    show_ids=False,
    show_pose_axes=True,
    show_reticles=True,
):
    """Renders cybernetic face mesh, anatomical skeleton connections, and 3D HUD telemetry."""
    theme = COLOR_THEMES.get(theme_name, COLOR_THEMES["Cyberpunk Neon"])
    h, w, _ = canvas_img.shape
    telemetry_list = []

    for face_idx, face in enumerate(faces):
        pts = face["landmarks"]
        num_pts = len(pts)
        blendshapes = face.get("blendshapes", {})

        # 1. Draw Full Triangulated Surface Mesh (Tessellation)
        if detail_mode == "full_mesh" and FACE_CONNECTIONS["tesselation"]:
            mesh_color = theme["mesh"]
            for start_idx, end_idx in FACE_CONNECTIONS["tesselation"]:
                if start_idx < num_pts and end_idx < num_pts:
                    x1, y1, _ = pts[start_idx]
                    x2, y2, _ = pts[end_idx]
                    cv2.line(canvas_img, (x1, y1), (x2, y2), mesh_color, 1, cv2.LINE_AA)

        # 2. Draw Major Anatomical Contours (Oval, Eyes, Eyebrows, Lips, Nose)
        if detail_mode in ("full_mesh", "contours_only"):
            for group in FACE_CONTOUR_GROUPS:
                conns = FACE_CONNECTIONS.get(group, [])
                color = theme.get(group, (255, 255, 255))
                thickness = 3 if group in ("oval", "lips") else 2

                for start_idx, end_idx in conns:
                    if start_idx < num_pts and end_idx < num_pts:
                        x1, y1, _ = pts[start_idx]
                        x2, y2, _ = pts[end_idx]
                        # Glowing outer trace
                        cv2.line(canvas_img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
                        # Crisp bright center highlight
                        cv2.line(canvas_img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw Iris Rings & Eye Tracking Reticles (Landmarks 468-477 if available)
        iris_centers = []
        for iris_group, iris_color in [("left_iris", theme["left_iris"]), ("right_iris", theme["right_iris"])]:
            conns = FACE_CONNECTIONS.get(iris_group, [])
            if conns:
                pts_in_iris = []
                for start_idx, end_idx in conns:
                    if start_idx < num_pts and end_idx < num_pts:
                        x1, y1, _ = pts[start_idx]
                        x2, y2, _ = pts[end_idx]
                        cv2.line(canvas_img, (x1, y1), (x2, y2), iris_color, 2, cv2.LINE_AA)
                        pts_in_iris.extend([(x1, y1), (x2, y2)])

                if pts_in_iris:
                    avg_x = int(sum(p[0] for p in pts_in_iris) / len(pts_in_iris))
                    avg_y = int(sum(p[1] for p in pts_in_iris) / len(pts_in_iris))
                    iris_centers.append((avg_x, avg_y, iris_group))

        # Render Futuristic Iris Reticles & Targeting HUD
        if (show_reticles or detail_mode == "iris_focus") and iris_centers:
            reticle_color = theme["reticle"]
            for cx, cy, label in iris_centers:
                # Center target dot
                cv2.circle(canvas_img, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas_img, (cx, cy), 9, reticle_color, 1, cv2.LINE_AA)
                # Outer targeting bracket
                r = 16
                cv2.line(canvas_img, (cx - r, cy), (cx - r // 2, cy), reticle_color, 2, cv2.LINE_AA)
                cv2.line(canvas_img, (cx + r // 2, cy), (cx + r, cy), reticle_color, 2, cv2.LINE_AA)
                cv2.line(canvas_img, (cx, cy - r), (cx, cy - r // 2), reticle_color, 2, cv2.LINE_AA)
                cv2.line(canvas_img, (cx, cy + r // 2), (cx, cy + r), reticle_color, 2, cv2.LINE_AA)

        # 4. Optional Numeric Landmark IDs on key structural nodes
        if show_ids:
            key_nodes = [1, 33, 61, 133, 152, 263, 291, 362, 10, 159, 386]
            for idx in key_nodes:
                if idx < num_pts:
                    x, y, _ = pts[idx]
                    cv2.circle(canvas_img, (x, y), 4, theme["joint_halo"], -1, cv2.LINE_AA)
                    cv2.circle(canvas_img, (x, y), 2, theme["joint_core"], -1, cv2.LINE_AA)
                    cv2.putText(
                        canvas_img,
                        str(idx),
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (240, 245, 255),
                        1,
                        cv2.LINE_AA,
                    )

        # 5. Head Pose Estimation & 3D Vector Axes (Pitch, Yaw, Roll)
        pose_data = estimate_head_pose(pts, w, h)
        if show_pose_axes and pose_data is not None:
            origin = pose_data["axis_points"][0]
            pt_x = pose_data["axis_points"][1]  # Red = X (Right)
            pt_y = pose_data["axis_points"][2]  # Green = Y (Down/Up)
            pt_z = pose_data["axis_points"][3]  # Blue = Z (Forward)

            # Draw 3D axis arrows
            cv2.line(canvas_img, origin, pt_x, (0, 0, 255), 3, cv2.LINE_AA)     # Red
            cv2.line(canvas_img, origin, pt_y, (0, 255, 0), 3, cv2.LINE_AA)     # Green
            cv2.line(canvas_img, origin, pt_z, (255, 100, 0), 3, cv2.LINE_AA)   # Blue
            cv2.circle(canvas_img, origin, 4, (255, 255, 255), -1, cv2.LINE_AA)

        # 6. Compute Real-time Facial Telemetry (Blink, Smile, MAR, EAR)
        metrics = calculate_facial_metrics(pts, blendshapes)
        if pose_data:
            metrics["pitch"] = pose_data["pitch"]
            metrics["yaw"] = pose_data["yaw"]
            metrics["roll"] = pose_data["roll"]
        else:
            metrics["pitch"] = 0.0
            metrics["yaw"] = 0.0
            metrics["roll"] = 0.0

        # Draw Face Badge on forehead
        forehead_idx = 10
        if forehead_idx < num_pts:
            fx, fy, _ = pts[forehead_idx]
            badge_text = f"FACE #{face_idx + 1}"
            cv2.putText(
                canvas_img,
                badge_text,
                (fx - 35, max(fy - 14, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                theme["oval"],
                2,
                cv2.LINE_AA,
            )

        telemetry_list.append(metrics)

    return telemetry_list


# ==============================================================================
# TKINTER APPLICATION
# ==============================================================================
class FaceTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("👤 Face Skeleton Tracker — MediaPipe & Tkinter")
        self.root.geometry("1240x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg="#0f172a")

        # Application State
        self.cap = None
        self.camera_index = 0
        self.is_running = True
        self.fps = 0.0
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_timer = time.time()

        # Settings
        self.view_mode = tk.StringVar(value="overlay")       # "overlay", "skeleton_only", "raw_camera"
        self.detail_mode = tk.StringVar(value="full_mesh")   # "full_mesh", "contours_only", "iris_focus"
        self.theme_name = tk.StringVar(value="Cyberpunk Neon")
        self.mirror_mode = tk.BooleanVar(value=True)
        self.show_pose_axes = tk.BooleanVar(value=True)
        self.show_reticles = tk.BooleanVar(value=True)
        self.show_ids = tk.BooleanVar(value=False)

        # Snapshots directory
        self.snapshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)
        self.latest_processed_frame = None

        # Initialize MediaPipe Detector
        try:
            self.detector = FaceDetector(confidence=0.5, max_faces=2)
        except Exception as e:
            messagebox.showerror("Detector Error", f"Failed to initialize MediaPipe Face Detector:\n{e}")
            sys.exit(1)

        # Build GUI Layout
        self._create_widgets()

        # Initialize Camera
        self._start_camera()

        # Handle Clean Close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start Video Loop
        self.update_video()

    def _create_widgets(self):
        # -------------------------------------------------------------
        # TOP HEADER
        # -------------------------------------------------------------
        header_frame = tk.Frame(self.root, bg="#1e293b", height=70, padx=20, pady=12)
        header_frame.pack(side=tk.TOP, fill=tk.X)

        # Title & Subtitle
        title_box = tk.Frame(header_frame, bg="#1e293b")
        title_box.pack(side=tk.LEFT)

        title_lbl = tk.Label(
            title_box,
            text="👤 FACE SKELETON TRACKER",
            font=("Segoe UI", 16, "bold"),
            fg="#38bdf8",
            bg="#1e293b",
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = tk.Label(
            title_box,
            text="Real-time 468/478-Point 3D Face Mesh & Facial Geometry with MediaPipe & Tkinter",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1e293b",
        )
        subtitle_lbl.pack(anchor="w")

        # Live Badges (Right side of header)
        badges_box = tk.Frame(header_frame, bg="#1e293b")
        badges_box.pack(side=tk.RIGHT)

        self.fps_badge = tk.Label(
            badges_box,
            text="⚡ 0.0 FPS",
            font=("Consolas", 10, "bold"),
            fg="#10b981",
            bg="#0f172a",
            padx=12,
            pady=6,
            relief=tk.FLAT,
        )
        self.fps_badge.pack(side=tk.LEFT, padx=6)

        self.faces_badge = tk.Label(
            badges_box,
            text="👤 0 Faces",
            font=("Segoe UI", 10, "bold"),
            fg="#f59e0b",
            bg="#0f172a",
            padx=12,
            pady=6,
        )
        self.faces_badge.pack(side=tk.LEFT, padx=6)

        self.cam_status_badge = tk.Label(
            badges_box,
            text="🟢 Cam 0 Active",
            font=("Segoe UI", 10, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
            padx=12,
            pady=6,
        )
        self.cam_status_badge.pack(side=tk.LEFT, padx=6)

        # -------------------------------------------------------------
        # MAIN BODY CONTAINER (Left: Video Frame, Right: Sidebar)
        # -------------------------------------------------------------
        body_frame = tk.Frame(self.root, bg="#0f172a", padx=16, pady=16)
        body_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left Column: Video Viewport
        self.video_container = tk.Frame(body_frame, bg="#1e293b", bd=2, relief=tk.GROOVE)
        self.video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 16))

        self.video_label = tk.Label(
            self.video_container,
            text="Connecting to Camera...\nPlease wait.",
            font=("Segoe UI", 14),
            fg="#94a3b8",
            bg="#020617",
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Right Column: Control Sidebar
        sidebar = tk.Frame(body_frame, bg="#1e293b", width=340, padx=16, pady=14)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Helper card creator
        def make_section(parent, text):
            sec = tk.LabelFrame(
                parent,
                text=f"  {text}  ",
                font=("Segoe UI", 10, "bold"),
                fg="#38bdf8",
                bg="#1e293b",
                bd=1,
                relief=tk.SOLID,
                padx=10,
                pady=6,
            )
            sec.pack(fill=tk.X, pady=5)
            return sec

        # --- Section 1: Display Mode ---
        sec_mode = make_section(sidebar, "Display Mode")
        modes = [
            ("Camera + Face Mesh", "overlay"),
            ("Mesh Only (Dark Void)", "skeleton_only"),
            ("Raw Camera Feed", "raw_camera"),
        ]
        for text, val in modes:
            rb = tk.Radiobutton(
                sec_mode,
                text=text,
                variable=self.view_mode,
                value=val,
                font=("Segoe UI", 9),
                fg="#f1f5f9",
                bg="#1e293b",
                activebackground="#1e293b",
                activeforeground="#38bdf8",
                selectcolor="#0f172a",
                cursor="hand2",
            )
            rb.pack(anchor="w", pady=1)

        # --- Section 2: Mesh Detail Level ---
        sec_detail = make_section(sidebar, "Mesh Detail Level")
        detail_modes = [
            ("Full Cyber Mesh (Tessellation)", "full_mesh"),
            ("Facial Contours Only", "contours_only"),
            ("Iris & Gaze Focus", "iris_focus"),
        ]
        for text, val in detail_modes:
            rb = tk.Radiobutton(
                sec_detail,
                text=text,
                variable=self.detail_mode,
                value=val,
                font=("Segoe UI", 9),
                fg="#f1f5f9",
                bg="#1e293b",
                activebackground="#1e293b",
                activeforeground="#38bdf8",
                selectcolor="#0f172a",
                cursor="hand2",
            )
            rb.pack(anchor="w", pady=1)

        # --- Section 3: Color Theme ---
        sec_theme = make_section(sidebar, "Color Theme")
        self.theme_combo = ttk.Combobox(
            sec_theme,
            textvariable=self.theme_name,
            values=list(COLOR_THEMES.keys()),
            state="readonly",
            font=("Segoe UI", 9),
        )
        self.theme_combo.pack(fill=tk.X, pady=4)

        # --- Section 4: Overlay Toggles ---
        sec_toggles = make_section(sidebar, "Visual Toggles")
        toggles = [
            ("🪞 Mirror Camera View", self.mirror_mode),
            ("🎯 3D Head Pose Axes", self.show_pose_axes),
            ("👁️ Iris Tracking Reticles", self.show_reticles),
            ("🔢 Show Landmark IDs", self.show_ids),
        ]
        for text, var in toggles:
            cb = tk.Checkbutton(
                sec_toggles,
                text=text,
                variable=var,
                font=("Segoe UI", 9),
                fg="#f1f5f9",
                bg="#1e293b",
                activebackground="#1e293b",
                activeforeground="#38bdf8",
                selectcolor="#0f172a",
                cursor="hand2",
            )
            cb.pack(anchor="w", pady=1)

        # --- Section 5: Real-time Facial Telemetry ---
        sec_telemetry = make_section(sidebar, "Real-Time Telemetry")

        self.lbl_expression = tk.Label(
            sec_telemetry,
            text="Status: No Face Detected",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#1e293b",
            anchor="w",
        )
        self.lbl_expression.pack(fill=tk.X, pady=1)

        self.lbl_eyes = tk.Label(
            sec_telemetry,
            text="Eyes: Left Open | Right Open",
            font=("Consolas", 8),
            fg="#e2e8f0",
            bg="#1e293b",
            anchor="w",
        )
        self.lbl_eyes.pack(fill=tk.X, pady=1)

        self.lbl_mouth = tk.Label(
            sec_telemetry,
            text="Mouth: Closed (MAR: 0.00)",
            font=("Consolas", 8),
            fg="#e2e8f0",
            bg="#1e293b",
            anchor="w",
        )
        self.lbl_mouth.pack(fill=tk.X, pady=1)

        self.lbl_pose = tk.Label(
            sec_telemetry,
            text="Head Pose: Yaw 0.0° | Pitch 0.0°",
            font=("Consolas", 8),
            fg="#38bdf8",
            bg="#1e293b",
            anchor="w",
        )
        self.lbl_pose.pack(fill=tk.X, pady=1)

        # --- Section 6: Action Controls ---
        sec_actions = make_section(sidebar, "Camera & Snapshots")

        btn_row = tk.Frame(sec_actions, bg="#1e293b")
        btn_row.pack(fill=tk.X, pady=3)

        self.btn_snapshot = tk.Button(
            btn_row,
            text="📸 Snapshot",
            command=self.save_snapshot,
            font=("Segoe UI", 9, "bold"),
            fg="#ffffff",
            bg="#0284c7",
            activebackground="#0369a1",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor="hand2",
        )
        self.btn_snapshot.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_pause = tk.Button(
            btn_row,
            text="⏸️ Pause",
            command=self.toggle_pause,
            font=("Segoe UI", 9, "bold"),
            fg="#ffffff",
            bg="#334155",
            activebackground="#475569",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor="hand2",
        )
        self.btn_pause.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # Camera Switcher
        cam_row = tk.Frame(sec_actions, bg="#1e293b")
        cam_row.pack(fill=tk.X, pady=3)

        cam_lbl = tk.Label(
            cam_row, text="Camera:", font=("Segoe UI", 9), fg="#94a3b8", bg="#1e293b"
        )
        cam_lbl.pack(side=tk.LEFT, padx=(0, 6))

        self.cam_combo = ttk.Combobox(
            cam_row,
            values=["Camera 0", "Camera 1", "Camera 2"],
            state="readonly",
            width=12,
            font=("Segoe UI", 9),
        )
        self.cam_combo.current(0)
        self.cam_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cam_combo.bind("<<ComboboxSelected>>", self.on_camera_change)

        # -------------------------------------------------------------
        # BOTTOM STATUS FOOTER
        # -------------------------------------------------------------
        status_frame = tk.Frame(self.root, bg="#0f172a", height=30, padx=20, pady=4)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_text = tk.Label(
            status_frame,
            text="System initialized. Ready.",
            font=("Segoe UI", 8),
            fg="#64748b",
            bg="#0f172a",
        )
        self.status_text.pack(side=tk.LEFT)

        snap_count = len(os.listdir(self.snapshots_dir)) if os.path.exists(self.snapshots_dir) else 0
        self.lbl_snap_count = tk.Label(
            status_frame,
            text=f"📁 Snapshots: {snap_count}",
            font=("Segoe UI", 8),
            fg="#64748b",
            bg="#0f172a",
        )
        self.lbl_snap_count.pack(side=tk.RIGHT)

    def _start_camera(self):
        """Initializes OpenCV video capture device."""
        if self.cap is not None:
            self.cap.release()

        # On Windows, cv2.CAP_DSHOW provides fastest camera startup
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.camera_index, backend)

        if not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.camera_index)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cam_status_badge.config(
                text=f"🟢 Cam {self.camera_index} Active", fg="#38bdf8"
            )
            self.status_text.config(text=f"Connected to Camera {self.camera_index}.")
        else:
            self.cam_status_badge.config(text="🔴 Camera Disconnected", fg="#ef4444")
            self.status_text.config(text=f"Failed to open Camera {self.camera_index}.")

    def on_camera_change(self, event=None):
        """Handles switching between available camera devices."""
        idx = self.cam_combo.current()
        if idx != self.camera_index:
            self.camera_index = idx
            self.status_text.config(text=f"Switching to Camera {self.camera_index}...")
            self._start_camera()

    def toggle_pause(self):
        """Pauses or resumes live video stream."""
        self.is_running = not self.is_running
        if self.is_running:
            self.btn_pause.config(text="⏸️ Pause", bg="#334155")
            self.cam_status_badge.config(text=f"🟢 Cam {self.camera_index} Active", fg="#38bdf8")
            self.status_text.config(text="Video stream resumed.")
        else:
            self.btn_pause.config(text="▶️ Resume", bg="#10b981")
            self.cam_status_badge.config(text="⏸️ Stream Paused", fg="#f59e0b")
            self.status_text.config(text="Video stream paused.")

    def save_snapshot(self):
        """Saves current processed frame to snapshots/ directory with timestamp."""
        if self.latest_processed_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_skeleton_{timestamp}.png"
            filepath = os.path.join(self.snapshots_dir, filename)
            cv2.imwrite(filepath, self.latest_processed_frame)
            snap_count = len(os.listdir(self.snapshots_dir))
            self.lbl_snap_count.config(text=f"📁 Snapshots: {snap_count}")
            self.status_text.config(text=f"📸 Snapshot saved: {filename}")
        else:
            self.status_text.config(text="No active frame available to snapshot.")

    def update_video(self):
        """Main rendering loop: captures frame, detects face mesh, updates GUI."""
        if self.is_running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 1. Calculate FPS
                curr_time = time.time()
                self.frame_count += 1
                if curr_time - self.fps_timer >= 1.0:
                    self.fps = self.frame_count / (curr_time - self.fps_timer)
                    self.frame_count = 0
                    self.fps_timer = curr_time
                    self.fps_badge.config(text=f"⚡ {self.fps:.1f} FPS")

                # 2. Mirror mode (horizontal flip)
                if self.mirror_mode.get():
                    frame = cv2.flip(frame, 1)

                h, w, _ = frame.shape

                # 3. Detect Face Landmarks
                faces = self.detector.process(frame)
                num_faces = len(faces)

                # 4. Update Face Counter Badge
                if num_faces == 0:
                    self.faces_badge.config(text="👤 0 Faces", fg="#94a3b8")
                    self.lbl_expression.config(text="Status: No Face Detected", fg="#94a3b8")
                    self.lbl_eyes.config(text="Eyes: -- | --")
                    self.lbl_mouth.config(text="Mouth: -- (MAR: 0.00)")
                    self.lbl_pose.config(text="Head Pose: Yaw 0.0° | Pitch 0.0°")
                else:
                    self.faces_badge.config(
                        text=f"👤 {num_faces} Face{'s' if num_faces > 1 else ''}", fg="#10b981"
                    )

                # 5. Canvas Preparation based on Display Mode
                mode = self.view_mode.get()
                if mode == "skeleton_only":
                    # Deep cyberpunk dark void canvas
                    display_frame = np.full((h, w, 3), 18, dtype=np.uint8)
                    # Subtle cyber grid lines
                    for y in range(0, h, 40):
                        cv2.line(display_frame, (0, y), (w, y), (28, 33, 44), 1)
                    for x in range(0, w, 40):
                        cv2.line(display_frame, (x, 0), (x, h), (28, 33, 44), 1)
                else:
                    display_frame = frame.copy()

                # 6. Render Skeleton & Mesh Overlay
                if mode != "raw_camera" and num_faces > 0:
                    telemetry = draw_face_skeleton(
                        display_frame,
                        faces,
                        theme_name=self.theme_name.get(),
                        detail_mode=self.detail_mode.get(),
                        show_ids=self.show_ids.get(),
                        show_pose_axes=self.show_pose_axes.get(),
                        show_reticles=self.show_reticles.get(),
                    )

                    if telemetry:
                        t = telemetry[0]
                        self.lbl_expression.config(text=f"Status: {t['expression']}", fg="#10b981")
                        eye_l_str = "Blink" if t["left_blinking"] else f"Open ({t['ear_left']:.2f})"
                        eye_r_str = "Blink" if t["right_blinking"] else f"Open ({t['ear_right']:.2f})"
                        self.lbl_eyes.config(text=f"Eyes: L {eye_l_str} | R {eye_r_str}")

                        mouth_str = "Smiling" if t["is_smiling"] else ("Open" if t["mouth_open"] else "Closed")
                        self.lbl_mouth.config(text=f"Mouth: {mouth_str} (MAR: {t['mar']:.2f})")

                        self.lbl_pose.config(
                            text=f"Head Pose: Y {t['yaw']:+.1f}° | P {t['pitch']:+.1f}° | R {t['roll']:+.1f}°"
                        )

                self.latest_processed_frame = display_frame.copy()

                # 7. Convert OpenCV BGR to RGB for PIL/Tkinter
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                # Resize proportionally to fit video container
                lbl_w = max(self.video_container.winfo_width(), 640)
                lbl_h = max(self.video_container.winfo_height(), 480)

                scale = min(lbl_w / w, lbl_h / h)
                new_w = max(int(w * scale), 320)
                new_h = max(int(h * scale), 240)

                resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                img = Image.fromarray(resized)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk, text="")

        # Schedule next video frame (~30 FPS)
        self.root.after(25, self.update_video)

    def on_closing(self):
        """Releases camera resources and closes Tkinter application safely."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    root = tk.Tk()
    app = FaceTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
