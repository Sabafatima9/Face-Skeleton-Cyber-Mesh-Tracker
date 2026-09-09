<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2563EB,100:14B8A6&height=220&section=header&text=Face%20Skeleton%20%26%20Cyber%20Mesh%20Tracker&fontSize=35&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Real-Time%203D%20Facial%20Landmark%20%26%20Head%20Pose%20Tracker&descAlignY=55&descSize=18" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=2500&pause=500&color=2563EB&center=true&vCenter=true&width=700&lines=%F0%9F%A7%A0+468+3D+Facial+Landmarks;%F0%9F%91%81%EF%B8%8F+Iris+%26+Gaze+Tracking;%F0%9F%93%90+Pitch+%2F+Yaw+%2F+Roll+Head+Pose;%F0%9F%98%8A+Blink+%26+Smile+Detection" alt="Typing SVG" />

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/CV-MediaPipe-00C9A7?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Tkinter](https://img.shields.io/badge/UI-Tkinter-FF4B4B?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-14B8A6?style=for-the-badge)

</div>

---

## 🎯 Overview

**Face Skeleton & Cyber Mesh Tracker** is a real-time desktop Computer Vision app built with **MediaPipe**, **OpenCV**, and **Tkinter**. It maps 468/478 3D facial landmarks onto the live camera feed, renders a futuristic cyber-mesh wireframe over the face, and computes real-time head pose (Pitch, Yaw, Roll) along with blink, smile, and mouth-aspect telemetry — all inside a dark, glowing desktop UI.

---

## 🎬 Demo Video

<div align="center">

https://github.com/user-attachments/assets/your-demo-video-link-here

*(Replace the link above with your uploaded demo video / GIF showing the cyber mesh, iris tracking, and head pose axes in action.)*

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%" valign="top">

### 🧠 Facial Landmark Tracking
- 468 / 478 3D facial landmarks (jawline, lips, eyes, eyebrows, nose, irises)
- **Full Cyber Mesh** — 468-point triangulated tessellation wireframe
- **Facial Contours Only** — clean outlines mode
- **Iris & Gaze Focus** — dedicated eye reticles + pupil tracking rings

</td>
<td width="50%" valign="top">

### 📐 Pose & Telemetry
- 3D Head Pose Estimation via `cv2.solvePnP` (Pitch / Yaw / Roll)
- 3D coordinate vector axes rendered on the nose tip
- Eye Aspect Ratio (EAR) + real-time blink detection
- Mouth Aspect Ratio (MAR) + smile recognition

</td>
</tr>
</table>

### 🎨 Shared UI Features
- **Dual MediaPipe Architecture** — works with modern MediaPipe Tasks (`>=1.0.0`) and legacy Solutions (`<1.0.0`)
- **3 Display Modes** — Camera + Skeleton, Skeleton Only (Dark Void), Raw Camera Stream
- **Color Themes** — Cyberpunk Neon, Matrix Emerald, Electric Sunset, Sci-Fi Cyan
- **Live Badges** — FPS counter, detected faces count, camera status
- **Interactive Toggles** — landmark number labels, selfie mirror mode, joint metrics
- **Action Controls** — ⏸️ Pause/Resume, 📸 Snapshot to `snapshots/`, 🔄 Switch camera index

---

## 🛠️ Tech Stack

<div align="center">
<img src="https://skillicons.dev/icons?i=python,opencv" />
</div>

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Landmark Detection | Google MediaPipe (Face Mesh / Face Landmarker) |
| Video & Rendering | OpenCV |
| Desktop UI | Tkinter (Dark Theme) |
| Math | NumPy (solvePnP head pose) |

---

## 📦 Requirements & Installation

**1. Activate your virtual environment**
```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```
*(or `pip install -r requirement.txt`)*

---

## ▶️ Running the Application

```bash
python face.py
```

> On launch, select your camera index and preferred theme/display mode from the Tkinter sidebar.

---

## 🏗️ Project Structure

```
Face_Skeleton_Tracker/
├── face.py                # 🎮 Entry point — Tkinter UI + face mesh pipeline
├── assets/                 # Demo media / icons
├── snapshots/               # 📸 Saved snapshots
├── requirements.txt          # Dependencies
└── README.md                # 📖 You're here
```

---

## 🧠 How It Works

| Concept | Implementation |
|---|---|
| **Facial Landmarks** | MediaPipe Face Mesh detects 468/478 3D points per frame |
| **Cyber Mesh Rendering** | Triangulated tessellation drawn over OpenCV frame with glow effects |
| **Head Pose** | `cv2.solvePnP` maps 2D landmarks to a 3D face model to compute Pitch/Yaw/Roll |
| **Blink / Smile Detection** | EAR and MAR ratios thresholded frame-by-frame |
| **Theming & Modes** | Tkinter sidebar toggles control OpenCV draw routines in real time |

---

## 📈 Roadmap

- [ ] Multi-face simultaneous tracking overlay
- [ ] Emotion/expression classification
- [ ] Video file input mode (not just live webcam)
- [ ] Export facial telemetry as CSV

---

<div align="center">

### ⭐ Star this repo if the Face Tracker impressed you!

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:14B8A6,100:2563EB&height=120&section=footer"/>

</div>
