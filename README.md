# violence-detection-using-deep-learning-for-real-time-HAR
This repository contains the implementation of "Video Surveillance System Using Deep Learning for Real-Time Human Action Recognition (HAR)". The system is designed for real-time violence detection using deep learning techniques, integrating YOLO for object detection and GRU for temporal sequence modeling.

 Overview
This project implements an AI-powered surveillance system for real-time Human Action Recognition (HAR), specifically focusing on violence detection. Using YOLO for spatial feature extraction and Gated Recurrent Units (GRU) for temporal sequence modeling, the system accurately detects violent actions in live video streams or uploaded footage.

🔹 Technologies Used: YOLO, GRU, OpenCV, Django, Twilio API
🔹 Use Cases: Public surveillance, workplace security, law enforcement
🔹 Output: Generates alerts upon detecting violent activities

Features
✔️ Real-time violence detection from video streams
✔️ YOLO for object detection to analyze human interactions
✔️ GRU for temporal sequence modeling to track motion patterns
✔️ Attention mechanism to focus on critical frames
✔️ Automated alerts via Twilio API (SMS & calls to security personnel)
✔️ Django-based UI for uploading & monitoring videos
✔️ Optimized deep learning models for fast inference

 How It Works?
1️⃣ The system continuously processes video streams or uploaded footage.
2️⃣ YOLO extracts spatial features to identify human interactions.
3️⃣ GRU models sequential movements to detect violent actions.
4️⃣ If violence is detected, an alert is sent via Twilio (SMS/Call).
5️⃣ The results are displayed in the Django-based dashboard.

🤝 Contributors
👨‍💻 Ranjith N, Shashank M Gowda, Sujatha P V, Umesh Khanal
📌 Under the guidance of Ms. Vijayalaxmi (Cambridge Institute of Technology, Bangalore)
