# 🧠 Face Age & Gender Detection using OpenCV and Deep Learning

<p align="center">
  <img src="https://img.shields.io/github/repo-size/Dev-AdityaPathania/face-age-gender-detection"/>
  <img src="https://img.shields.io/github/license/Dev-AdityaPathania/face-age-gender-detection"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue"/>
  <img src="https://img.shields.io/badge/OpenCV-DNN-green"/>
  <img src="https://img.shields.io/badge/Model-Caffe-orange"/>
</p>

---

<div align="center">

## 🎯 Objective

To build a **real-time age and gender detector** using OpenCV’s deep learning module (`cv2.dnn`) that can estimate the **approximate age and gender** of a person from an image or webcam stream.

---

## 🧩 About the Project

This project detects **faces**, then predicts **gender (Male/Female)** and **exact age (approximation)** using pre-trained Caffe models.

The models were trained by [Tal Hassner and Gil Levi](https://talhassner.github.io/home/projects/Adience/Adience-data.html) on the **Adience Dataset**, which contains over **26,000 facial images** across multiple age ranges under real-world conditions.

> ⚙️ Prediction is based on probability-weighted averages, providing a near-exact age instead of a fixed range.

---

## 🧠 Model Files Used

| Type | Description | File |
|------|--------------|------|
| Face Detection | TensorFlow face detection model | `opencv_face_detector_uint8.pb`, `opencv_face_detector.pbtxt` |
| Age Detection | Caffe model trained on Adience dataset | `age_net.caffemodel`, `age_deploy.prototxt` |
| Gender Detection | Caffe model trained on Adience dataset | `gender_net.caffemodel`, `gender_deploy.prototxt` |

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Dev-AdityaPathania/face-age-gender-detection.git
cd face-age-gender-detection
```

### 2️⃣ Install dependencies
```bash
pip install opencv-python argparse
```

> Optional: For GPU acceleration, install `opencv-contrib-python` instead.

### 3️⃣ Ensure model files exist
Make sure all `.pb`, `.pbtxt`, `.prototxt`, and `.caffemodel` files are in the same directory as `detect.py`.

---

## 🚀 Usage

### 📸 Detect Age & Gender from Image
```bash
python detect.py --image your_image.jpg
```

### 🎥 Real-time Webcam Detection
```bash
python detect.py
```

Press **Q** to quit the webcam window.

---

## 🧍‍♂️ Sample Output

| Input | Output |
|-------|---------|
| <img src="input.jpg" width="300"/> | <img src="output.jpg" width="300"/> |

*Example Output:*  
> Detected: **Male, 26 yrs**

---

## 🧪 Features

✅ Real-time detection using webcam  
✅ Exact age approximation (not age groups)  
✅ Pre-trained DNN models  
✅ Smooth frame processing every 1 second  
✅ Written in clean, modular Python  

---

## 🖼️ Demo Preview

<p align="center">
  <img src="https://github.com/Dev-AdityaPathania/face-age-gender-detection/blob/main/demo.gif?raw=true" width="600" alt="Demo Preview">
</p>

---

## 🧾 License

This project is licensed under the **MIT License** — feel free to use and modify with attribution.

---

## 👨‍💻 Author

**Aditya Pathania**  
📍 GitHub: [@Dev-AdityaPathania](https://github.com/Dev-AdityaPathania)

> Made with ❤️ using Python and OpenCV.

</div>


