# 🎯 Gender and Age Detection using Deep Learning

![GitHub License](https://img.shields.io/github/license/smahesh29/Gender-and-Age-Detection)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-CV2-orange)

---

## 🧠 Objective

To build an **AI-powered age and gender detector** that predicts the **exact age** and **gender** of a person from a single image or webcam feed using deep learning.

---

## 📖 About the Project

This project uses a **Convolutional Neural Network (CNN)** to estimate the **gender** (`Male` or `Female`) and **exact age** of a person based on facial features.

The model is built using **OpenCV’s deep learning module (cv2.dnn)** and is trained on the **Adience dataset**.  
Unlike traditional age group classification (e.g., 0–2, 4–6, etc.), this version predicts the **approximate numeric age**.

> ⚡ This model demonstrates how deep learning can analyze facial patterns to infer demographic attributes.

---

## 📦 Dataset

- Dataset: [Adience Benchmark Dataset](https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification)
- Contains **26,580 images** of **2,284 subjects**.
- Includes faces under diverse real-world conditions like **lighting**, **pose**, **makeup**, and **backgrounds**.
- Collected from **Flickr albums** under the **Creative Commons (CC) license**.

---

## 🛠️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/Gender-and-Age-Detection.git
cd Gender-and-Age-Detection
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate  # (Windows)
source venv/bin/activate  # (Mac/Linux)
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, use:

```bash
pip install opencv-python argparse numpy
```

---

## 🧩 Project Structure

```
📂 Gender-and-Age-Detection
 ┣ 📜 detect.py
 ┣ 📜 age_deploy.prototxt
 ┣ 📜 age_net.caffemodel
 ┣ 📜 gender_deploy.prototxt
 ┣ 📜 gender_net.caffemodel
 ┣ 📜 opencv_face_detector.pbtxt
 ┣ 📜 opencv_face_detector_uint8.pb
 ┣ 🖼️ sample1.jpg
 ┣ 🖼️ sample2.jpg
 ┗ 📜 README.md
```

---

## 🚀 Usage

### 🖼️ Detect Gender and Age from Image

```bash
python detect.py --image your_image.jpg
```

> The image should be in the same folder as your `detect.py` file.

### 🎥 Detect Gender and Age in Real-Time (Webcam)

```bash
python detect.py
```

Press `Ctrl + C` to stop execution.

---

## ⚙️ Model Details

| Model File | Description |
|-------------|-------------|
| `opencv_face_detector_uint8.pb` | Pre-trained TensorFlow model for face detection |
| `opencv_face_detector.pbtxt` | Configuration file for the face detector |
| `age_deploy.prototxt` | Model architecture for age estimation |
| `age_net.caffemodel` | Trained model weights for age estimation |
| `gender_deploy.prototxt` | Model architecture for gender classification |
| `gender_net.caffemodel` | Trained model weights for gender classification |

---

## 🧑‍💻 Example Output

| Input Image | Output |
|--------------|---------|
| 🧒 `person1.jpg` | 👦 Male, Age: 21 |
| 👩 `person2.jpg` | 👩 Female, Age: 34 |

---

## 🧰 Dependencies

- Python 3.8+
- OpenCV
- NumPy
- argparse

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 💡 Author

**Aditya Singh Pathania**  
📧 [GitHub Profile](https://github.com/AdityaSinghPathania)  
🧑‍💻 Pursuing B.Tech CSE | Passionate about Deep Learning and AI Vision Systems

---

⭐ If you like this project, consider giving it a star on GitHub!

