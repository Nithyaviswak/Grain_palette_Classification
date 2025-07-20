## 🌾 Overview
- **GrainPalette** is a web-based application that classifies rice grain types using **deep learning**.
- Built with **Flask** and **MobileNetV2**, it leverages **transfer learning** to fine-tune a pre-trained model for rice grain images.

---

## 🧠 Technical Highlights

### 🔍 Transfer Learning
- Uses **MobileNetV2**, a lightweight CNN pre-trained on ImageNet.
- Fine-tuned on rice grain images to adapt to the specific classification task.
- Reduces training time and improves accuracy with limited data.

### 🛠️ Architecture & Tools
- **Frameworks**: Python, Flask, TensorFlow/Keras.
- **Model File**: `rice.h5` – stores the trained model.
- **Training Script**: `train.py` – handles model training and evaluation.
- **Visualization**: Includes `accuracy.png` and `loss.png` to track model performance.

---

## 📈 Features & Functionality
- Users upload rice grain images via the web interface.
- The model predicts the rice variety (e.g., Basmati, Jasmine, etc.).
- Results are displayed with confidence scores and visual feedback.

---

## 🌱 Applications
- **Agriculture**: Helps farmers and agronomists identify rice types for crop planning.
- **Education**: Useful for teaching machine learning in agricultural contexts.
- **Quality Control**: Supports non-destructive testing of rice grains.

---

## 📚 Supporting Resources
You can explore the full project, including code, documentation, and demo files on [GitHub](https://github.com/Pujitha1407/Grainpalette---a-deep-learning-odyssey-in-rice-type-classification-through-transfer-learning). It also includes:
- Project Report
- Project PPT
- HTML templates for results and history
- Training and evaluation plots

---

Would you like me to help you turn this into a study guide or presentation? I can also explain how transfer learning works in more detail if you're curious!
