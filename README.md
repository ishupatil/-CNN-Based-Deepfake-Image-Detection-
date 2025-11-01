🧠 CNN-Based Deepfake Image Detection
🔍 Overview

This project is a Deepfake Image Detection System built using Convolutional Neural Networks (CNN) and MobileNetV2 architecture. It helps identify whether an image is real or AI-generated (fake) by analyzing subtle visual artifacts.
The system is deployed using Streamlit, offering an intuitive and interactive web interface where users can upload an image and instantly see detection results with confidence levels.

🧩 Features

🖼️ Upload any image (JPG, JPEG, PNG) and analyze it instantly.

🤖 CNN-based model classifies images as Real or Fake.

📈 Displays confidence percentage and descriptive explanation.

💡 Uses MobileNetV2 for efficient and accurate deep learning inference.

🌐 Simple and clean Streamlit web UI.

📊 Optionally displays training performance graphs.

⚙️ Technologies Used & Why
Technology	Purpose	Why It Was Used
Python	Programming Language	High-level, easy to use, and rich ML libraries.
TensorFlow / Keras	Deep Learning Framework	Used for model building, training, and inference. Provides pre-trained CNNs like MobileNetV2.
MobileNetV2	CNN Architecture	Lightweight yet powerful network ideal for fast inference on limited hardware.
OpenCV	Image Processing	Handles image reading, resizing, and conversion efficiently.
NumPy	Numerical Computation	Supports efficient array manipulation and preprocessing.
Streamlit	Web Framework	Enables quick creation of data-science web apps without front-end coding.
Matplotlib / Seaborn (optional)	Visualization	Used to visualize accuracy and loss during training.
Dataset (Custom / Kaggle)	Deepfake vs Real Images	Used for model training and testing.
🧠 Model Information

Architecture: MobileNetV2 (Fine-Tuned)

Input Shape: 224x224x3

Output: Binary classification (Fake / Real)

Activation: sigmoid

Loss Function: binary_crossentropy

Optimizer: Adam

Metrics: accuracy

The trained model is saved as deepfake_detection_model.h5 and is loaded dynamically during app execution.

📦 Project Structure
├── app.py                          # Streamlit Web App
├── deepfake_detection_model.h5     # Trained CNN model
├── dataset_fixed/                  # Dataset used for training
├── training_results.png            # Accuracy & loss visualization
├── coverpage.png                   # Landing image for the app
└── README.md                       # Project documentation

🚀 How to Run the Project Locally
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/CNN-Deepfake-Detection.git
cd CNN-Deepfake-Detection

2️⃣ Install Dependencies
pip install -r requirements.txt


(You can create requirements.txt with the following:)

streamlit
tensorflow
opencv-python
numpy
keras

3️⃣ Run the Streamlit App
streamlit run app.py

4️⃣ Upload an Image

Click “Choose an image…”

Wait for analysis

See prediction (Real or Fake) with confidence score.

📊 Sample Output
✅ The image is Real  
Confidence: 92.45%


or

❌ The image is Fake  
Confidence: 89.76%

🧪 Future Enhancements

Support for Deepfake Video Detection

Integration with Social Media Monitoring APIs

Model optimization for real-time detection

Use of Explainable AI (Grad-CAM) to visualize decision areas

👩‍💻 Author

Ishwaree Patil
📧 ishupatil2003@gmail.com
