# CNN Image Classification using PyTorch

This project implements an image classification system using **Convolutional Neural Networks (CNNs)** with the **CIFAR-10 dataset** in **PyTorch**. The trained model can classify images into 10 different categories.

## 📂 Project Structure
cifar10_project/ <br>
│── datasets/ # (⚠️ Not included in Git) Place CIFAR-10 dataset here <br>
│── logs/ # (⚠️ Not included in Git) Training logs and model checkpoints <br>
│── models/ # Contains the CNN model architecture <br>
│── static/ # Static files (CSS, JS, images) for web app <br>
│── templates/ # HTML templates for Flask app │── app.py # Flask web app to upload and classify images <br>
│── requirements.txt # List of required dependencies <br>
│── train_model_pytorch.ipynb # Jupyter Notebook for training the model <br>
│── README.md # Project documentation <br>


## 📥 Dataset Installation
This project uses the **CIFAR-10 dataset**, which contains 60,000 32x32 color images in 10 classes.

### **1️⃣ Download the CIFAR-10 Dataset**
You can manually download the dataset from:
👉 [CIFAR-10 Dataset Download](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

### **2️⃣ Place the Dataset in the `datasets/` Folder**
After downloading the dataset, extract it inside the **datasets/** folder.

## 🚀 Installation & Setup

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/your-username/CNN-Image-Classification.git
cd CNN-Image-Classification
```
### 2️⃣ Create a Virtual Environment
```sh
python -m venv cifar10_env
source cifar10_env/bin/activate  # On Windows: cifar10_env\Scripts\activate
```
### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```
### 4️⃣ Train the Model
```sh
jupyter notebook train_model_pytorch.ipynb
```

### 3️⃣ Install Dependencies
```sh
python app.py
```
Then open http://127.0.0.1:5000/ in your browser to upload and classify images.

### 🛠️ Technologies Used
* Python
* PyTorch
* Flask (for web interface)
* NumPy & Pandas (for data processing)
* OpenCV (for image handling)
* HTML, CSS, JavaScript (for the front-end)

### 📜 License
This project is open-source

## 🔹 Contributors:
[Anshul Khaire](https://github.com/anshul-dying)