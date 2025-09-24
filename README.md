🚀 Classifying-MNIST-Handwritten-Digits-using-PyTorch
Classifying handwritten digits (0–9) from the MNIST dataset using PyTorch.
The project covers data loading, preprocessing, building a neural network, training with GPU acceleration, and visualizing predictions.
A beginner-friendly introduction to image classification using deep learning.
________________________________________
📌 Project Overview
This project demonstrates how to:
•	Load and preprocess image data using torchvision datasets and transforms.
•	Build and train a multi-layer perceptron (MLP) model for image classification.
•	Implement a custom training loop with forward pass → loss → backpropagation → optimizer update.
•	Utilize GPU acceleration (CUDA) for faster training.
•	Visualize predictions on test images.
The dataset used is the MNIST dataset, containing 60,000 training images and 10,000 test images of handwritten digits (28×28 pixels, grayscale).
________________________________________
⚙️ Tech Stack
•	Python 3.8+
•	PyTorch (for model training and GPU acceleration)
•	torchvision (for datasets and transforms)
•	NumPy & Matplotlib (for data handling and visualization)
•	Jupyter Notebook (for development)
________________________________________
🚀 How to Run
Clone this repository
      
    git clone https://github.com/satyamk1234/Classifying-MNIST-Handwritten-Digits-using-PyTorch.git
    cd Classifying-MNIST-Handwritten-Digits-using-PyTorch

Install dependencies 

    pip install -r requirements.txt

Run the notebook jupyter notebook mnist_classification.ipynb
________________________________________
📊 Results
•	Built a neural network to classify handwritten digits with high accuracy (~97–98% on test set).
•	Demonstrated GPU usage for faster training.
•	Visualized predictions on sample test images to confirm model performance.
________________________________________
🎯 Learning Outcomes
•	Loading and preprocessing image datasets with PyTorch and torchvision.
•	Implementing multi-layer perceptron (MLP) models for classification.
•	Custom training loops with loss and optimizer.
•	Using CUDA-enabled GPU for deep learning.
•	Visualizing model predictions.
________________________________________
📂 Project Structure
├── mnist_classification.ipynb  # Main notebook
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
________________________________________
🙌 Acknowledgments
•	Dataset: MNIST Handwritten Digits
•	Book Reference: Machine Learning with PyTorch and Scikit-Learn by Sebastian Raschka et al.




