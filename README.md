# Fashion-MNIST Image Classification with PyTorch

A deep learning project that classifies images of clothing into **10 different categories** using a fully connected neural network built with **PyTorch**.

The model takes a 28×28 grayscale Fashion-MNIST image, represented as **784 pixel values**, and predicts which clothing category the image belongs to.

## Project Overview

The **Fashion-MNIST** dataset contains grayscale images of clothing items.

Each image:

* Has a resolution of **28 × 28 pixels**
* Contains **784 pixel values**
* Has pixel values ranging from **0 to 255**
* Belongs to one of **10 classes**

The dataset used in this project contains:

* **60,000 training images**
* **10,000 test images**
* **785 columns** in the original CSV files

  * 1 label column
  * 784 pixel columns

## Classes

| Label | Clothing Category |
| ----: | ----------------- |
|     0 | T-shirt/top       |
|     1 | Trouser           |
|     2 | Pullover          |
|     3 | Dress             |
|     4 | Coat              |
|     5 | Sandal            |
|     6 | Shirt             |
|     7 | Sneaker           |
|     8 | Bag               |
|     9 | Ankle boot        |

## Model Architecture

The project uses a fully connected neural network implemented using PyTorch's `nn.Sequential`.

```text
Input Image
28 × 28 = 784 pixels
        ↓
Linear Layer
784 → 128
        ↓
ReLU
        ↓
Linear Layer
128 → 64
        ↓
ReLU
        ↓
Linear Layer
64 → 10
        ↓
Predicted Class
```

### Architecture

```text
784 Input Features
       ↓
   Linear(784, 128)
       ↓
      ReLU
       ↓
   Linear(128, 64)
       ↓
      ReLU
       ↓
   Linear(64, 10)
       ↓
  10 Class Output
```

## Data Preprocessing

The pixel values were originally in the range **0–255**.

They were normalized by dividing by 255:

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

The data was then converted into PyTorch tensors.

The labels were converted to `torch.long` because they are used with `CrossEntropyLoss`.

## Training

The model was trained using:

* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam
* **Learning Rate:** 0.001
* **Batch Size:** 664 for training
* **Epochs:** 1000
* **Hardware:** NVIDIA Tesla T4 GPU when available

The notebook automatically selects CUDA when a compatible GPU is available; otherwise, it uses the CPU.

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

## Training Process

For each batch, the model performs:

```text
Input Images
     ↓
Forward Pass
     ↓
Predicted Outputs
     ↓
Calculate Cross-Entropy Loss
     ↓
Backpropagation
     ↓
Optimizer Update
```

The optimizer updates the neural network parameters using the gradients calculated during backpropagation.

## Results

The trained model achieved:

### Test Accuracy: **89.19%**

The model was evaluated on the separate **10,000-image test set**.

```text
Test Accuracy: 89.19%
```

## Prediction Visualization

The project also visualizes predictions from the test dataset.

For each image, the notebook displays:

* The input clothing image
* Predicted class
* Actual class

Example workflow:

```text
784 Pixel Values
       ↓
Reshape
       ↓
28 × 28 Image
       ↓
Neural Network
       ↓
Predicted Clothing Category
```

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Google Colab
* CUDA / GPU acceleration

## Project Structure

```text
fashion-mnist-pytorch-classifier/
│
├── Fashion_MNIST.ipynb
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/MuhammadAsir/fashion-mnist-pytorch-classifier.git
cd fashion-mnist-pytorch-classifier
```

### 2. Install dependencies

```bash
pip install torch pandas numpy matplotlib
```

### 3. Add the dataset

Place the following CSV files in the project directory:

```text
fashion-mnist_train.csv
fashion-mnist_test.csv
```

### 4. Run the notebook

Open:

```text
Fashion_MNIST.ipynb
```

and run the cells sequentially.

For faster training, a CUDA-enabled GPU can be used.

## What I Learned

Through this project, I practiced:

* Loading and preparing image datasets from CSV
* Converting pixel data into model-ready tensors
* Pixel normalization
* Creating custom PyTorch `Dataset` and `DataLoader`
* Building neural networks with `nn.Sequential`
* Using Linear layers and ReLU activation
* Multi-class classification
* Cross-entropy loss
* Adam optimization
* Forward propagation
* Backpropagation
* GPU acceleration with CUDA
* Model evaluation
* Visualizing predictions

## Dataset

This project uses the **Fashion-MNIST** dataset created by Zalando Research.

The original dataset contains grayscale images of clothing items belonging to 10 categories.

Dataset source:

https://github.com/zalandoresearch/fashion-mnist

## License

The Fashion-MNIST dataset is provided under the MIT License by Zalando SE.

## Author

**Muhammad Asir Hossain Chowdhury**

Computer Science Undergraduate | AI/ML Enthusiast

---

⭐ If you found this project useful, consider giving the repository a star.
