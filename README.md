# MNIST Classification using TensorFlow

This repository contains code for a simple MNIST digit classification project implemented using TensorFlow. MNIST (Modified National Institute of Standards and Technology) is a popular dataset consisting of 28x28 pixel grayscale images of handwritten digits (0-9), with corresponding labels.

## Overview

In this project, we use TensorFlow, an open-source machine learning library developed by Google, to build and train a convolutional neural network (CNN) model for classifying handwritten digits from the MNIST dataset.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib (for visualization)
- Jupyter Notebook (optional, for running the notebook)

You can install the required dependencies using pip:

```
pip install tensorflow numpy matplotlib
```

## Dataset

The MNIST dataset is included with TensorFlow, so there's no need to download it separately. TensorFlow provides convenient functions to load and preprocess the data.

## Usage

1. Clone this repository:

```
git clone https://github.com/your-username/mnist-classification.git
```

2. Navigate to the project directory:

```
cd mnist-classification
```

3. Run the Jupyter Notebook `mnist_classification.ipynb` to see the code and execute the cells step by step.

```
jupyter notebook mnist_classification.ipynb
```

4. Alternatively, you can directly run the Python script `mnist_classification.py`:

```
python mnist_classification.py
```

## Model Architecture

The CNN model used for this project consists of multiple convolutional and pooling layers followed by fully connected layers. The architecture is as follows:

- Convolutional Layer 1
- Pooling Layer 1
- Convolutional Layer 2
- Pooling Layer 2
- Flatten Layer
- Fully Connected Layer 1
- Fully Connected Layer 2 (Output Layer)

## Performance

After training the model on the MNIST dataset, we achieved an accuracy of over 99% on the test set.
