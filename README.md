# MNIST Image Classifier with TensorFlow

A Python project that builds, trains, and evaluates a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. This script uses TensorFlow and its high-level Keras API.

## Features

-   **Deep Learning Model:** Implements a Convolutional Neural Network (CNN), the standard architecture for image classification tasks.
-   **Data Handling:** Automatically loads and preprocesses the MNIST dataset from `tensorflow.keras.datasets`.
-   **Model Training:** Trains the CNN on 60,000 training images and validates its performance.
-   **Performance Evaluation:** Measures the model's accuracy on a test set of 10,000 unseen images.
-   **Prediction Visualization:** Displays a few sample images from the test set along with the model's prediction and the true label.

## Technology Stack

-   **Python**
-   **TensorFlow:** The core deep learning framework used to build and train the neural network.
-   **NumPy:** For numerical operations and data manipulation.
-   **Matplotlib:** For visualizing the sample predictions.

## Setup and Usage

To run this project, you will need Python and pip installed.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/image-classifier.git](https://github.com/your-username/image-classifier.git)
    cd image-classifier
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install the required libraries
    pip install tensorflow numpy matplotlib
    ```

3.  **Run the Script:**
    Execute the Python script from your terminal. TensorFlow will automatically download the MNIST dataset the first time you run it.
    ```bash
    python image_classifier.py
    ```

The script will output the model's architecture, training progress, final test accuracy, and then display a window with sample predictions.
