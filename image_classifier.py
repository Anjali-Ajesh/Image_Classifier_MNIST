# image_classifier.py

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

def build_and_run_classifier():
    """
    Loads the MNIST dataset, builds, trains, and evaluates a CNN model.
    """
    print(f"Using TensorFlow version: {tf.__version__}")

    # --- 1. Load and Preprocess the MNIST Dataset ---
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension (for CNN input)
    # MNIST images are grayscale, so the channel dimension is 1
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    print(f"\nTraining data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    # --- 2. Build the Convolutional Neural Network (CNN) Model ---
    model = models.Sequential([
        # Convolutional Layer 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Layer 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Layer 3
        layers.Conv2D(64, (3, 3), activation='relu'),

        # Flatten the results to feed into a dense layer
        layers.Flatten(),

        # Dense Layer
        layers.Dense(64, activation='relu'),

        # Output Layer (10 classes for digits 0-9)
        layers.Dense(10)
    ])

    print("\n--- Model Architecture ---")
    model.summary()

    # --- 3. Compile the Model ---
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # --- 4. Train the Model ---
    print("\n--- Training the Model ---")
    history = model.fit(x_train, y_train, epochs=5, 
                        validation_data=(x_test, y_test))

    # --- 5. Evaluate the Model ---
    print("\n--- Evaluating Model Performance ---")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # --- 6. Make and Visualize Predictions ---
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(x_test)
    
    visualize_predictions(x_test, y_test, predictions)


def visualize_predictions(images, labels, predictions):
    """
    Displays a few sample images with their predicted and true labels.
    """
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    plt.suptitle("Model Predictions vs. True Labels", fontsize=16)

    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], labels, images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], labels)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_image(i, predictions_array, true_label, img):
    """Helper function to plot the image."""
    img = img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label[i]:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f"Predicted: {predicted_label} ({100*np.max(predictions_array):2.0f}%)\nTrue: {true_label[i]}", color=color)


def plot_value_array(i, predictions_array, true_label):
    """Helper function to plot the prediction probabilities."""
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


if __name__ == '__main__':
    build_and_run_classifier()
