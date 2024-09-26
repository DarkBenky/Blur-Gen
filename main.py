import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import fiftyone as fo
import fiftyone.zoo as foz
from tensorflow.keras.applications.vgg19 import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import tensorboard
import datetime
import os
from tensorflow.keras.models import load_model

# Load the dataset
def load_dataset(max_samples=50):
    return foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=["person", "car"],
        max_samples=max_samples
    )

# Extract images from the FiftyOne dataset for training
def get_images_from_dataset(dataset, target_size=(128, 128)):
    images = []
    for sample in dataset:
        image = cv2.imread(sample.filepath)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = cv2.resize(image, target_size)
            image = image.astype(np.float32) / 255.0
            images.append(image)
    return np.array(images)

def add_noise(image, std=10):
    # Calculate the mean intensity of the image
    mean_intensity = np.mean(image)
    
    # Generate Gaussian noise with the mean of the image and a given standard deviation
    noise = np.random.normal(mean_intensity, std, image.shape).astype(np.float32)
    
    # Add noise to the image
    noisy_image = image.astype(np.float32) + noise
    return noisy_image

# Adjustable blur strength
def create_blur_levels(image, num_levels, blur_strength, noise = False, std = 5):
    blurred_images = [image]
    for i in range(1, num_levels + 1):
        sigma = i * blur_strength
        if noise:
            blurred = add_noise(image, std=std)
        else:
            blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=sigma)
        blurred_images.append(blurred)
    return blurred_images

# Perceptual loss function using VGG19
def perceptual_loss():
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    vgg.trainable = False
    loss_model = Model(vgg.input, vgg.get_layer('block5_conv4').output)
    loss_model.trainable = False
    
    def loss(y_true, y_pred):
        y_true = preprocess_input(y_true * 255.0)  # Scale back to 0-255 range
        y_pred = preprocess_input(y_pred * 255.0)
        return tf.reduce_mean(tf.square(loss_model(y_true) - loss_model(y_pred)))
    
    return loss

# U-Net model architecture
def unet_model(input_size=(128, 128, 3), model_name="unet"):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up5 = UpSampling2D(size=(2, 2))(conv4)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    output = Conv2D(3, 1, activation='sigmoid')(conv7)

    model = Model(inputs, output, name=model_name)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss=perceptual_loss(), metrics=['mae'])
    
    return model

# Function to blur batches of images
def blur_batch(images, num_levels, blur_strength, noise=False, std=5):
    return np.array([create_blur_levels(image, num_levels, blur_strength, noise=noise, std=std) for image in images])

def load_or_create_models(num_levels, model_name="unet", noise=False):
    models = []
    for i in range(1, num_levels + 1):
        model_path = f'deblur_model_noise_level_{i}.h5' if noise else f'deblur_model_level_{i}.h5'
        if os.path.exists(model_path):
            print(f"Loading saved model: {model_path}")
            model = load_model(model_path, custom_objects={'loss': perceptual_loss()})
        else:
            print(f"No saved model found. Creating new model: {model_name}_{i}")
            model = unet_model(model_name=f"{model_name}_{i}")
        models.append(model)
    return models

# Function to prepare train and validation sets
def train_on_fiftyone_dataset(images, num_levels=5, blur_strength=2.0, batch_size=16, epochs=10, noise=False, std=5):
    # Load or create models
    models = load_or_create_models(num_levels, model_name="unet", noise=noise)
    
    # Apply the blurring and prepare the dataset for training
    blurred_batches = blur_batch(images, num_levels, blur_strength, noise=noise, std=std)
    
    for i in range(1, num_levels + 1):
        X = np.array([batch[i] for batch in blurred_batches])  # Input: Blur `i`
        Y = np.array([batch[i - 1] for batch in blurred_batches])  # Output: Predict previous blur level
        
        # Split into training and validation sets
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
        
        # Set up TensorBoard callback
        log_dir = os.path.join("logs", f"level_{i}", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
        
        # Continue training the model for each blur level
        models[i - 1].fit(
            X_train, Y_train, 
            batch_size=batch_size, 
            epochs=epochs,
            validation_data=(X_val, Y_val),
            callbacks=[tensorboard_callback]
        )
    
    return models

def visualize_results(models, test_images, num_levels, blur_strength, noise=False, std=5):
    num_images = min(5, len(test_images))  # Visualize up to 5 images
    # Adjust the number of columns to account for the random noise and its deblurring steps (+1 for noise and +number of models for deblurring it)
    fig, axes = plt.subplots(num_images, num_levels + len(models) + 3, figsize=(3 * (num_levels + len(models) + 3), 3 * num_images))
    
    for i, image in enumerate(test_images[:num_images]):
        blurred_images = create_blur_levels(image, num_levels, blur_strength, noise=noise, std=std)
        
        # Original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Blurred image (most blurred)
        axes[i, 1].imshow(blurred_images[-1])
        axes[i, 1].set_title('Blurred')
        axes[i, 1].axis('off')
        
        # Deblurred images
        deblurred = blurred_images[-1]
        for j, model in enumerate(reversed(models)):
            deblurred = model.predict(np.expand_dims(deblurred, 0))[0]
            axes[i, j + 2].imshow(np.clip(deblurred, 0, 1))
            axes[i, j + 2].set_title(f'Deblur {j + 1}')
            axes[i, j + 2].axis('off')

    # Random noise image generation and visualization
    random_noise = np.random.normal(0, 1, test_images[0].shape)  # Match noise size to the test image shape
    axes[0, num_levels + 2].imshow(np.clip(random_noise, 0, 1))
    axes[0, num_levels + 2].set_title('Random Noise')
    axes[0, num_levels + 2].axis('off')

    # Deblurring the random noise image
    for j, model in enumerate(reversed(models)):
        random_noise = model.predict(np.expand_dims(random_noise, 0))[0]
        axes[0, j + num_levels + 3].imshow(np.clip(random_noise, 0, 1))
        axes[0, j + num_levels + 3].set_title(f'Deblur {j + 1}')
        axes[0, j + num_levels + 3].axis('off')
    
    plt.tight_layout()
    if noise:
        plt.savefig('deblurring_results_noise.png')
    else:
        plt.savefig('deblurring_results.png')
    plt.close()

if __name__ == "__main__":
    dataset = load_dataset(max_samples=128_000)
    images = get_images_from_dataset(dataset)
    
    # Split data into train and test sets
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=72)
    
    # TODO: More Levels and Lower Blur Strength
    num_levels = 4
    blur_strength = 0.1
    epochs = 100
    batch_size = 1024
    noise = True
    std = 0.2

    # load saved models

    
    models = train_on_fiftyone_dataset(train_images, num_levels=num_levels, blur_strength=blur_strength, epochs=epochs, batch_size=batch_size, noise=noise, std=std)

    # Save the models
    for i, model in enumerate(models):
        model.save(f'deblur_model_noise_level_{i+1}.h5')

    print("Training complete. Models saved.")

    # Visualize results
    visualize_results(models, test_images, num_levels, blur_strength, noise=noise, std=std)
    print("Results visualized and saved as 'deblurring_results.png'")
    
    print("To view TensorBoard logs, run:")
    print("tensorboard --logdir=logs")