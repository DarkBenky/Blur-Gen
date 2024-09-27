import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input , BatchNormalization, LeakyReLU
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
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

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
    def encoder_block(x, filters, kernel_size=3, padding='same'):
        x = Conv2D(filters, kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters, kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    # Decoder
    def decoder_block(x, skip_features, filters, kernel_size=3, padding='same'):
        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, skip_features])
        x = Conv2D(filters, kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters, kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    # Encoder
    e1 = encoder_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(e1)
    e2 = encoder_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(e2)
    e3 = encoder_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(e3)
    e4 = encoder_block(p3, 512)
    p4 = MaxPooling2D((2, 2))(e4)

    # Bridge
    b = encoder_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b, e4, 512)
    d2 = decoder_block(d1, e3, 256)
    d3 = decoder_block(d2, e2, 128)
    d4 = decoder_block(d3, e1, 64)

    # Output
    outputs = Conv2D(3, 1, activation='sigmoid')(d4)

    model = Model(inputs, outputs, name=model_name)
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
    models = []
    for i in range(1, num_levels + 1):
        # check if model exists
        if os.path.exists(f'best_model_level_{i}.h5'):
            print(f"Loading saved model: best_model_level_{i}.h5")
            model = load_model(f'best_model_level_{i}.h5', custom_objects={'loss': perceptual_loss()})

        model = unet_model(model_name=f"unet_{i}")
        model.compile(optimizer=Adam(learning_rate=1e-4), loss=perceptual_loss())
        
        blurred_batches = blur_batch(images, num_levels, blur_strength, noise=noise, std=std)
        
        X = np.array([batch[i] for batch in blurred_batches])
        Y = np.array([batch[i - 1] for batch in blurred_batches])
        
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
        
        log_dir = os.path.join("logs", f"level_{i}", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        callbacks = [
            TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True),
            ModelCheckpoint(f'best_model_level_{i}.h5', save_best_only=True, monitor='val_loss'),
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        model.fit(
            X_train, Y_train, 
            batch_size=batch_size, 
            epochs=epochs,
            validation_data=(X_val, Y_val),
            callbacks=callbacks
        )
        
        models.append(model)
    
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
    dataset = load_dataset(max_samples=256_000)
    images = get_images_from_dataset(dataset)
    
    # Split data into train and test sets
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=72)
    
    # TODO: More Levels and Lower Blur Strength
    num_levels = 8
    blur_strength = 0.1
    epochs = 25
    batch_size = 128
    noise = True
    std = 0.025

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