import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import albumentations as A
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from model import Classifier
from PIL import Image

# Load the model
model = Classifier.load_from_checkpoint("./models/checkpoint.ckpt")
model.eval()

# Define labels
labels = [
    "dog",
    "horse",
    "elephant",
    "butterfly",
    "chicken",
    "cat",
    "cow",
    "sheep",
    "spider",
    "squirrel",
]

# Preprocess function
def preprocess(image):
    image = np.array(image)
    resize = A.Resize(224, 224)
    normalize = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    to_tensor = ToTensorV2()
    transform = A.Compose([resize, normalize, to_tensor])
    image = transform(image=image)["image"]
    return image


# Define the function to make predictions on an image
def predict(image):
    try:
        image = preprocess(image).unsqueeze(0)

        # Prediction
        # Make a prediction on the image
        with torch.no_grad():
            output = model(image)
            # convert to probabilities
            probabilities = torch.nn.functional.softmax(output[0])

            # Return the top 3 predictions
        return {labels[i]: float(probabilities[i]) for i in range(3)}
    except Exception as e:
        print(f"Error predicting image: {e}")
        return []


# Define the interface
def app():
    title = "Animal-10 Image Classification"
    description = "Classify images using a custom CNN model and deploy using Gradio."

    gr.Interface(
        title=title,
        description=description,
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(
            num_top_classes=3,
        ),
        examples=[
            "./test_images/dog.jpeg",
            "./test_images/cat.jpeg",
            "./test_images/butterfly.jpeg",
            "./test_images/elephant.jpg",
            "./test_images/horse.jpeg",
        ],
    ).launch()


# Run the app
if __name__ == "__main__":
    app()
