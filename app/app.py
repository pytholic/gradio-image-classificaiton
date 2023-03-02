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
from PIL import Image

from model import Classifier

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


# Define the sample images
sample_images = {
    "dog": "./test_images/dog.jpeg",
    "cat": "./test_images/cat.jpeg",
    "butterfly": "./test_images/butterfly.jpeg",
    "elephant": "./test_images/elephant.jpg",
    "horse": "./test_images/horse.jpeg",
}

# Define the function to make predictions on an image
def predict(image):
    try:
        image = preprocess(image).unsqueeze(0)

        # Prediction
        # Make a prediction on the image
        with torch.no_grad():
            output = model(image)
            # convert to probabilities
            probabilities = torch.nn.functional.softmax(torch.exp(output[0]), dim=0)
            print(probabilities)
            topk_prob, topk_label = torch.topk(probabilities, 3)

            # convert the predictions to a list
            predictions = []
            for i in range(topk_prob.size(0)):
                prob = topk_prob[i].item()
                label = topk_label[i].item()
                predictions.append((prob, label))

            return predictions
    except Exception as e:
        print(f"Error predicting image: {e}")
        return []


# Define the Streamlit app
def app():

    # Define the input and output interfaces
    inputs = gr.inputs.Image()
    outputs = gr.outputs.Label(num_top_classes=3)

    # Create the Gradio interface
    iface = gr.Interface(
        predict,
        inputs,
        outputs,
        title="Animal-10 Classifier",
        description="Classify images of 10 different animals.",
        examples=[
            ["./test_images/dog.jpeg"],
            ["./test_images/cat.jpeg"],
            ["./test_images/butterfly.jpeg"],
            ["./test_images/elephant.jpg"],
            ["./test_images/horse.jpeg"],
        ],
    )

    # Set the styling of the title bar
    iface.layout["title_bar_color"] = "#4E4E4E"
    iface.layout["title_color"] = "white"

    # Set the styling of the progress bars
    for i in range(3):
        iface.layout[f"output_{i}"]["style"]["progress_bar_color"] = "#FFA500"

    return iface


# Run the app
app().launch()


# Run the app
if __name__ == "__main__":
    app()
