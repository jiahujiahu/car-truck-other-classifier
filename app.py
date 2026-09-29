import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib

from PIL import Image
from torchvision.models import alexnet, AlexNet_Weights


device = torch.device("cpu")


class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=3):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Load saved MLP
checkpoint = torch.load(
    "three_class_alexnet_mlp.pt",
    map_location=device
)

model = FeatureMLP(
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    num_classes=checkpoint["num_classes"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# Load scaler
scaler = joblib.load(
    "alexnet_scaler.joblib"
)


# AlexNet feature extractor
weights = AlexNet_Weights.DEFAULT
alexnet_model = alexnet(weights=weights)

extractor = nn.Sequential(
    alexnet_model.features,
    alexnet_model.avgpool,
    nn.Flatten(),
    *list(alexnet_model.classifier.children())[:6],
).eval()


class_names = [
    "car",
    "truck",
    "other"
]


def predict(image):

    image = image.convert("RGB")

    x = weights.transforms()(image)
    x = x.unsqueeze(0)

    with torch.no_grad():
        features = extractor(x)

    features = (
        features.numpy()
        .astype(np.float32)
    )

    features_scaled = scaler.transform(
        features
    ).astype(np.float32)

    x_features = torch.from_numpy(
        features_scaled
    )

    with torch.no_grad():
        logits = model(x_features)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0].numpy()

    return probabilities


st.title("Car, Truck, or Other?")

st.write(
    "Upload an image and the model will predict "
    "whether it is a car, truck, or another object."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded image",
        width=400
    )

    probabilities = predict(image)

    prediction = int(
        np.argmax(probabilities)
    )

    st.subheader(
        f"Prediction: {class_names[prediction]}"
    )

    st.write(
        f"Car: {probabilities[0] * 100:.2f}%"
    )

    st.write(
        f"Truck: {probabilities[1] * 100:.2f}%"
    )

    st.write(
        f"Other: {probabilities[2] * 100:.2f}%"
    )