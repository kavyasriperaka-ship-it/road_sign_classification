import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Road Sign Classification",
    page_icon="🚦",
    layout="centered"
)


# -----------------------------
# GTSRB class names
# -----------------------------
class_names = [
    "Speed limit 20 km/h",
    "Speed limit 30 km/h",
    "Speed limit 50 km/h",
    "Speed limit 60 km/h",
    "Speed limit 70 km/h",
    "Speed limit 80 km/h",
    "End of speed limit 80 km/h",
    "Speed limit 100 km/h",
    "Speed limit 120 km/h",
    "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 tons"
]


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():

    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(weights=weights)

    model.classifier[1] = nn.Linear(
        in_features=1280,
        out_features=43
    )

    model.load_state_dict(
        torch.load(
            "efficientnet_b0_road_signs.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model, weights.transforms()


model, preprocess = load_model()


# -----------------------------
# Streamlit interface
# -----------------------------
st.title("🚦 Road Sign Classification")
st.write(
    "Upload a road-sign image and the pretrained "
    "EfficientNet-B0 model will classify it."
)

uploaded_file = st.file_uploader(
    "Upload a road-sign image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Road Sign",
        use_container_width=True
    )

    input_image = preprocess(image)
    input_image = input_image.unsqueeze(0)

    with torch.no_grad():
        output = model(input_image)
        predicted_class = torch.argmax(output, dim=1).item()

    predicted_name = class_names[predicted_class]

    st.success(
        f"Predicted Road Sign: {predicted_name}"
    )

    st.info(
        f"Class Number: {predicted_class}"
    )
