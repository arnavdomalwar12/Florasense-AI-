from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "kaggle_species_mobilenet_v3_large_best.pth"
PROFILES_PATH = APP_DIR / "care_profiles.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(architecture: str, num_classes: int) -> nn.Module:
    if architecture == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, num_classes
        )
    elif architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    return model


@st.cache_resource
def load_model() -> dict:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    try:
        checkpoint = torch.load(
            MODEL_PATH, map_location=DEVICE, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    class_names = checkpoint["class_names"]
    model = build_model(checkpoint["architecture"], len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return {
        "model": model,
        "class_names": class_names,
        "image_size": int(checkpoint.get("image_size", IMAGE_SIZE)),
        "mean": tuple(checkpoint.get("mean", IMAGENET_MEAN)),
        "std": tuple(checkpoint.get("std", IMAGENET_STD)),
    }


@st.cache_data
def load_profiles() -> dict:
    if not PROFILES_PATH.is_file():
        return {}
    import json

    return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))


def predict(image: Image.Image, model_info: dict, top_k: int = 3) -> list[dict]:
    image = image.convert("RGB")
    size = model_info["image_size"]
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(model_info["mean"], model_info["std"]),
        ]
    )(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        scores = torch.softmax(model_info["model"](tensor), dim=1)[0]
        values, indices = torch.topk(
            scores, min(top_k, len(model_info["class_names"]))
        )

    return [
        {"label": model_info["class_names"][int(index)], "score": float(value)}
        for value, index in zip(values.cpu(), indices.cpu())
    ]


def main() -> None:
    st.set_page_config(page_title="FloraSense AI", page_icon="🌿")
    st.title("🌿 FloraSense AI")
    st.write("Upload a plant image for a supported-species prediction.")
    st.warning(
        "Predictions are limited to the classes used to train this model. "
        "Do not treat an image-only prediction as a confirmed diagnosis."
    )

    try:
        model_info = load_model()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        st.error(f"Could not load the model: {error}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    predictions = predict(image, model_info)
    top_prediction = predictions[0]

    st.subheader(f"Detected species: {top_prediction['label']}")
    st.metric("Top score", f"{top_prediction['score']:.1%}")
    if top_prediction["score"] < 0.40:
        st.warning("Low model score. Retake the image or treat this result as uncertain.")

    st.subheader("Top predictions")
    for prediction in predictions:
        st.write(f"**{prediction['label']}** — {prediction['score']:.1%}")

    profile = load_profiles().get(top_prediction["label"])
    if profile:
        st.subheader("Plant information")
        plant_profile = profile.get("plant_profile", {})
        st.write(
            f"**{plant_profile.get('common_name', top_prediction['label'])}** "
            f"({plant_profile.get('scientific_name', 'Scientific name unavailable')})"
        )
        st.write(plant_profile.get("description", ""))
        care_profile = profile.get("care_profile", {})
        for label, key in (
            ("Soil", "soil"),
            ("Sunlight", "sunlight"),
            ("Watering", "watering"),
            ("Temperature", "temperature"),
        ):
            if care_profile.get(key):
                st.write(f"**{label}:** {care_profile[key]}")
        notes = care_profile.get("management_notes", [])
        if notes:
            st.write("**Management notes:**")
            for note in notes:
                st.write(f"- {note}")


if __name__ == "__main__":
    main()
