import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
import joblib
from torchvision import transforms, models
import torch.nn as nn
import os

# --- Model Definition ---
class MultiOutputModel(nn.Module):
    def __init__(self, num_genders, num_colors, num_seasons, num_products):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.dropout = nn.Dropout(0.3)
        self.gender = nn.Linear(512, num_genders)
        self.color = nn.Linear(512, num_colors)
        self.season = nn.Linear(512, num_seasons)
        self.product = nn.Linear(512, num_products)

    def forward(self, x):
        x = self.features(x).squeeze()
        x = self.dropout(x)
        return {
            'gender': self.gender(x),
            'color': self.color(x),
            'season': self.season(x),
            'product': self.product(x)
        }

# --- Load Model and Encoders ---
@st.cache_resource(show_spinner=False)
def load_model_and_encoders():
    device = torch.device("cpu")  # Force CPU for compatibility (esp. M1/M2 Macs)

    try:
        gender_enc = joblib.load("gender_encoder.pkl")
        color_enc = joblib.load("baseColour_encoder.pkl")
        season_enc = joblib.load("usage_encoder.pkl")
        product_enc = joblib.load("masterCategory_encoder.pkl")
    except FileNotFoundError as e:
        st.error(f"Missing encoder file: {e}")
        st.stop()

    model = MultiOutputModel(
        len(gender_enc.classes_),
        len(color_enc.classes_),
        len(season_enc.classes_),
        len(product_enc.classes_)
    )

    try:
        model.load_state_dict(torch.load("fashion_model.pth", map_location=device))
    except FileNotFoundError:
        st.error("Model file 'fashion_model.pth' not found.")
        st.stop()

    model = model.to(device).eval()
    return model, gender_enc, color_enc, season_enc, product_enc, device

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# --- Prediction Function ---
def predict(image, model, gender_enc, color_enc, season_enc, product_enc, device):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        results = {
            'Gender': gender_enc.inverse_transform([outputs['gender'].argmax(0).item()])[0],
            'Color': color_enc.inverse_transform([outputs['color'].argmax(0).item()])[0],
            'Season': season_enc.inverse_transform([outputs['season'].argmax(0).item()])[0],
            'Product': product_enc.inverse_transform([outputs['product'].argmax(0).item()])[0]
        }
    return results

# --- Streamlit UI ---
st.set_page_config(page_title="Fashion Product Classifier", page_icon="👗", layout="centered")
st.title("👗 Fashion Product Classifier")
st.write("Upload a fashion product image to predict its attributes.")

uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load model and encoders only after successful image upload
        with st.spinner("Loading model..."):
            model, gender_enc, color_enc, season_enc, product_enc, device = load_model_and_encoders()

        with st.spinner("Predicting..."):
            results = predict(image, model, gender_enc, color_enc, season_enc, product_enc, device)
        st.success("Prediction complete!")

        st.markdown("### 🎯 Predicted Attributes:")
        for attr, value in results.items():
            st.markdown(f"- **{attr}:** {value}")

        with st.expander("🧠 See all possible classes"):
            st.markdown(f"**Gender:** {', '.join(gender_enc.classes_)}")
            st.markdown(f"**Color:** {', '.join(color_enc.classes_)}")
            st.markdown(f"**Season:** {', '.join(season_enc.classes_)}")
            st.markdown(f"**Product:** {', '.join(product_enc.classes_)}")

    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a JPG or PNG file.")
else:
    st.info("Please upload an image to get predictions.")
