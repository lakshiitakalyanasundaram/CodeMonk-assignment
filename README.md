# Fashion Product Classifier

This project is a deep learning application that classifies fashion products from images, predicting multiple attributes such as gender, color, season, and product category. It is designed for use cases like e-commerce cataloging and inventory management.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Model and Approach](#model-and-approach)
- [Step-by-Step Setup & Usage](#step-by-step-setup--usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The Fashion Product Classifier uses a convolutional neural network (CNN) to analyze product images and predict four key attributes:
- **Gender** (e.g., Men, Women, Boys, Girls)
- **Base Colour** (e.g., Black, Blue, Red)
- **Season/Usage** (e.g., Summer, Winter, Casual, Sports)
- **Product Category** (e.g., Apparel, Footwear, Accessories)

The project includes:
- Data preprocessing and EDA
- Model training and evaluation
- Inference and prediction on new images
- A Streamlit web app for easy user interaction

---

## Model and Approach

### Data Preparation

- **EDA & Cleaning:** The dataset is explored and cleaned using `1_EDA_and_Preprocessing.ipynb`. Missing values are handled, and categorical variables are encoded using label encoders.
- **Image Processing:** Images are resized to 128x128 pixels and normalized for model input.

### Model Architecture

- **Base Model:** The core of the model is a pre-trained ResNet-18 CNN (from torchvision), with its final classification layer removed.
- **Multi-Output Head:** The extracted features are passed through a dropout layer and then into four separate fully connected (linear) layers, each predicting one attribute (gender, color, season, product).
- **Loss Function:** The model is trained using a sum of cross-entropy losses for each output.
- **Training:** The model is trained on labeled images, and label encoders are saved for decoding predictions.

**Model Definition (PyTorch):**
```python
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
```

### Inference

- The trained model and encoders are loaded.
- An uploaded image is preprocessed and passed through the model.
- The model outputs logits for each attribute, which are decoded back to human-readable labels using the saved encoders.

---

## Step-by-Step Setup & Usage

### 1. Clone the Repository

```bash
git clone <repository-url>
cd fashion-product-classifier
```

### 2. Install Dependencies

It is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare Data (Optional)

- Use `1_EDA_and_Preprocessing.ipynb` to explore and clean your dataset.
- Use `2_Model_Training.ipynb` to train the model if you want to retrain or use your own data.
- The project already includes a trained model (`fashion_model.pth`) and encoders.

### 4. Run the Streamlit Web App

```bash
streamlit run app.py
```

- Upload a JPG or PNG image of a fashion product.
- The app will display the predicted attributes.

### 5. (Optional) Inference and API

- Use `3_Inference_and_Prediction.ipynb` to test predictions on new images in a notebook.
- Use `4_API_Deployment_FastAPI.ipynb` to deploy the model as a REST API with FastAPI.

---

## File Structure

- `app.py` — Streamlit web app for image upload and prediction.
- `1_EDA_and_Preprocessing.ipynb` — Data exploration and cleaning.
- `2_Model_Training.ipynb` — Model training and saving.
- `3_Inference_and_Prediction.ipynb` — Notebook for testing predictions.
- `4_API_Deployment_FastAPI.ipynb` — Example of API deployment.
- `fashion_model.pth` — Trained model weights.
- `*_encoder.pkl` — Label encoders for each attribute.
- `requirements.txt` — Python dependencies.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

---

## License

This project is licensed under the MIT License.

---

**Enjoy classifying your fashion products!**
