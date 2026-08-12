import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import PotatoCNN
from llm import get_recommendation

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Potato Leaf Disease Classifier",
    page_icon="🌿",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

.main{
    background-color:#F7F9FC;
}

.title{
    text-align:center;
    color:#2E7D32;
    font-size:40px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}

.result{
    background:#000000;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================

st.markdown(
    "<div class='title'>🌿 Potato Leaf Disease Classifier</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Deep Learning using PyTorch</div>",
    unsafe_allow_html=True
)

st.write("")

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Disease Prediction",
        "Disease Guide",
        "About"
    ]
)

# ======================================================
# CLASS NAMES
# ======================================================

class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

# ======================================================
# IMAGE TRANSFORM
# ======================================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():

    model = PotatoCNN(num_classes=3)

    model.load_state_dict(
        torch.load(
            "best_potato_model.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


# ======================================================
# PREDICTION PAGE
# ======================================================

if page == "Disease Prediction":

    st.markdown("## Upload a Potato Leaf")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        disease = None
        recommendation = None

        with col2:
            if st.button("🔍 Predict Disease"):
                model = load_model()
                img = transform(image)
                img = img.unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(img)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, prediction = torch.max(probabilities, dim=1)

                disease = class_names[prediction.item()]
                confidence_val = confidence.item() * 100

                st.markdown(
                    f"""
                    <div class="result">
                    Prediction
                    <br><br>
                    🌿 {disease}
                    <br><br>
                    Confidence
                    <br>
                    {confidence_val:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")
                st.subheader("Prediction Probabilities")

                probs = probabilities.cpu().numpy()[0]
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(class_names, probs)
                ax.set_xlim([0, 1])
                ax.set_xlabel("Probability")
                st.pyplot(fig)

                # Retrieve recommendation to display outside the column layout
                recommendation = get_recommendation(disease)

        if recommendation:
            st.subheader("🌱 AI Recommendation")
            st.write(recommendation)
# ======================================================
# DISEASE GUIDE
# ======================================================

elif page == "Disease Guide":

    st.header("Disease Information")

    disease = st.selectbox(
        "Choose a disease",
        [
            "Early Blight",
            "Late Blight",
            "Healthy"
        ]
    )

    if disease == "Early Blight":

        st.success("Symptoms")

        st.write("""
        - Brown circular spots
        - Yellow edges
        - Older leaves affected first
        """)

        st.info("Treatment")

        st.write("""
        - Remove infected leaves
        - Apply fungicide
        - Rotate crops
        """)

    elif disease == "Late Blight":

        st.success("Symptoms")

        st.write("""
        - Dark brown lesions
        - White fungal growth
        - Rapid spread
        """)

        st.info("Treatment")

        st.write("""
        - Mancozeb
        - Chlorothalonil
        - Remove infected plants
        """)

    else:

        st.success("Healthy Leaf")

        st.write("""
        The leaf appears healthy.

        Continue regular monitoring.
        """)

# ======================================================
# ABOUT
# ======================================================

else:

    st.header("About This Project")

    st.write("""
    This application uses a Deep Learning Convolutional Neural Network (CNN)
    developed with PyTorch to classify potato leaf diseases.

    ### Classes

    - Potato Early Blight
    - Potato Late Blight
    - Healthy Potato Leaf

    ### Frameworks

    - PyTorch
    - Streamlit
    - TorchVision
    - Pillow
    - Matplotlib

    ### Input

    RGB image (224 × 224)

    ### Output

    Predicted disease and confidence score.
""")

st.sidebar.markdown("---")
st.sidebar.write("Developed by Natural Sciences Group, Techrise Cohort 3.0")