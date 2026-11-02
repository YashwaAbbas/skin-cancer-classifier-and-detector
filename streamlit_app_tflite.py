import streamlit as st
import numpy as np
from PIL import Image
import os
from pathlib import Path
import tensorflow as tf

# App layout configuration
st.set_page_config(
    page_title="VisionAI Skin Cancer Detection Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling injector for Modern Deep Dark Interface and Custom Credits Layout
st.markdown("""
    <style>
    .main, .stApp {
        background-color: #0E1117;
        color: #E0E2E6;
    }
    div[data-testid="stMarkdownContainer"] h1, h2, h3 {
        color: #00D2C4 !important;
    }
    .credit-box {
        background-color: #1E222B;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #00D2C4;
        margin-bottom: 25px;
        text-align: right;
    }
    .metadata-card {
        background-color: #161A24;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2B303C;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- STREAMLIT SIDEBAR: SKIN CANCER EDUCATIONAL REFERENCE ---
with st.sidebar:
    st.header("🩺 Skin Cancer Reference")
    st.write("This database provides verified clinical markers for computer-vision research and educational exploration.")

    with st.expander("🔍 The ABCDE Criteria", expanded=True):
        st.markdown("""
        Dermatologists use the ABCDE criteria to identify warning signs of Melanoma:
        - **A is for Asymmetry**: One half of the spot does not match the other half.
        - **B is for Border**: The edges are irregular, ragged, notched, or blurred.
        - **C is for Color**: The color is not uniform; shades of brown, black, red, or blue may be present.
        - **D is for Diameter**: The spot is larger than 6 millimeters across (about the size of a pencil eraser).
        - **E is for Evolving**: The lesion is changing in size, shape, color, or height over time.
        """)

    with st.expander("🔬 Common Lesion Classes"):
        st.markdown("""
        - **Melanocytic Nevi (Moles)**: Benign proliferations of melanocytes. Uniform color, round/oval with clean boundaries.
        - **Melanoma**: Highly aggressive skin cancer originating in pigment-producing cells. Requires immediate clinical attention.
        - **Basal Cell Carcinoma (BCC)**: The most common type of skin cancer. Usually appears as a pearly or waxy bump or a flat, flesh-colored lesion.
        - **Seborrheic Keratosis**: Non-cancerous benign skin growth common in older adults, often has a stuck-on appearance.
        """)

    with st.expander("📸 Imaging Standards"):
        st.markdown("""
        - **Dermatoscopy**: Specialized polarized/magnified imagery. High resolution, minimal surface reflection.
        - **Clinical Macro Images**: Taken with typical cameras, high noise levels, lighting variance, and background skin distractions.
        """)

# Credit Area top-right using columns
col_title, col_credit = st.columns([3, 2])

with col_title:
    st.title("🌌 VisionAI Skin Cancer Lab")
    st.write("Computer Vision Portal Dedicated to Skin Cancer Detection & Analysis")

with col_credit:
    st.markdown("""
    <div class='credit-box'>
         <h4 style='margin:0; color:#00D2C4;'>Student Portal Verified</h4>
         <p style='margin:2px 0; font-size:14px; font-weight:bold;'>Yashwa Abbas</p>
         <p style='margin:2px 0; font-size:12px; color:#A0A4B0;'>Course: Generative AI</p>
         <p style='margin:2px 0; font-size:12px; color:#A0A4B0;'>Instructor: Sir Karar</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Educational Notice Alert
st.warning("⚠️ **Notice**: This tool is developed strictly to demonstrate skin cancer detection using computer vision algorithms. AI predictions must always be validated by clinical professionals.")

# TFLite model loader: keep skin_cancer_cnn.tflite beside this app file.
@st.cache_resource
def load_vision_model():
    model_path = Path(__file__).resolve().parent / "skin_cancer_cnn.tflite"
    classes_path = Path(__file__).resolve().parent / "classes.npy"
    class_names = ["Benign", "Malignant"]
    if classes_path.is_file():
        class_names = np.load(classes_path, allow_pickle=True).astype(str).tolist()
    if not model_path.is_file():
        return None, class_names
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter, class_names


def predict_tflite(interpreter, image):
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    input_shape = input_info["shape"]
    height, width = int(input_shape[1]), int(input_shape[2])
    resized = image.convert("RGB").resize((width, height))
    values = np.asarray(resized, dtype=np.float32)[None, ...]
    input_dtype = input_info["dtype"]

    # Quantize input only when the TFLite model expects an integer tensor.
    if np.issubdtype(input_dtype, np.integer):
        scale, zero_point = input_info.get("quantization", (0.0, 0))
        if scale:
            values = np.round(values / scale + zero_point)
        limits = np.iinfo(input_dtype)
        values = np.clip(values, limits.min, limits.max).astype(input_dtype)
    else:
        values = values.astype(input_dtype)

    interpreter.set_tensor(input_info["index"], values)
    interpreter.invoke()
    output = interpreter.get_tensor(output_info["index"])

    # Convert quantized outputs back to floating-point scores.
    if np.issubdtype(output.dtype, np.integer):
        scale, zero_point = output_info.get("quantization", (0.0, 0))
        if scale:
            output = (output.astype(np.float32) - zero_point) * scale
    return np.asarray(output[0], dtype=np.float32).reshape(-1)


model, class_names = load_vision_model()

# Left Column: Upload / Right Column: Analytics
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📂 Lesion Upload & Diagnostics (Skin Cancer CV Only)")
    uploaded_file = st.file_uploader("Select a skin lesion image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Lesion Target Image", use_container_width=True)

with col_right:
    st.subheader("📊 Computer Vision Analytics Engine")
    st.markdown("### 🧠 Skin Cancer CV Inferences")
    if uploaded_file is not None:
        width, height = image.size
        file_size_kb = len(uploaded_file.getvalue()) / 1024.0

        st.markdown("### 🛠️ Image Metadata Diagnostics")
        st.markdown(f"""
        <div class='metadata-card'>
            <b>File Dimensions:</b> {width} x {height} px<br>
            <b>File Format:</b> {uploaded_file.type}<br>
            <b>Approximate Weight:</b> {file_size_kb:.2f} KB<br>
            <b>Dominant Color Mode:</b> {image.mode}
        </div>
        """, unsafe_allow_html=True)

        if model is not None:
            with st.spinner("Running forward pass for lesion classification..."):
                # Run the TFLite model using its declared input tensor shape and dtype.
                predictions = predict_tflite(model, image)
                if len(predictions) == len(class_names):
                    best_idx = np.argmax(predictions)
                    confidence = predictions[best_idx] * 100
                    st.success(f"Classification: **{class_names[best_idx]}** ({confidence:.2f}% confidence)")

                    # Individual class progress meters
                    for name, prob in zip(class_names, predictions):
                        st.progress(float(prob), text=f"{name}: {prob*100:.1f}%")
                else:
                    if len(predictions) == 1 or len(class_names) == 2:
                        prob_val = float(predictions[0])
                        prob_list = [1.0 - prob_val, prob_val]
                        best_idx = np.argmax(prob_list)
                        confidence = prob_list[best_idx] * 100
                        st.success(f"Classification: **{class_names[best_idx]}** ({confidence:.2f}% confidence)")
                        for idx, name in enumerate(class_names):
                            st.progress(float(prob_list[idx]), text=f"{name}: {prob_list[idx]*100:.1f}%")
                    else:
                        best_idx = np.argmax(predictions)
                        st.info(f"Inference finished. Dominant index: {best_idx}")
        else:
            st.error("TFLite model not found or could not be loaded. Add skin_cancer_cnn.tflite beside streamlit_app.py.")
    else:
        st.info("Please upload a skin lesion image to begin running computer vision diagnostics.")
