import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- Config ---
st.set_page_config(
    page_title="Mango Guard AI Lab",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded"
)
MODEL_PATH = "multi_model_bundle.pth"
IMG_SIZE = 224
DEVICE = torch.device("cpu") # CPU for inference stability

# --- Model Factory (Must match train.py) ---
def get_model(model_name, num_classes):
    """Rebuilds the model architecture."""
    if model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=None)
        # Recreate the classifier head structure used in training
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )
        target_layer = model.features[-1]
    
    elif model_name == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer = model.layer4[-1]

    elif model_name == "MobileNetV3-Large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        target_layer = model.features[-1]
    
    elif model_name == "DenseNet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        # DenseNet target layer for GradCAM
        target_layer = model.features.denseblock4.denselayer16
    
    return model, target_layer

@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Bundle {MODEL_PATH} not found! Run train.py first.")
        return None
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    return checkpoint

def process_image(image):
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    return transform(image).unsqueeze(0)

# --- Main App ---
def main():
    st.title("🥭 Mango Guard AI Lab")
    st.markdown("### Multi-Model Agricultural Disease Detection System")

    bundle = load_bundle()
    if not bundle:
        return

    class_names = bundle['class_names']
    metrics_data = bundle['metrics']
    trained_models = bundle['models']
    
    # Sidebar
    st.sidebar.header("🔬 Experiment Settings")
    selected_models = st.sidebar.multiselect(
        "Active Models", 
        list(trained_models.keys()),
        default=["DenseNet121", "MobileNetV3-Large"]
    )
    
    uploaded_file = st.sidebar.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

    # Tabs
    tab_diagnosis, tab_xai, tab_perf, tab_eda = st.tabs(["🔍 Consensus Diagnosis", "🧠 Explainability (Grad-CAM)", "📊 Performance Lab", "📂 Dataset EDA"])

    # --- TAB 1: DIAGNOSIS ---
    with tab_diagnosis:
        if uploaded_file:
            col1, col2 = st.columns([1, 2])
            image = Image.open(uploaded_file).convert("RGB")
            
            with col1:
                st.image(image, caption="Input Leaf", use_container_width=True)
            
            with col2:
                if not selected_models:
                    st.warning("Select models to run inference.")
                else:
                    results = []
                    input_tensor = process_image(image)

                    for name in selected_models:
                        # Load Model State
                        model, _ = get_model(name, len(class_names))
                        model.load_state_dict(trained_models[name])
                        model.eval()
                        
                        # Infer
                        with torch.no_grad():
                            out = model(input_tensor)
                            probs = torch.softmax(out, 1)
                            conf, idx = torch.max(probs, 1)
                        
                        results.append({
                            "Model": name,
                            "Prediction": class_names[idx.item()],
                            "Confidence": conf.item()
                        })
                    
                    df_res = pd.DataFrame(results)
                    
                    # Consensus
                    if not df_res.empty:
                        top_vote = df_res['Prediction'].mode()[0]
                        vote_count = df_res[df_res['Prediction'] == top_vote].shape[0]
                        
                        st.success(f"**Consensus Diagnosis:** {top_vote} ({vote_count}/{len(selected_models)} votes)")
                        st.dataframe(df_res.style.format({"Confidence": "{:.2%}"}), use_container_width=True)

    # --- TAB 2: EXPLAINABILITY ---
    with tab_xai:
        if uploaded_file and selected_models:
            st.subheader("Visual Attention Comparison")
            st.caption("Red regions show where the model is looking to make its decision.")
            
            cols = st.columns(len(selected_models))
            input_tensor = process_image(image)
            img_resized = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
            img_norm = img_resized.astype(np.float32) / 255.0

            for i, name in enumerate(selected_models):
                with cols[i]:
                    st.markdown(f"**{name}**")
                    with st.spinner("Generating CAM..."):
                        try:
                            model, target_layer = get_model(name, len(class_names))
                            model.load_state_dict(trained_models[name])
                            model.eval()
                            
                            cam = GradCAM(model=model, target_layers=[target_layer])
                            
                            # Get prediction first for target
                            with torch.no_grad():
                                out = model(input_tensor)
                                _, idx = torch.max(out, 1)
                            
                            targets = [ClassifierOutputTarget(idx.item())]
                            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                            viz = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)
                            
                            st.image(viz, caption=f"Pred: {class_names[idx.item()]}", use_container_width=True)
                        except Exception as e:
                            st.error(f"GradCAM failed: {e}")

    # --- TAB 3: PERFORMANCE LAB ---
    with tab_perf:
        st.subheader("🏆 Model Leaderboard")
        
        # DataFrame
        df_metrics = pd.DataFrame(metrics_data).T
        df_metrics.index.name = "Model"
        # Highlight best model
        st.dataframe(df_metrics.style.highlight_max(axis=0, color="#d1e7dd").format("{:.4f}"), use_container_width=True)
        
        # Radar Chart
        st.subheader("⚔️ Radar Comparison")
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        fig = go.Figure()

        for name in metrics_data.keys():
            # Only plot selected models to avoid clutter, or all if none selected
            if not selected_models or name in selected_models:
                values = [metrics_data[name][k] for k in categories]
                values += [values[0]] # Close the loop
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=name
                ))

        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0.9, 1.0])), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 4: DATASET EDA ---
    with tab_eda:
        st.subheader("📂 Dataset Distribution")
        
        # Count files in the data directory
        data_dir = "data/MangoLeafBD Dataset"
        if os.path.exists(data_dir):
            class_counts = {}
            for cls in class_names:
                cls_path = os.path.join(data_dir, cls)
                if os.path.exists(cls_path):
                    class_counts[cls] = len(os.listdir(cls_path))
            
            if class_counts:
                df_counts = pd.DataFrame.from_dict(class_counts, orient='index', columns=['Count'])
                st.bar_chart(df_counts)
                
                st.subheader("📸 Sample Images")
                selected_cls = st.selectbox("View Class Samples", class_names)
                cls_path = os.path.join(data_dir, selected_cls)
                if os.path.exists(cls_path):
                    files = os.listdir(cls_path)[:4] # Show first 4
                    cols = st.columns(4)
                    for i, f in enumerate(files):
                        img_path = os.path.join(cls_path, f)
                        cols[i].image(Image.open(img_path), caption=f, use_container_width=True)
        else:
            st.info("Dataset folder structure not found for EDA. (Expected 'data/MangoLeafBD Dataset')")

if __name__ == "__main__":
    main()