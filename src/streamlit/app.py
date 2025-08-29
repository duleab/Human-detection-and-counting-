# streamlit_app.py
import streamlit as st
from ultralytics import YOLO
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load YOLOv8 model
@st.cache_resource
def load_model():
    model_path = r'D:\Project\Human detection\Human detection\Yolov8m\Yolov8m.pt'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()

# App UI
st.title("🧍 People Detection")
st.write("Upload an image to detect and count people")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        # Load and process image
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        
        # Run detection
        results = model.predict(img_np, verbose=False)
        boxes = results[0].boxes
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img_np)
        ax.axis('off')
        
        people_count = 0
        
        if boxes is not None:
            for box, cls_id in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)):
                if cls_id == 0:  # Person class
                    x1, y1, x2, y2 = box
                    rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=2, edgecolor='lime', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, f'Person {people_count + 1}', 
                           color='white', fontsize=8,
                           bbox=dict(facecolor='black', alpha=0.5, pad=1))
                    people_count += 1
        
        # Display results
        ax.set_title("Detection Results", fontweight='bold', fontsize=18)
        ax.text(0.5, -0.05, f"Total People: {people_count}",
               fontsize=12, ha='center', va='center',
               transform=ax.transAxes, fontweight='bold')
        
        st.pyplot(fig)
        st.success(f"Detected {people_count} people in the image!")
        
        # Show original and processed side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)  # Updated parameter
        with col2:
            st.pyplot(fig, use_container_width=True)  # Updated parameter
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")