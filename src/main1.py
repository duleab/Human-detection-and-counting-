import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import datetime
import os
import time
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="People Counter",
    page_icon="\U0001F465",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for model selection and configuration
st.sidebar.title("Configuration")

# Upload custom YOLOv8 model
uploaded_model = st.sidebar.file_uploader("Upload your YOLOv8 model (.pt)", type=["pt"])

# Model selection
model_options = {
    "YOLOv8n": "YOLOv8 Nano (Fastest)",
    "YOLOv8s": "YOLOv8 Small (Balanced)",
    "YOLOv8m": "YOLOv8 Medium (Most Accurate)"
}

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(model_options.keys()),
    format_func=lambda x: model_options[x]
)

# Input source selection
source_options = ["Webcam", "Video File"]
selected_source = st.sidebar.radio("Select Input Source", source_options)

uploaded_file = None
if selected_source == "Video File":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Detection settings
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

# Counting zone settings
st.sidebar.subheader("Counting Zone")
zone_type = st.sidebar.radio("Zone Type", ["Line", "Area"])

# Main content
st.title("People Counter with YOLOv8")

# Initialize session state variables if they don't exist
if 'entry_count' not in st.session_state:
    st.session_state.entry_count = 0
if 'exit_count' not in st.session_state:
    st.session_state.exit_count = 0
if 'inside_count' not in st.session_state:
    st.session_state.inside_count = 0
if 'log_data' not in st.session_state:
    st.session_state.log_data = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.datetime.now()

# Function to load model
@st.cache_resource
def load_model(model_name, uploaded_model_file=None):
    if uploaded_model_file is not None:
        model_save_path = f"uploaded_model_{int(time.time())}.pt"
        with open(model_save_path, "wb") as f:
            f.write(uploaded_model_file.read())
        return YOLO(model_save_path, task="detect")
    try:
        model_dirs = [d for d in os.listdir() if os.path.isdir(d) and model_name in d]
        if model_dirs:
            for model_dir in model_dirs:
                best_pt_path = os.path.join(model_dir, "weights", "best.pt")
                if os.path.exists(best_pt_path):
                    return YOLO(best_pt_path, task="detect")
    except Exception as e:
        st.warning(f"Could not load trained model: {e}")
    return YOLO(model_name, task="detect")


# Function to log data
def log_data(timestamp, entries, exits, people_inside):
    st.session_state.log_data.append({
        'timestamp': timestamp,
        'entries': entries,
        'exits': exits,
        'people_inside': people_inside
    })

# Function to process video frames
def process_video(model, source, confidence):
    # Initialize tracker
    tracker = sv.ByteTrack()
    
    # Initialize video capture
    if source == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        st.error(f"Error: Could not open video source")
        return
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define counting zone based on user selection
    if zone_type == "Line":
        # Create a horizontal line in the middle of the frame
        LINE_START = sv.Point(0, height // 2)
        LINE_END = sv.Point(width, height // 2)
        zone = sv.LineZone(start=LINE_START, end=LINE_END)
        zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)
    else:  # Area
        # Create a polygon zone covering the center area
        ZONE_POLYGON = np.array([
            [int(width * 0.2), int(height * 0.2)],
            [int(width * 0.8), int(height * 0.2)],
            [int(width * 0.8), int(height * 0.8)],
            [int(width * 0.2), int(height * 0.8)]
        ])
        zone = sv.PolygonZone(polygon=ZONE_POLYGON)
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone, 
            color=sv.Color.GREEN,
            thickness=2,
            text_thickness=2,
            text_scale=1
        )
    
    # Initialize box annotator
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    
    # Store tracking history
    track_history = defaultdict(list)
    inside_ids = set()
    
    # Create placeholders for video and metrics
    video_placeholder = st.empty()
    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        entries_metric = st.metric("Entries", 0)
    with metrics_cols[1]:
        exits_metric = st.metric("Exits", 0)
    with metrics_cols[2]:
        inside_metric = st.metric("People Inside", 0)
    
    # Create a chart for visualization
    chart_placeholder = st.empty()
    
    # Create a table for the log data
    table_placeholder = st.empty()
    
    # Last log time
    last_log_time = datetime.datetime.now()
    log_interval = 5  # seconds
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Run YOLOv8 inference
        results = model(frame, classes=0, conf=confidence)  # Class 0 is person in COCO dataset
        
        # Get detections
        detections = sv.Detections.from_yolov8(results[0])
        
        # Track objects
        detections = tracker.update_with_detections(detections)
        
        # Process each tracked object
        for detection_idx, (track_id, bbox) in enumerate(zip(detections.tracker_id, detections.xyxy)):
            if track_id is None:
                continue
                
            # Get center of the bounding box
            x1, y1, x2, y2 = bbox
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Add to tracking history
            track_history[track_id].append((center_x, center_y))
            
            # Keep only last 30 frames of history
            track_history[track_id] = track_history[track_id][-30:]
            
            # Check if person is in the zone
            if zone_type == "Line":
                # For line zone, we need to check if the person crossed the line
                if len(track_history[track_id]) >= 2:
                    # Get current and previous positions
                    current_pos = track_history[track_id][-1]
                    prev_pos = track_history[track_id][-2]
                    
                    # Check if the line was crossed
                    if (prev_pos[1] < height // 2 and current_pos[1] >= height // 2):
                        # Crossed from top to bottom (entry)
                        if track_id not in inside_ids:
                            inside_ids.add(track_id)
                            st.session_state.entry_count += 1
                    elif (prev_pos[1] >= height // 2 and current_pos[1] < height // 2):
                        # Crossed from bottom to top (exit)
                        if track_id in inside_ids:
                            inside_ids.remove(track_id)
                            st.session_state.exit_count += 1
            else:  # Area zone
                # For area zone, check if the person is in the zone
                is_in_zone = zone.trigger(detections=sv.Detections(
                    xyxy=np.array([bbox]),
                    tracker_id=np.array([track_id])
                ))[0]
                
                # Count entries and exits
                if is_in_zone and track_id not in inside_ids:
                    inside_ids.add(track_id)
                    st.session_state.entry_count += 1
                elif not is_in_zone and track_id in inside_ids:
                    inside_ids.remove(track_id)
                    st.session_state.exit_count += 1
            
            # Draw tracking lines
            points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], False, (0, 255, 0), 2)
        
        # Annotate frame
        labels = [
            f"ID: {tracker_id}" 
            for tracker_id in detections.tracker_id
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )
        
        # Annotate zone
        if zone_type == "Line":
            # Draw the line
            cv2.line(frame, 
                    (LINE_START.x, LINE_START.y), 
                    (LINE_END.x, LINE_END.y), 
                    (0, 255, 0), 2)
            # Add text
            cv2.putText(frame, "Entry/Exit Line", (width // 2 - 100, height // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            zone_annotator.annotate(frame=frame)
        
        # Update metrics
        st.session_state.inside_count = len(inside_ids)
        entries_metric.metric("Entries", st.session_state.entry_count)
        exits_metric.metric("Exits", st.session_state.exit_count)
        inside_metric.metric("People Inside", st.session_state.inside_count)
        
        # Log data at specified interval
        current_time = datetime.datetime.now()
        if (current_time - last_log_time).total_seconds() >= log_interval:
            log_data(
                current_time.strftime("%H:%M:%S"),
                st.session_state.entry_count,
                st.session_state.exit_count,
                st.session_state.inside_count
            )
            last_log_time = current_time
            
            # Update chart
            if st.session_state.log_data:
                df = pd.DataFrame(st.session_state.log_data)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['entries'], mode='lines+markers', name='Entries'))
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['exits'], mode='lines+markers', name='Exits'))
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['people_inside'], mode='lines+markers', name='People Inside'))
                fig.update_layout(
                    title="People Count Over Time",
                    xaxis_title="Time",
                    yaxis_title="Count",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Update table
                table_placeholder.dataframe(df)
        
        # Display the frame
        video_placeholder.image(frame, channels="BGR", use_column_width=True)
    
    cap.release()

# Main app logic
try:
    # Load the selected model
    with st.spinner(f"Loading {selected_model} model..."):
        model = load_model(selected_model)
        st.success(f"{selected_model} model loaded successfully!")
    
    # Start button
    start_button = st.button("Start People Counting")
    
    if start_button:
        if selected_source == "Webcam":
            process_video(model, "Webcam", confidence_threshold)
        elif uploaded_file is not None:
            # Save the uploaded file to a temporary location
            temp_file = f"temp_video_{int(time.time())}.mp4"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.read())
            
            # Process the video
            process_video(model, temp_file, confidence_threshold)
            
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
        else:
            st.warning("Please upload a video file.")
    
    # Display instructions
    if not start_button:
        st.info("""
        ### Instructions:
        1. Select a model from the sidebar (Nano is fastest, Medium is most accurate)
        2. Choose your input source (webcam or video file)
        3. Adjust the confidence threshold as needed
        4. Select the counting zone type (line or area)
        5. Click "Start People Counting" to begin
        
        The app will track people entering and exiting the defined zone and display real-time statistics.
        """)
        
        # Display sample image
        st.image("https://ultralytics.com/images/bus.jpg", caption="Sample detection (not from your model)")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Make sure you have installed all required packages: streamlit, ultralytics, supervision, opencv-python, numpy, pandas, plotly")