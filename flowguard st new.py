import cv2
import torch
import numpy as np
import streamlit as st
import tempfile
from fpdf import FPDF

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Global variables
log_file_name = "detection_log.txt"
fps = 30
output_width = 800
output_height = 600

def detect_objects(frame, frame_index):
    """
    Detect objects in the frame and draw bounding boxes.
    """
    original_h, original_w = frame.shape[:2]
    resized_frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)

    # Perform detection
    results = model(resized_frame)
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class_id]

    detection_summary = {}
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        x1 = int(x1 * original_w / 640)
        y1 = int(y1 * original_h / 640)
        x2 = int(x2 * original_w / 640)
        y2 = int(y2 * original_h / 640)

        class_name = model.names[int(class_id)]
        if class_name not in detection_summary:
            detection_summary[class_name] = 0
        detection_summary[class_name] += 1

        label = f"{class_name}: {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    timestamp = frame_index / fps
    timestamp_str = f"{int(timestamp // 60)}:{int(timestamp % 60):02d}"

    with open(log_file_name, "a") as log_file:
        for class_name, count in detection_summary.items():
            log_file.write(f"{frame_index}, {timestamp_str}, {class_name}, {count}\n")

    return frame


def process_video(video_file):
    """
    Process uploaded video file and detect objects.
    """
    global fps
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_index = 0
    with open(log_file_name, "w") as log_file:
        log_file.write("Frame, Timestamp, Object, Count\n")

    stframe = st.empty()  # Streamlit video placeholder
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = detect_objects(frame, frame_index)
        display_frame = cv2.resize(output_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
        stframe.image(display_frame, channels="BGR", use_container_width=True)

        frame_index += 1
    cap.release()


def extract_to_pdf(start_time, end_time):
    """
    Extract specific log entries and save to PDF.
    """
    start_seconds = int(start_time.split(":")[0]) * 60 + int(start_time.split(":")[1])
    end_seconds = int(end_time.split(":")[0]) * 60 + int(end_time.split(":")[1])

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Detection Log", ln=True, align="C")

    with open(log_file_name, "r") as log_file:
        lines = log_file.readlines()
        pdf.set_font("Arial", size=10)
        for line in lines:
            if line.startswith("Frame"):
                continue
            frame, timestamp, obj, count = line.strip().split(", ")
            current_seconds = int(timestamp.split(":")[0]) * 60 + int(timestamp.split(":")[1])
            if start_seconds <= current_seconds <= end_seconds:
                pdf.cell(0, 10, f"Frame: {frame}, Time: {timestamp}, Object: {obj}, Count: {count}", ln=True)

    pdf_path = "detection_log.pdf"
    pdf.output(pdf_path)
    return pdf_path


# Streamlit App Layout
st.title("Flow Guard v1.3")
st.header("Object Detection")

# Video upload and processing
uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mkv"])
if uploaded_file:
    st.write("Processing video...")
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    process_video(temp_file.name)

# Time input for log extraction
st.subheader("Extract Log Section to PDF")
start_time = st.text_input("Start Time (MM:SS)", "00:00")
end_time = st.text_input("End Time (MM:SS)", "00:10")

if st.button("Extract to PDF"):
    pdf_file = extract_to_pdf(start_time, end_time)
    st.success("PDF generated successfully!")
    st.download_button("Download PDF", open(pdf_file, "rb"), "detection_log.pdf", "application/pdf")

# View log file
if st.button("View Log File"):
    if os.path.exists(log_file_name):
        with open(log_file_name, "r") as log_file:
            st.text(log_file.read())
    else:
        st.warning("Log file not found.")
