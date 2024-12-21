import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import threading
import time
import base64

##########################
# Model Definitions
##########################

class ConvMixer(nn.Module):

    def __init__(self, dim=64, depth=4, kernel_size=5, patch_size=2, n_classes=6):
        super(ConvMixer, self).__init__()
        self.dim = dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.n_classes = n_classes

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=3 if i == 0 else self.dim,
                      out_channels=self.dim,
                      kernel_size=self.kernel_size,
                      stride=self.patch_size,
                      padding=1)
            for i in range(self.depth)
        ])

        dummy_input = torch.randn(1, 3, 128, 128)
        self.flattened_size = self._get_output_size(dummy_input)
        self.fc = nn.Linear(self.flattened_size, self.n_classes)

    def _get_output_size(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        return int(torch.prod(torch.tensor(x.size()[1:])).item())

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load the freshness model
freshness_model = ConvMixer()
freshness_model.load_state_dict(torch.load(
    r'C:\Users\cheng\Documents\VSC\Higher_level_CV\Examination_project\saved_models_for_freshness_d\model_freshness_base_model.pth',
    map_location=torch.device('cpu')
))
freshness_model.eval()

# Load YOLO model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
yolo_model.conf = 0.25

allowed_fruit_classes = ['banana', 'apple', 'orange']
class_mapping = {
    0: 'Apple_Good', 1: 'Apple_Bad',
    2: 'Banana_Good', 3: 'Banana_Bad',
    4: 'Orange_Good', 5: 'Orange_Bad'
}

def preprocess_image(img, target_size=(128, 128)):
    # For model input only: convert BGR to RGB and normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img_pil = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_pil).unsqueeze(0)
    return img_tensor

def predict_freshness(cropped_img):
    img_tensor = preprocess_image(cropped_img)
    freshness_model.eval()
    with torch.no_grad():
        outputs = freshness_model(img_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()
        predicted_label = class_mapping[predicted_class]

        if 'Good' in predicted_label:
            freshness_label = 'Fresh'
            freshness_conf = torch.sigmoid(outputs)[0][predicted_class].item()
        else:
            freshness_label = 'Spoiled'
            freshness_conf = 1 - torch.sigmoid(outputs)[0][predicted_class].item()
    return freshness_label, freshness_conf

def crop_and_encode_image(frame, x1, y1, x2, y2):
    # Just encode directly without color conversion
    cropped = frame[y1:y2, x1:x2]
    if cropped.size > 0:
        _, buffer = cv2.imencode('.jpg', cropped)
        encoded = base64.b64encode(buffer).decode('utf-8')
        return encoded
    return None

historical_detections = []
last_detection_time = {}
cooldown = 2.0  # 2-second cooldown per class

def process_with_model(frame):
    results = yolo_model(frame)
    detections = results.xyxy[0].cpu().numpy()
    output_detections = []

    for *box, conf, cls in detections:
        class_name = results.names[int(cls)]
        if class_name in allowed_fruit_classes:
            x1, y1, x2, y2 = map(int, box)
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                freshness_label, freshness_conf = predict_freshness(cropped)
                color = (0, 255, 0) if freshness_label == 'Fresh' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{class_name}: {freshness_label} ({freshness_conf:.2f})"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                encoded_image = crop_and_encode_image(frame, x1, y1, x2, y2)
                detection_data = {
                    'class': class_name,
                    'freshness': freshness_label,
                    'bbox': [x1, y1, x2, y2],
                    'image': encoded_image
                }
                output_detections.append(detection_data)

                # Add to historical detections with cooldown
                current_time = time.time()
                if class_name not in last_detection_time or (current_time - last_detection_time[class_name]) > cooldown:
                    historical_detections.append(detection_data)
                    last_detection_time[class_name] = current_time

    return frame, output_detections

app = Flask(__name__)

camera_active = False
capture = None
frame_lock = threading.Lock()
latest_frame = None
latest_detections = []

def camera_loop():
    global capture, camera_active, latest_frame, latest_detections
    while camera_active:
        ret, frame = capture.read()
        if not ret:
            continue
        processed_frame, detections = process_with_model(frame)
        with frame_lock:
            latest_frame = processed_frame.copy()
            latest_detections = detections
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active, capture
    if not camera_active:
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            return jsonify({'status': 'error', 'message': 'Camera not found'}), 500
        camera_active = True
        threading.Thread(target=camera_loop, daemon=True).start()
    return jsonify({'status': 'started'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active, capture
    camera_active = False
    if capture is not None:
        capture.release()
        capture = None
    return jsonify({'status': 'stopped'})

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        while True:
            if not camera_active:
                time.sleep(0.1)
                continue
            with frame_lock:
                if latest_frame is not None:
                    # Directly encode the BGR frame
                    ret, buffer = cv2.imencode('.jpg', latest_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    time.sleep(0.01)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_historical_detections', methods=['GET'])
def get_historical_detections():
    return jsonify(historical_detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

