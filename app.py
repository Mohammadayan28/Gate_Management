from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
import cv2, time, re, csv, sys, subprocess
from datetime import datetime
from collections import Counter

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# -------------------------------
# INSTALL MODEL
# -------------------------------
try:
    from fast_alpr import ALPR
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "fast-alpr", "onnxruntime", "Pillow",
        "--break-system-packages"
    ])
    from fast_alpr import ALPR

# -------------------------------
# CAMERA
# -------------------------------
cap = cv2.VideoCapture(0)
time.sleep(1)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# -------------------------------
# LOAD MODEL
# -------------------------------
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-s-v2-global-model",
)

# -------------------------------
# VALIDATION
# -------------------------------
PLATE_PATTERNS = [
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$'),
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{5}$'),
]

def clean(text):
    return re.sub(r'[\s\-\.\|_,;:\n]', '', text).upper()

def validate(text):
    t = clean(text)
    for p in PLATE_PATTERNS:
        if p.match(t):
            return t
    return None

# -------------------------------
# DATABASE
# -------------------------------
vehicles = {}
try:
    with open("vehicles.csv") as f:
        for row in csv.DictReader(f):
            vehicles[row["plate"]] = {
                "authorized": row["authorized"] == "True",
                "owner": row["owner"]
            }
except:
    print("No DB found")

def check_access(plate):
    if plate in vehicles:
        v = vehicles[plate]
        return ("ACCESS GRANTED" if v["authorized"] else "DENIED"), v["owner"]
    return "NOT FOUND", "Unknown"

# -------------------------------
# GET LAST VISIT (IMPORTANT)
# -------------------------------
def get_last_visit(plate):
    try:
        with open("logs.csv", "r") as f:
            rows = list(csv.reader(f))

        # search from latest
        for row in reversed(rows[:-1]):  # skip current write
            log_plate, owner, status, timestamp = row
            if log_plate == plate:
                return {
                    "plate": log_plate,
                    "owner": owner,
                    "status": status,
                    "time": timestamp
                }

    except:
        pass

    return None

# -------------------------------
# LOGGING
# -------------------------------
def log_entry(plate, owner, status):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("logs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([plate, owner, status, now])

# -------------------------------
# STATE
# -------------------------------
plate_buffer = []
confidence_buffer = []
last_emit_time = 0
COOLDOWN = 3

# -------------------------------
# VIDEO STREAM
# -------------------------------
def generate_frames():
    while True:
        try:
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.resize(frame, (480, 360))
            _, buffer = cv2.imencode('.jpg', frame)

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

        except:
            continue

# -------------------------------
# SCAN LOOP
# -------------------------------
def scan_loop():
    global plate_buffer, confidence_buffer, last_emit_time

    print("Scan loop started")

    while True:
        try:
            success, frame = cap.read()
            if not success:
                continue

            results = alpr.predict(frame)

            for r in results:
                raw = r.ocr.text if r.ocr else ""
                conf = r.ocr.confidence if r.ocr else 0

                if isinstance(conf, list):
                    conf = sum(conf) / len(conf)

                plate = validate(raw)

                if plate:
                    plate_buffer.append(plate)
                    confidence_buffer.append(conf)

            if len(plate_buffer) >= 3:

                final_plate = Counter(plate_buffer).most_common(1)[0][0]
                avg_conf = sum(confidence_buffer) / len(confidence_buffer)

                plate_buffer.clear()
                confidence_buffer.clear()

                current_time = time.time()

                if current_time - last_emit_time > COOLDOWN:

                    status, owner = check_access(final_plate)

                    # 🔥 GET LAST VISIT BEFORE LOGGING NEW ONE
                    last_visit = get_last_visit(final_plate)

                    log_entry(final_plate, owner, status)

                    socketio.emit("plate_detected", {
                        "plate": final_plate,
                        "status": status,
                        "owner": owner,
                        "confidence": round(avg_conf, 2),
                        "last_visit": last_visit
                    })

                    print(final_plate, status)

                    last_emit_time = current_time

        except Exception as e:
            print("Scan error:", e)

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    socketio.start_background_task(scan_loop)
    socketio.run(app, debug=True, use_reloader=False)