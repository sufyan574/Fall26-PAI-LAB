import os
import uuid
from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import threading
import time

# in-memory job store for background processing
jobs = {}

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = dict()
        self.disappeared = dict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = np.linalg.norm(np.array(objectCentroids)[:, None] - inputCentroids[None, :], axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects


def process_video(input_path, output_path, min_area=500, job=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError('Could not open video')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    tracker = CentroidTracker(maxDisappeared=40)
    counted = set()
    total_count = 0

    # counting line (horizontal)
    line_y = height // 2

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg = backSub.apply(frame)
        _, fg = cv2.threshold(fg, 244, 255, cv2.THRESH_BINARY)
        fg = cv2.medianBlur(fg, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            rects.append((x, y, x + w, y + h))

        objects = tracker.update(rects)

        # Draw counting line
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

        for (objectID, centroid) in objects.items():
            cX, cY = centroid
            text = f"ID {objectID}"
            cv2.putText(frame, text, (cX - 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 4, (0, 255, 0), -1)

            # simple crossing logic
            if objectID not in counted:
                # if centroid has crossed the line (from top to bottom)
                # (we don't have previous centroid stored separately, so approximate by checking y)
                if cY >= line_y:
                    counted.add(objectID)
                    total_count += 1

        # draw rects
        for (startX, startY, endX, endY) in rects:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # overlay count
        cv2.putText(frame, f"Count: {total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        writer.write(frame)

        # update job with latest frame (JPEG bytes) and count
        if job is not None:
            try:
                _, jpg = cv2.imencode('.jpg', frame)
                job['latest_frame'] = jpg.tobytes()
                job['total'] = total_count
                job['frame_idx'] = frame_idx
            except Exception:
                pass

        frame_idx += 1

    cap.release()
    writer.release()
    # mark job finished
    if job is not None:
        job['finished'] = True
        job['total'] = total_count
        job['output'] = output_path

    return total_count


@app.route('/')
def index():
    # list processed output videos to show as examples on the front-end
    outputs = []
    try:
        for fname in os.listdir(OUTPUT_FOLDER):
            if fname.lower().endswith(('.mp4', '.webm', '.ogg')):
                outputs.append(os.path.join(OUTPUT_FOLDER, fname))
    except Exception:
        outputs = []

    return render_template('index.html', outputs=outputs)


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('index'))

    ext = os.path.splitext(file.filename)[1]
    uid = f"{uuid.uuid4().hex}{ext}"
    input_path = os.path.join(UPLOAD_FOLDER, uid)
    file.save(input_path)

    output_name = f"processed_{uid.split('.')[0]}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    # create background job
    job_id = uuid.uuid4().hex
    jobs[job_id] = {'id': job_id, 'status': 'queued', 'latest_frame': None, 'total': 0, 'finished': False, 'output': None}

    def bg():
        jobs[job_id]['status'] = 'processing'
        try:
            process_video(input_path, output_path, job=jobs[job_id])
            jobs[job_id]['status'] = 'done'
        except Exception as e:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = str(e)

    thread = threading.Thread(target=bg, daemon=True)
    thread.start()

    return render_template('result_live.html', job_id=job_id)


@app.route('/stream/<job_id>')
def stream(job_id):
    if job_id not in jobs:
        return ('', 404)

    def generate():
        # MJPEG stream
        while True:
            job = jobs.get(job_id)
            if job is None:
                break
            frame = job.get('latest_frame')
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            if job.get('finished'):
                break
            time.sleep(0.05)

    return app.response_class(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/job_status/<job_id>')
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return ({'status': 'missing'}, 404)
    return {
        'status': job.get('status'),
        'total': job.get('total', 0),
        'finished': bool(job.get('finished')),
        'output': job.get('output') and '/' + job.get('output')
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
