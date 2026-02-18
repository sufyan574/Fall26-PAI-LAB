AI Video Object Counter

Overview
- Flask web app that accepts video uploads and counts moving objects crossing a horizontal line using OpenCV background subtraction + simple centroid tracking.

Quick start (Windows)
1. Create a virtual environment and activate it:

```
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install requirements:

```
pip install -r requirements.txt
```

3. Run the app:

```
python "d:/New folder (4)/run.py"
```

4. Open http://127.0.0.1:5000 in your browser, upload a video, and wait for processing.

Notes
- Processing is synchronous and may take time depending on video length and CPU.
- The processed video will be saved to `static/outputs`.
- Tweak `min_area` inside `process_video` in `app.py` to tune sensitivity.

Next steps (ideas)
- Add asynchronous processing with task queue (Celery/RQ).
- Use a DNN detector (MobileNet-SSD) for class-aware counting.
- Stream progress updates to the front-end.

Deployment
- Docker: build and run with:

```
docker build -t ai-counter .
docker run -p 5000:5000 ai-counter
```

- Heroku / PaaS: `Procfile` included for quick deploy.
