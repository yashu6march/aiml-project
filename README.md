# Image Exposure Analysis and Enhancement

This project is a small web app for looking at a photo's brightness and image details. Upload a JPG, PNG, or WEBP image to see an exposure category, measurements, a color histogram, suggestions, and an automatically enhanced preview.

## Run the app

You need Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5001> in your browser. Keep the terminal open while using the app. Press `Control+C` in the terminal to stop it.

The regular upload feature processes images in the local Flask app and does not save them. It accepts images up to 12 MB and 24 megapixels. The **Random dark image** button downloads a sample photo from Lorem Picsum, darkens it locally, and analyzes it. That button needs an internet connection; uploaded photos do not get sent to Lorem Picsum.

## What the results mean

The app reports one of seven categories, from **Severe Underexposed** through **Well Exposed** to **Severe Overexposed**. It also shows brightness, estimated exposure correction in stops, shadow and highlight clipping percentages, contrast, dynamic range, saturation, sharpness, and RGB channel averages. The histogram shows how the red, green, and blue pixel values are distributed.

The current classifier is a rule-based baseline. It uses image measurements and fixed score boundaries; it is not a trained machine-learning model. The displayed confidence is a score-boundary margin estimate, not a probability or a measured accuracy. The enhancement adjusts luminance with gamma correction and local contrast enhancement. It cannot recreate detail that was clipped or never captured.

## Files

- `app.py` contains the Flask routes, image analysis, exposure rules, recommendations, random sample handling, and enhancement.
- `templates/index.html` defines the dashboard.
- `static/style.css` styles the dashboard and smaller-screen layout.
- `static/app.js` handles uploads, requests, and result displays.
- `requirements.txt` lists the Python packages.

See the project report for the equations, implementation walkthrough, and limitations.
