# Image Exposure Analysis and Enhancement

This project is a small web app for looking at a photo's brightness and image details. Upload a JPG, PNG, or WEBP image to see a seven-level exposure category, measurements, a color histogram, suggestions, and an automatically enhanced preview. A trained model also gives a separate three-way exposure-direction check.

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

The seven-level category is an explainable rule-based baseline. Its displayed confidence is a score-boundary margin estimate, not a probability. The separate trained Random Forest predicts **Underexposed / Well Exposed / Overexposed** using 11 image-statistics features; its displayed tree-vote share is also not a calibrated probability. The model loads from `models/msec_exposure_direction.xml`. The enhancement adjusts luminance with gamma correction and local contrast enhancement. It cannot recreate detail that was clipped or never captured.

## Dataset and measured results

The trained direction model uses the [MSEC exposure-correction dataset and official Google Drive mirrors](https://github.com/mahmoudnafifi/Exposure_Correction#dataset). The three provided splits contain 17,675 training input images, 750 validation input images, and 5,905 test input images. Each source scene has five known exposure variants in the filenames: −1.5 EV, −1 EV, 0 EV, +1 EV, and +1.5 EV. The variants are grouped into three model labels: underexposed, well exposed, and overexposed. The original exposure-correction references are included in the archives, but this classifier's labels come from the known EV suffixes.

On the held-out testing split, the trained OpenCV Random Forest achieved **74.5% three-class accuracy**, **66.5% balanced accuracy**, and **66.0% macro F1** over all 5,905 images. Validation accuracy was 74.3%. For comparison, the original seven-level rule baseline, grouped into the same three directions, achieved 63.9% accuracy, 61.8% balanced accuracy, and 60.4% macro F1 on that same test split. The model is materially better on this benchmark, though its well-exposed recall is 26.4%; that class is the main weakness.

These numbers measure exposure direction on MSEC's known exposure-shift images. They are **not** accuracy for the app's seven severity levels, and should not be presented as guaranteed performance on arbitrary user photos. The seven-level category remains the rule-based estimate; the model appears separately in the results panel as a three-way cross-check. The MSEC dataset is based on MIT-Adobe FiveK and follows its license; download the data from the source and do not add the large image archives to this GitHub repository.

To reproduce training and evaluation, download the training, validation, and testing ZIPs from the dataset page, then run this from the project folder:

```bash
python scripts/train_msec_model.py /path/to/training.zip /path/to/validation.zip /path/to/testing.zip
```

The script saves the trained model under `models/` and writes validation/test metrics to `outputs/msec_model_results.json`. Restart the web app after training so it loads the new model. The original seven-level rule baseline can be evaluated separately with:

```bash
python scripts/evaluate_exposure_benchmark.py /path/to/testing.zip --output outputs/msec_rule_baseline.json
```

## Files

- `app.py` contains the Flask routes, image analysis, exposure rules, recommendations, random sample handling, and enhancement.
- `scripts/train_msec_model.py` trains and evaluates the three-way exposure-direction model.
- `scripts/evaluate_exposure_benchmark.py` measures the existing rule baseline on MSEC's test split.
- `models/msec_exposure_direction.xml` is the trained model used by the dashboard.
- `templates/index.html` defines the dashboard.
- `static/style.css` styles the dashboard and smaller-screen layout.
- `static/app.js` handles uploads, requests, and result displays.
- `requirements.txt` lists the Python packages.

See the project report for the equations, implementation walkthrough, and limitations.
