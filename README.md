# Exposure Lab

A small, demo-ready AIML computer-vision project. Upload a photo to classify its exposure, inspect image-quality measurements and an RGB histogram, read metric-based recommendations, and download a conservatively enhanced version.

## Run locally

**Requirements:** Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5001>. The first dependency installation needs internet access. Uploaded images are processed in memory by the local Flask app and are not saved to disk or sent to a third-party service. The **Random dark image** demo button fetches a sample photo from Lorem Picsum, darkens it locally, and sends it through the same analyzer; that option requires internet access. Supported uploads are JPEG, PNG, and WEBP, up to 12 MB and 24 megapixels.

## What the app measures

- Mean and median grayscale luminance (0–255), estimated exposure offset in stops, and robust P10–P90 dynamic range; RGB channel means and saturation
- Near-black and near-white pixel percentages (clipping indicators)
- Luminance standard deviation (contrast), Laplacian variance (edge detail), and mean HSV saturation
- A 48-bin RGB channel histogram with luminance summary

## Classification method

`classify_exposure()` in `app.py` is an explainable baseline, rather than a trained neural network. It combines mean luminance relative to middle gray, an estimate of the median from robust percentiles, and the difference between highlight and shadow clipping. Wide-dynamic-range scenes receive a small score adjustment. The signed score maps to seven ordered categories: Severe, Moderate, and Mild Underexposed; Well Exposed; and Mild, Moderate, and Severe Overexposed.

The category boundaries are baseline heuristics, not a photographic standard. The displayed confidence is explicitly a **boundary-margin estimate**: it rises when a score is farther from a category boundary. It is not a calibrated probability and should not be presented as ML model accuracy. To use a trained classifier, replace `classify_exposure(gray, metrics)` while keeping its `(category, confidence, score)` return interface; use a calibrated model probability for confidence if available.

## Enhancement

The preview applies adaptive gamma based on measured mean brightness, then blends in a restrained CLAHE local-contrast adjustment on the Lab luminance channel. This preserves color and limits aggressive corrections. Severe clipping, sensor noise, and missing focus detail cannot be reconstructed from pixel statistics; the app calls those limits out in its recommendations.

## Project layout

- `app.py` — Flask API, OpenCV analysis, baseline classifier, recommendations, enhancement
- `templates/index.html` — dashboard structure
- `static/style.css` — responsive dark dashboard styling
- `static/app.js` — upload interactions, API integration, metric rendering, histogram

## Demo notes

For a quick presentation, compare a bright image, a dark image, and a balanced image. The category follows the full-image luminance distribution, so strongly backlit scenes can be classified by their overall brightness even when the subject itself is well exposed. This is a useful limitation to discuss: a future trained model should use labeled examples, test-set metrics, and possibly local/region-aware features.

This project is an educational baseline. It does not claim a learned model or clinical/forensic image-quality judgment.
