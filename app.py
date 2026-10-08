from __future__ import annotations

import base64
import io
import os
import secrets
import urllib.request
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024

CATEGORIES = [
    'Severe Underexposed', 'Moderate Underexposed', 'Mild Underexposed',
    'Well Exposed', 'Mild Overexposed', 'Moderate Overexposed', 'Severe Overexposed'
]


def load_image(file_bytes: bytes) -> np.ndarray:
    """Decode supported uploads safely and return an RGB uint8 image."""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image).convert('RGB')
        if image.width * image.height > 24_000_000:
            raise ValueError('Image is too large to process (maximum 24 megapixels).')
        return np.asarray(image)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError('That file could not be read as an image. Try JPG, PNG, or WEBP.') from exc


def classify_exposure(gray: np.ndarray, metrics: dict[str, float]) -> tuple[str, float, float]:
    """Interpretable exposure baseline. Score is centered at 0; |score| is severity.

    The score is centered on middle gray (128/255). Boundaries in exposure-score units: 0.45 / 0.95 / 1.55. The returned
    confidence is a boundary-margin heuristic, not a calibrated probability.
    A learned classifier can replace this function behind the same interface.
    """
    mean = metrics['mean_luminance']
    p10, p90 = metrics['p10_luminance'], metrics['p90_luminance']
    dark, bright = metrics['dark_pixel_pct'], metrics['bright_pixel_pct']
    # Robust luminance and clipped-pixel evidence reduce sensitivity to tiny highlights.
    score = (0.72 * ((mean - 128.0) / 55.0) +
             0.20 * ((p50_safe(p10, p90) - 128.0) / 60.0) +
             0.08 * ((bright - dark) / 18.0))
    # A wide dynamic scene can have a low mean without being uniformly underexposed.
    if p90 - p10 > 145:
        score *= 0.88
    magnitude = abs(score)
    if magnitude < 0.45:
        category = 'Well Exposed'
        lower, upper = 0.0, 0.45
    elif magnitude < 0.95:
        category = 'Mild Underexposed' if score < 0 else 'Mild Overexposed'
        lower, upper = 0.45, 0.95
    elif magnitude < 1.55:
        category = 'Moderate Underexposed' if score < 0 else 'Moderate Overexposed'
        lower, upper = 0.95, 1.55
    else:
        category = 'Severe Underexposed' if score < 0 else 'Severe Overexposed'
        lower, upper = 1.55, 2.8
    confidence = min(0.98, 0.56 + 0.40 * min(1.0, min(magnitude - lower, upper - magnitude) / 0.25))
    return category, round(confidence, 2), round(float(score), 3)


def p50_safe(p10: float, p90: float) -> float:
    return (p10 + p90) / 2



def fetch_random_dark_sample() -> np.ndarray:
    """Fetch a random demo photo and darken it locally for an exposure demo."""
    seed = secrets.token_hex(8)
    url = f'https://picsum.photos/seed/exposurelab-{seed}/960/720'
    req = urllib.request.Request(url, headers={'User-Agent': 'ExposureLab/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            content_type = response.headers.get('Content-Type', '')
            data = response.read(8 * 1024 * 1024 + 1)
        if not content_type.startswith('image/') or len(data) > 8 * 1024 * 1024:
            raise ValueError('The sample image service returned an unsupported response.')
        rgb = load_image(data)
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError('Could not fetch a sample image right now. Please try again or upload your own image.') from exc
    # Simulate a darker capture while retaining enough shadow texture for the
    # enhancement preview to demonstrate recoverable detail.
    darkened = np.clip((rgb.astype(np.float32) / 255.0) ** 1.75 * 255.0, 0, 255).astype(np.uint8)
    return darkened

def analyze(rgb: np.ndarray) -> dict[str, Any]:
    height, width = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mean = float(np.mean(gray))
    channel_means = np.mean(rgb, axis=(0, 1))
    luminance_std = float(np.std(gray))
    exposure_offset = float(np.log2(118.0 / max(mean, 1.0)))
    dynamic_range = float(np.percentile(gray, 90) - np.percentile(gray, 10))
    p10, p50, p90 = [float(v) for v in np.percentile(gray, [10, 50, 90])]
    metrics = {
        'mean_luminance': round(mean, 1), 'median_luminance': round(p50, 1),
        'p10_luminance': round(p10, 1), 'p90_luminance': round(p90, 1),
        'dark_pixel_pct': round(float(np.mean(gray < 32) * 100), 1),
        'bright_pixel_pct': round(float(np.mean(gray > 235) * 100), 1),
        'contrast': round(float(np.std(gray)), 1),
        'sharpness': round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        'mean_saturation': round(float(np.mean(hsv[:, :, 1]) / 255 * 100), 1),
        'dynamic_range': round(p90 - p10, 1),
        'luminance_stddev': round(luminance_std, 1),
        'exposure_offset_ev': round(float(np.clip(exposure_offset, -6, 6)), 2),
        'red_mean': round(float(channel_means[0]), 1),
        'green_mean': round(float(channel_means[1]), 1),
        'blue_mean': round(float(channel_means[2]), 1),
        'dynamic_range_label': 'Wide' if dynamic_range > 150 else 'Moderate' if dynamic_range > 85 else 'Low',
    }
    category, confidence, score = classify_exposure(gray, metrics)
    rgb_hist = {}
    for name, channel in zip(['red', 'green', 'blue'], cv2.split(bgr)[::-1]):
        channel_hist = cv2.calcHist([channel], [0], None, [48], [0, 256]).flatten()
        rgb_hist[name] = (channel_hist / max(float(channel_hist.max()), 1) * 100).round(1).tolist()
    hist = cv2.calcHist([gray], [0], None, [48], [0, 256]).flatten()
    hist = (hist / max(float(hist.max()), 1) * 100).round(1).tolist()
    recommendations = []
    if mean < 90:
        recommendations.append({'title': 'Increase exposure', 'detail': f"Try about +{abs(exposure_offset):.1f} EV as a starting point; review highlights after brightening.", 'tag': 'Exposure'})
    elif mean > 170:
        recommendations.append({'title': 'Recover highlight detail', 'detail': 'Lower exposure or highlights to bring texture back into bright regions.', 'tag': 'Highlights'})
    else:
        recommendations.append({'title': 'Keep exposure balanced', 'detail': 'Overall brightness is in a useful range; make only small local adjustments.', 'tag': 'Exposure'})
    if metrics['contrast'] < 42:
        recommendations.append({'title': 'Add gentle local contrast', 'detail': 'A restrained contrast adjustment can separate subjects from the background.', 'tag': 'Contrast'})
    elif metrics['contrast'] > 85:
        recommendations.append({'title': 'Soften extreme contrast', 'detail': 'Lift shadows or reduce highlights to preserve detail at both ends.', 'tag': 'Contrast'})
    if metrics['dark_pixel_pct'] > 18:
        recommendations.append({'title': 'Inspect shadow clipping', 'detail': f"{metrics['dark_pixel_pct']}% of pixels are near black; check important shadow areas.", 'tag': 'Clipping'})
    if metrics['bright_pixel_pct'] > 10:
        recommendations.append({'title': 'Inspect highlight clipping', 'detail': f"{metrics['bright_pixel_pct']}% of pixels are near white; clipped detail cannot be fully recovered.", 'tag': 'Clipping'})
    if metrics['sharpness'] < 70:
        recommendations.append({'title': 'Check focus and motion', 'detail': 'The image has low measured edge detail; sharpening cannot restore missing focus information.', 'tag': 'Detail'})
    if metrics['mean_saturation'] < 12:
        recommendations.append({'title': 'Enhance saturation', 'detail': 'Colors are fairly muted; increase vibrance slightly only if it suits the scene.', 'tag': 'Color'})
    red, green, blue = channel_means
    cast = max((red, green, blue)) - min((red, green, blue))
    if cast > 24:
        warm = red > blue
        recommendations.append({'title': 'Review white balance', 'detail': f"Channel means are R {red:.0f}, G {green:.0f}, B {blue:.0f}; a {'cooler' if warm else 'warmer'} white balance may neutralize the color cast.", 'tag': 'Color'})
    enhanced = enhance(rgb, mean, category)
    ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError('Could not encode the enhanced image.')
    return {
        'category': category, 'confidence': confidence, 'confidence_label': 'boundary-margin estimate',
        'score': score, 'metrics': metrics, 'histogram': hist, 'rgb_histogram': rgb_hist,
        'recommendations': recommendations, 'width': width, 'height': height,
        'enhanced_image': 'data:image/jpeg;base64,' + base64.b64encode(encoded.tobytes()).decode('ascii'),
    }


def enhance(rgb: np.ndarray, mean: float, category: str) -> np.ndarray:
    """Conservative adaptive gamma + luminance CLAHE; preserve original colors."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0]
    if 'Underexposed' in category:
        target = 112.0
        gamma = float(np.clip(np.log(max(target, 1) / 255) / np.log(max(mean, 1) / 255), 0.58, 0.92))
    elif 'Overexposed' in category:
        target = 142.0
        gamma = float(np.clip(np.log(max(target, 1) / 255) / np.log(max(mean, 1) / 255), 1.05, 1.65))
    else:
        gamma = 1.0
    lut = np.array([np.clip(((i / 255.0) ** gamma) * 255, 0, 255) for i in range(256)], dtype=np.uint8)
    corrected = cv2.LUT(luminance, lut)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    local = clahe.apply(corrected)
    # Blend conservatively so texture remains natural and noise amplification is limited.
    lab[:, :, 0] = cv2.addWeighted(corrected, 0.72, local, 0.28, 0)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


@app.get('/')
def index():
    return render_template('index.html')


@app.post('/api/analyze')
def api_analyze():
    if 'image' not in request.files:
        return jsonify(error='Choose an image to analyze.'), 400
    file = request.files['image']
    if not file.filename:
        return jsonify(error='Choose an image to analyze.'), 400
    try:
        rgb = load_image(file.read())
        return jsonify(analyze(rgb))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception('Image analysis failed')
        return jsonify(error='Analysis failed. Try another image file.'), 500


@app.get('/api/random-dark-image')
def api_random_dark_image():
    try:
        rgb = fetch_random_dark_sample()
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 91])
        if not ok:
            raise ValueError('Could not prepare the sample image.')
        return jsonify(image='data:image/jpeg;base64,' + base64.b64encode(encoded.tobytes()).decode('ascii'),
                       filename='random-dark-sample.jpg', source='Lorem Picsum; darkened locally for demonstration')
    except ValueError as exc:
        return jsonify(error=str(exc)), 502


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', '5001')), debug=os.environ.get('FLASK_DEBUG') == '1')
