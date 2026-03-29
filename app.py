from flask import Flask, render_template, request
import joblib
import pytesseract
import cv2
import numpy as np
import requests
import re
from textblob import TextBlob
from deep_translator import GoogleTranslator
from deepface import DeepFace

# ----------------------------
# Configuration
# ----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

# Load text sentiment model
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ----------------------------
# Clean Text
# ----------------------------
def clean_text(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()


# ----------------------------
# Translate Text
# ----------------------------
def translate_text(text):
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text


# ----------------------------
# Predict Text Sentiment
# Returns: label, score
# score: positive=+1, negative=-1
# ----------------------------
def predict_text_sentiment(text):
    if text is None or text.strip() == "":
        return "No text detected", 0

    translated = translate_text(text)
    cleaned = clean_text(translated)

    if cleaned == "":
        return "No text detected", 0

    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    if pred == 1:
        return "Positive", 1
    else:
        return "Negative", -1


# ----------------------------
# OCR Text Extraction
# ----------------------------
def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    text = pytesseract.image_to_string(gray, lang="eng", config="--psm 6")
    text = str(TextBlob(text).correct())
    return text.strip()


# ----------------------------
# Detect Face Emotion
# Returns: emotion_label, score
# score mapping:
# happy/surprise -> +1
# neutral -> 0
# sad/angry/fear/disgust -> -1
# ----------------------------
def predict_facial_emotion(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            return "No face detected", 0

        # largest face select chestam
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        face_roi = image[y:y+h, x:x+w]

        analysis = DeepFace.analyze(
            face_roi,
            actions=["emotion"],
            enforce_detection=False
        )

        if isinstance(analysis, list):
            analysis = analysis[0]

        emotion = analysis.get("dominant_emotion", "neutral").lower()

        if emotion in ["happy", "surprise"]:
            return emotion.capitalize(), 1
        elif emotion in ["neutral"]:
            return emotion.capitalize(), 0
        else:
            return emotion.capitalize(), -1

    except Exception:
        return "Emotion detection failed", 0


# ----------------------------
# Fusion Logic
# text has more weight than face
# final_score = 0.7 * text + 0.3 * face
# ----------------------------
def fuse_sentiment(text_score, face_score):
    final_score = (0.7 * text_score) + (0.3 * face_score)

    if final_score >= 0:
        return "Positive"
    else:
        return "Negative"


# ----------------------------
# Read image from uploaded file
# ----------------------------
def load_uploaded_image(file):
    file_bytes = file.read()
    if not file_bytes:
        return None

    image_array = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


# ----------------------------
# Read image from URL
# ----------------------------
def load_image_from_url(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type.lower():
        return None

    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


# ----------------------------
# Home Route - Text Input
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = {
        "input_text": "",
        "extracted_text": "",
        "text_sentiment": "",
        "face_emotion": "",
        "final_sentiment": ""
    }

    if request.method == "POST":
        text = request.form.get("text", "").strip()

        text_label, text_score = predict_text_sentiment(text)
        final_label = "Positive" if text_score >= 0 else "Negative"

        result["input_text"] = text
        result["extracted_text"] = text
        result["text_sentiment"] = text_label
        result["face_emotion"] = "Not applicable"
        result["final_sentiment"] = final_label

    return render_template("index.html", result=result)


# ----------------------------
# Image Upload Route
# ----------------------------
@app.route("/image", methods=["POST"])
def image():
    result = {
        "input_text": "",
        "extracted_text": "",
        "text_sentiment": "",
        "face_emotion": "",
        "final_sentiment": ""
    }

    try:
        file = request.files.get("file")

        if file is None or file.filename == "":
            result["final_sentiment"] = "Please upload an image file"
            return render_template("index.html", result=result)

        image = load_uploaded_image(file)

        if image is None:
            result["final_sentiment"] = "Invalid image file"
            return render_template("index.html", result=result)

        extracted_text = extract_text_from_image(image)
        text_label, text_score = predict_text_sentiment(extracted_text)
        face_label, face_score = predict_facial_emotion(image)
        final_label = fuse_sentiment(text_score, face_score)

        result["extracted_text"] = extracted_text if extracted_text else "No text detected"
        result["text_sentiment"] = text_label
        result["face_emotion"] = face_label
        result["final_sentiment"] = final_label

    except Exception as e:
        result["final_sentiment"] = f"Error processing uploaded image: {str(e)}"

    return render_template("index.html", result=result)


# ----------------------------
# Image URL Route
# ----------------------------
@app.route("/image-url", methods=["POST"])
def image_url():
    result = {
        "input_text": "",
        "extracted_text": "",
        "text_sentiment": "",
        "face_emotion": "",
        "final_sentiment": ""
    }

    try:
        url = request.form.get("url", "").strip()

        if url == "":
            result["final_sentiment"] = "Please enter an image URL"
            return render_template("index.html", result=result)

        image = load_image_from_url(url)

        if image is None:
            result["final_sentiment"] = "Invalid image URL or image could not be loaded"
            return render_template("index.html", result=result)

        extracted_text = extract_text_from_image(image)
        text_label, text_score = predict_text_sentiment(extracted_text)
        face_label, face_score = predict_facial_emotion(image)
        final_label = fuse_sentiment(text_score, face_score)

        result["extracted_text"] = extracted_text if extracted_text else "No text detected"
        result["text_sentiment"] = text_label
        result["face_emotion"] = face_label
        result["final_sentiment"] = final_label

    except requests.exceptions.RequestException:
        result["final_sentiment"] = "Could not download image from URL"

    except Exception as e:
        result["final_sentiment"] = f"Error processing image URL: {str(e)}"

    return render_template("index.html", result=result)


# ----------------------------
# Run Flask App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)