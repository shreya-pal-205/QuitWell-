# app.py
import os
import io
import json
import traceback
from datetime import datetime, date
import base64

import numpy as np
from PIL import Image
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, jsonify, flash
)
from werkzeug.utils import secure_filename
import tensorflow as tf
from dotenv import load_dotenv


# -----------------------
# Load env
# -----------------------
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
FLASK_SECRET = os.environ.get("FLASK_SECRET", "replace-this-for-prod")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")   # CHANGED (supports image)


# -----------------------
# Google Gemini SDK
# -----------------------
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini SDK configured.")
    else:
        print("⚠️ No API key — Gemini calls will use mock.")
except Exception as e:
    genai = None
    print("⚠️ Could not import Gemini:", e)



# -----------------------
# App config
# -----------------------
MODEL_PATH = "oral_cancer_model.h5"
CLASS_INDEX_PATH = "class_indices.json"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

app = Flask(__name__)
app.secret_key = FLASK_SECRET
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# -----------------------
# Load TF model
# -----------------------
model = None
try:
    print(f"Loading model from {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded.")
except Exception as e:
    print("❌ Failed to load model:", e)



# -----------------------
# Load class indices
# -----------------------
try:
    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
    class_labels = {int(v): k for k, v in class_indices.items()}
except Exception:
    class_labels = {0: "Oral Cancer photos", 1: "Normal"}



# -----------------------
# Helpers
# -----------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(path, target_size=(224, 224)):
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image(img_path):
    if model is None:
        return {"label": "Model not loaded", "prob": 0.0}

    img_array = preprocess_image(img_path)
    pred = model.predict(img_array)

    score = float(np.asarray(pred).reshape(-1)[0])
    predicted_class = 1 if score > 0.5 else 0
    label = class_labels.get(predicted_class, "Unknown")
    conf = score if predicted_class == 1 else (1 - score)
    return {"label": label, "prob": float(round(conf, 4))}



# =====================================================
# TEXT RESPONSE CLEANER (FIXED)
# =====================================================
def _extract_text_from_genai_response(resp):
    """Extract ONLY pure text. Remove all model metadata."""
    try:
        # Convert object → JSON
        if not isinstance(resp, (dict, list, str)):
            resp = getattr(resp, "text", None) or getattr(resp, "content", None) or str(resp)

        text = ""

        # If entire thing is string-like
        if isinstance(resp, str):
            text = resp

        # If dict with candidates
        elif isinstance(resp, dict):
            if "text" in resp:
                text = resp["text"]
            elif "content" in resp:
                text = resp["content"]
            elif "candidates" in resp and resp["candidates"]:
                cand = resp["candidates"][0]
                text = cand.get("content") or cand.get("text") or str(cand)
            elif "output" in resp and isinstance(resp["output"], list):
                first = resp["output"][0]
                text = first.get("content") or first.get("text") or str(first)
            else:
                text = str(resp)

        # If list
        elif isinstance(resp, list):
            text = resp[0] if resp else ""

        # Cleanup metadata
        text = (
            str(text)
            .replace("parts {", "")
            .replace("role: \"model\"", "")
            .replace("}", "")
            .strip()
        )

        return text

    except Exception:
        return str(resp)



# =====================================================
# Gemini Reasoning Text
# =====================================================
def gemini_reasoning_text(prompt, max_output_tokens=200):
    if not GEMINI_API_KEY or genai is None:
        return f"(Mock Gemini) {prompt}"

    try:
        clean_prompt = f"""
Respond only in short bullet points.
Do not use asterisks.
Use dash (-) for bullet points.
Keep points very short and crisp.
No long paragraphs.
No markdown.

User message:
{prompt}
"""

        model_obj = genai.GenerativeModel(GEMINI_MODEL)
        resp = model_obj.generate_content(
            clean_prompt,
            generation_config={
                "max_output_tokens": max_output_tokens,
                "temperature": 0.4,
                "response_mime_type": "text/plain"
            }
        )

        return _extract_text_from_genai_response(resp)

    except Exception as e:
        return f"(Gemini error) {e}"





# =====================================================
# CHAT
# =====================================================
chat_history_dict = {}  # store per-session history (simple memory, key by IP/session)

def gemini_chat_response(user_message, session_id=None):
    """Call Gemini API with short, crisp reply constraints"""
    if not GEMINI_API_KEY or genai is None:
        return f"(Mock Chat) {user_message}"

    try:
        # Fetch history
        history = chat_history_dict.get(session_id, [])

        # NEW: Add instruction for short crisp output
        system_instruction = (
            "You are an AI Quit Coach. "
            "Always reply in few sentences. "
            "Long answers should be written in bullet points. "
            "Keep it crisp, motivational, simple. "
            "Never use asterisks. "
            "Do NOT write long paragraphs."
        )

        prompt = (
            system_instruction + "\n\n"
            + "\n".join(history[-6:]) + "\n"
            + f"User: {user_message}\nBot:"
        )

        # Get response from Gemini
        reply = gemini_reasoning_text(prompt)

        # Remove **asterisks** if Gemini adds any
        reply = reply.replace("*", "").strip()

        # Update memory
        history.append(f"User: {user_message}")
        history.append(f"Bot: {reply}")
        chat_history_dict[session_id] = history

        return reply

    except Exception as e:
        return f"(Chat error) {e}"




# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    filename = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file selected.")
            return redirect(request.url)

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)
            result = predict_image(save_path)
            return render_template("predict.html", filename=filename, result=result)

        flash("Invalid file type.")
        return redirect(request.url)

    return render_template("predict.html", filename=filename, result=result)



@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)



@app.route("/ai", methods=["GET", "POST"])
def ai_features():
    result = None
    gen_text = None  # only text now

    if request.method == "POST":
        try:
            # Get values safely
            cpday = float(request.form.get("cigarettes_per_day", 0))
            years = float(request.form.get("years_smoked", 0))
            cost_per_pack = float(request.form.get("cost_per_pack", 0))
            cigs_per_pack = float(request.form.get("cigs_per_pack", 20))

            # Computations
            total_days = years * 365
            total_cigs = cpday * total_days
            packs = total_cigs / cigs_per_pack if cigs_per_pack > 0 else 0
            money_spent = packs * cost_per_pack

            # Prevent overflow
            risk_score = min(0.99, (cpday / 40) + (years / 30))

            # Diseases logic
            if cpday < 1:
                diseases = ["Lower risk"]
            else:
                diseases = [
                    "Lung Disease (COPD)",
                    "Heart Disease",
                    "Oral Cancer",
                    "Stroke"
                ]

            # Gemini text explanation
            prompt_text = (
                f"Explain health effects of smoking {int(cpday)} cigarettes per day "
                f"for {int(years)} years in very simple short words. "
                f"Make it readable for the public."
            )

            gen_text = gemini_reasoning_text(prompt_text)

            # Final result dictionary
            result = {
                "cigarettes_per_day": cpday,
                "years_smoked": years,
                "total_cigarettes": int(total_cigs),
                "packs_used": round(packs, 2),
                "money_spent": round(money_spent, 2),
                "risk_score": round(risk_score, 3),
                "diseases": diseases,
            }

        except Exception as e:
            flash(f"Invalid input: {e}")

    return render_template(
        "ai_features.html",
        result=result,
        gen_text=gen_text
    )



@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html")


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json(force=True) or {}
    msg = data.get("message", "")
    session_id = request.remote_addr  # simple session key
    reply = gemini_chat_response(msg, session_id=session_id)
    return jsonify({"reply": reply})



@app.route("/tracker", methods=["GET", "POST"])
def tracker():
    result = None

    if request.method == "POST":
        try:
            quit_date_str = request.form.get("quit_date")
            cpday = float(request.form.get("cigarettes_per_day_before", 0))
            cost_per_pack = float(request.form.get("cost_per_pack", 0))
            cigs_per_pack = float(request.form.get("cigs_per_pack", 20))

            quit_date = datetime.strptime(quit_date_str, "%Y-%m-%d").date()
            days_quit = (date.today() - quit_date).days

            cig_avoided = days_quit * cpday
            packs_saved = cig_avoided / cigs_per_pack
            money_saved = packs_saved * cost_per_pack
            life_days_added = int(days_quit * 0.02 * cpday)

            prompt = f"You quit smoking on {quit_date}. Give a short uplifting message."
            motivation = gemini_reasoning_text(prompt)

            result = {
                "days_quit": days_quit,
                "cigarettes_avoided": int(cig_avoided),
                "packs_saved": round(packs_saved, 2),
                "money_saved": round(money_saved, 2),
                "life_days_added": life_days_added,
                "motivation": motivation
            }

        except Exception as e:
            flash(f"Invalid input: {e}")

    return render_template("tracker.html", result=result)



@app.route("/status")
def status():
    return jsonify({
        "model_loaded": model is not None,
        "class_mapping": class_labels,
        "tensorflow_version": tf.__version__,
        "gemini_configured": bool(GEMINI_API_KEY and genai)
    })



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
