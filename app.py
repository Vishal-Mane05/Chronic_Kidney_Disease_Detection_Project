from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import matplotlib.cm as cm
import os
import json
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt


from groq import Groq

import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print("Failed to initialize Groq Client:", e)
    groq_client = None

app = Flask(__name__)

# --- DATABASE & SECURITY CONFIGURATION ---
app.config['SECRET_KEY'] = os.urandom(24) 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///hospital_records.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ─── DATABASE MODELS ──────────────────────────
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    clinical_records = db.relationship('ClinicalRecord', foreign_keys='ClinicalRecord.user_id', backref='owner', lazy=True)
    ctscan_records = db.relationship('CTScanRecord', foreign_keys='CTScanRecord.user_id', backref='owner', lazy=True)
    ledger_patients = db.relationship('DoctorPatient', backref='treating_doctor', lazy=True)

class DoctorPatient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=True)
    clinical_records = db.relationship('ClinicalRecord', foreign_keys='ClinicalRecord.doctor_patient_id', backref='doctor_file', lazy=True)
    ctscan_records = db.relationship('CTScanRecord', foreign_keys='CTScanRecord.doctor_patient_id', backref='doctor_file', lazy=True)

class ClinicalRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_tested = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    doctor_patient_id = db.Column(db.Integer, db.ForeignKey('doctor_patient.id'), nullable=True)
    input_data = db.Column(db.Text, nullable=False) 
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    doctor_notes = db.Column(db.Text, nullable=True)
    is_verified = db.Column(db.Boolean, default=False)

class CTScanRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_tested = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    doctor_patient_id = db.Column(db.Integer, db.ForeignKey('doctor_patient.id'), nullable=True)
    image_path = db.Column(db.String(200), nullable=False)
    heatmap_path = db.Column(db.String(200), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    class_probabilities = db.Column(db.Text, nullable=False)
    doctor_notes = db.Column(db.Text, nullable=True)
    is_verified = db.Column(db.Boolean, default=False)

with app.app_context():
    db.create_all()

# ─── ML MODELS & GRAD-CAM ───────────────
rf      = joblib.load("models/ckd_rf_model.pkl")
scaler  = joblib.load("models/ckd_scaler.pkl")
ohe     = joblib.load("models/ckd_ohe.pkl")
num_imp = joblib.load("models/ckd_num_imputer.pkl")
cat_imp = joblib.load("models/ckd_cat_imputer.pkl")
cnn     = tf.keras.models.load_model("models/ckd_resnet50.keras")

CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']
NUM_COLS = ['age','bloodpressure','specificgravity','albumin','sugar','bloodglucoserandom','bloodurea','serumcreatinine','sodium','potassium','hemoglobin','packedellvolume','wbccount','rbccount']
CAT_COLS = ['redbloodcells','puscells','puscellsclumps','bacteriapresence','hypertension','diabetesmellitus','coronaryarterydisease','appetite','pedaledema','anemia']
SCALER_FEATURES = scaler.feature_names_in_

def get_last_conv_layer(model):
    inner_model = model
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            inner_model = layer
            break
    for layer in reversed(inner_model.layers):
        try:
            output = layer.output
            if isinstance(output, list): output = output[0]
            if len(output.shape) == 4: return layer.name
        except AttributeError: continue 
    return 'conv5_block3_out' 

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    inner_model = model
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            inner_model = layer
            break
    grad_model = tf.keras.models.Model(inner_model.inputs, [inner_model.get_layer(last_conv_layer_name).output, inner_model.output])
    with tf.GradientTape() as tape:
        outputs = grad_model(img_array)
        last_conv_layer_output = outputs[0]
        preds = outputs[1]
        if isinstance(last_conv_layer_output, list): last_conv_layer_output = last_conv_layer_output[0]
        if isinstance(preds, list): preds = preds[0]
        if pred_index is None: pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None: return np.zeros((img_array.shape[1], img_array.shape[2]))
    if isinstance(grads, list): grads = grads[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0] 
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)

# ─── NEW GENERATIVE AI SCRIBE (GROQ) ──────────────────────────────
def generate_medical_report(prediction, raw_data_dict):
    """Feeds the raw data to Llama-3 via Groq to write a bespoke medical summary."""
    if not groq_client:
        return "<div class='alert alert-danger fw-bold'>Groq Client failed to initialize. Check API key.</div>"
        
    try:
        prompt = f"""
        Act as an empathetic but highly professional nephrologist. 
        I have a patient whose Predictive AI diagnosis is: '{prediction}'.
        Here are their current clinical parameters:
        {raw_data_dict}
        
        Write a personalized, 3-paragraph clinical summary for their medical file. 
        - Paragraph 1: State the diagnosis and gently explain what it means.
        - Paragraph 2: Identify the 2 or 3 most abnormal/concerning parameters from their data. Explain physiologically *why* these specific numbers are out of bounds and how they are causing kidney stress.
        - Paragraph 3: Suggest 3 highly actionable, specific lifestyle or dietary changes they can make immediately to protect their kidneys.
        
        Format the output in clean HTML using <p>, <ul>, <li>, and <strong> tags so it renders beautifully on a web page. Do NOT include ```html markdown blocks. Do not include greetings like 'Dear Patient'.
        """
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"<div class='alert alert-danger fw-bold'><i class='fas fa-exclamation-triangle'></i> Groq AI Error: {str(e)}</div>"

# ─── RADIOLOGY AI SCRIBE (GROQ) ──────────────────────────────
def generate_radiology_report(prediction):
    """Feeds the CT Scan prediction to Llama-3 to write a radiology summary with causes and cures."""
    if not groq_client:
        return "<div class='alert alert-danger fw-bold'>Groq Client failed to initialize. Check API key.</div>"
    try:
        prompt = f"""
        Act as an empathetic but highly professional radiologist and nephrologist. 
        I have a patient whose Kidney CT Scan AI prediction is: '{prediction}'.
        
        Write a personalized, 3-paragraph radiology summary for their medical file. 
        - Paragraph 1: State the diagnosis clearly and gently explain what it means anatomically.
        - Paragraph 2: Explain the common *causes* or risk factors for developing this specific condition.
        - Paragraph 3: Suggest the standard medical *treatments, cures, or next clinical steps* to address it. (If the diagnosis is 'Normal', suggest 3 lifestyle tips to maintain healthy kidneys).
        
        Format the output in clean HTML using <p>, <ul>, <li>, and <strong> tags so it renders beautifully on a web page. Do NOT include ```html markdown blocks. Do not include greetings like 'Dear Patient'.
        """
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"<div class='alert alert-danger fw-bold'><i class='fas fa-exclamation-triangle'></i> Groq AI Error: {str(e)}</div>"

def get_ctscan_insights(name):
    insights = []
    if name == "Normal":
        insights.append({"factor": "Cortical Tissue", "value": "Uniform", "desc": "The heatmap shows an even density distribution across the kidney cortex, indicating healthy, normal-functioning renal tissue."})
        insights.append({"factor": "Structural Integrity", "value": "Intact", "desc": "No abnormal masses, calcifications, or fluid-filled pockets were detected by the deep learning model."})
    elif name == "Cyst":
        insights.append({"factor": "Visual Signature", "value": "Fluid Sac", "desc": "The AI highlighted a well-defined, circular area with smooth borders, characteristic of a benign fluid-filled cyst."})
        insights.append({"factor": "Density Analysis", "value": "Low Density", "desc": "The internal density of the detected region matches fluid rather than solid, dangerous tissue."})
    elif name == "Stone":
        insights.append({"factor": "Visual Signature", "value": "Calcification", "desc": "The heatmap isolated a highly dense, bright focal point indicating crystallized minerals (kidney stone)."})
        insights.append({"factor": "Obstruction Risk", "value": "Elevated", "desc": "Depending on size and location, hard mineral deposits can obstruct the urinary tract and cause severe discomfort."})
    elif name == "Tumor":
        insights.append({"factor": "Visual Signature", "value": "Irregular Mass", "desc": "The heatmap highlights an area with uneven, jagged borders and mixed tissue density, suggesting abnormal cellular growth."})
        insights.append({"factor": "Clinical Action", "value": "Urgent", "desc": "This structural abnormality requires immediate follow-up with a nephrologist or oncologist for formal biopsy and diagnosis."})
    return insights


# ─── AUTH & DASHBOARD ROUTES ───────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful.', 'danger')
    return render_template('login.html')

@app.route("/register", methods=['POST'])
def register():
    name, email, password, role = request.form.get('name'), request.form.get('email'), request.form.get('password'), request.form.get('role')
    if User.query.filter_by(email=email).first():
        flash('Email registered.', 'warning')
        return redirect(url_for('login'))
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(name=name, email=email, password=hashed_password, role=role)
    db.session.add(new_user)
    db.session.commit()
    return redirect(url_for('login'))

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route("/dashboard")
@login_required
def dashboard():
    if current_user.role == 'doctor':
        patients = DoctorPatient.query.filter_by(doctor_id=current_user.id).all()
        return render_template('doctor_dashboard.html', patients=patients)
    else:
        clinical_history = ClinicalRecord.query.filter_by(user_id=current_user.id).order_by(ClinicalRecord.date_tested.desc()).all()
        ctscan_history = CTScanRecord.query.filter_by(user_id=current_user.id).order_by(CTScanRecord.date_tested.desc()).all()
        for record in clinical_history: record.input_data = json.loads(record.input_data)
        for record in ctscan_history: record.class_probabilities = json.loads(record.class_probabilities)
        return render_template('patient_dashboard.html', clinical_history=clinical_history, ctscan_history=ctscan_history)

@app.route("/enroll_patient", methods=["POST"])
@login_required
def enroll_patient():
    name, email = request.form.get('name'), request.form.get('email')
    new_patient = DoctorPatient(name=name, email=email, doctor_id=current_user.id)
    db.session.add(new_patient)
    db.session.commit()
    flash(f"{name} added to records.", "success")
    return redirect(url_for('dashboard'))

@app.route("/patient/<int:patient_id>")
@login_required
def view_patient(patient_id):
    patient = DoctorPatient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id: return redirect(url_for('dashboard'))
    clinical_history = ClinicalRecord.query.filter_by(doctor_patient_id=patient.id).order_by(ClinicalRecord.date_tested.desc()).all()
    ctscan_history = CTScanRecord.query.filter_by(doctor_patient_id=patient.id).order_by(CTScanRecord.date_tested.desc()).all()
    for r in clinical_history: r.input_data = json.loads(r.input_data)
    for r in ctscan_history: r.class_probabilities = json.loads(r.class_probabilities)
    return render_template('view_patient.html', patient=patient, clinical_history=clinical_history, ctscan_history=ctscan_history)

# ─── SECURED PREDICTION ROUTES ──────────────────────────────

@app.route("/clinical")
@login_required
def clinical():
    p_id = request.args.get('patient_id')
    return render_template("tabular.html", patient_id=p_id)

@app.route("/ctscan")
@login_required
def ctscan():
    p_id = request.args.get('patient_id')
    return render_template("ctscan.html", patient_id=p_id)

@app.route("/predict/tabular", methods=["POST"])
@login_required
def predict_tabular():
    try:
        num_data = [float(request.form.get(col, 0)) for col in NUM_COLS]
        num_df = pd.DataFrame(num_imp.transform(pd.DataFrame([num_data], columns=NUM_COLS)), columns=NUM_COLS)
        cat_data = [request.form.get(col, '') for col in CAT_COLS]
        cat_df = pd.DataFrame(cat_imp.transform(pd.DataFrame([cat_data], columns=CAT_COLS)), columns=CAT_COLS)
        encoded_df = pd.DataFrame(ohe.transform(cat_df), columns=ohe.get_feature_names_out(CAT_COLS))
        final_df = pd.concat([num_df, encoded_df], axis=1)
        
        for col in SCALER_FEATURES:
            if col not in final_df.columns: final_df[col] = 0
            
        prediction = rf.predict(scaler.transform(final_df[SCALER_FEATURES]))[0]
        confidence = round(max(rf.predict_proba(scaler.transform(final_df[SCALER_FEATURES]))[0]) * 100, 2)
        res = "CKD Detected" if prediction == 1 else "No CKD"

        # Generate Insights via Groq Llama-3
        raw_data_dict = dict(zip(NUM_COLS + CAT_COLS, num_data + cat_data))
        ai_report = generate_medical_report(res, raw_data_dict)

        target_p_id = request.form.get('patient_id')
        if current_user.role == 'doctor' and target_p_id:
            db.session.add(ClinicalRecord(doctor_patient_id=int(target_p_id), input_data=json.dumps(num_data + cat_data), prediction=res, confidence=confidence))
            db.session.commit()
        else:
            db.session.add(ClinicalRecord(user_id=current_user.id, input_data=json.dumps(num_data + cat_data), prediction=res, confidence=confidence))
            db.session.commit()
        
        return render_template("result.html", mode="tabular", result=res, status="danger" if prediction == 1 else "success", confidence=confidence, ai_report=ai_report, raw_data=raw_data_dict)
    except Exception as e: 
        return render_template("result.html", mode="error", result=str(e))

@app.route("/predict/ctscan", methods=["POST"])
@login_required
def predict_ctscan():
    try:
        file = request.files["ctscan"]
        filepath, heatmap_path = f"static/uploads/{file.filename}", f"static/uploads/heatmap_{file.filename}"
        os.makedirs("static/uploads", exist_ok=True)
        file.save(filepath)
        
        img_arr = np.expand_dims(preprocess_input(np.array(Image.open(filepath).convert("RGB").resize((224, 224)), dtype=np.float32)), axis=0)
        preds = cnn.predict(img_arr)[0]
        idx = np.argmax(preds)
        name, conf = CLASS_NAMES[idx], round(float(preds[idx]) * 100, 2)
        save_and_display_gradcam(filepath, make_gradcam_heatmap(img_arr, cnn, get_last_conv_layer(cnn), pred_index=idx), heatmap_path)

        insights = get_ctscan_insights(name)
        
        # --- NEW: Generate Radiology Report ---
        ai_report = generate_radiology_report(name)

        target_p_id = request.form.get('patient_id')
        if current_user.role == 'doctor' and target_p_id:
            db.session.add(CTScanRecord(doctor_patient_id=int(target_p_id), image_path=filepath, heatmap_path=heatmap_path, prediction=name, confidence=conf, class_probabilities=json.dumps({CLASS_NAMES[i]: round(float(preds[i]) * 100, 2) for i in range(4)})))
            db.session.commit()
        else:
            db.session.add(CTScanRecord(user_id=current_user.id, image_path=filepath, heatmap_path=heatmap_path, prediction=name, confidence=conf, class_probabilities=json.dumps({CLASS_NAMES[i]: round(float(preds[i]) * 100, 2) for i in range(4)})))
            db.session.commit()
            
        # Notice ai_report=ai_report is added to the render_template below!
        return render_template("result.html", mode="ctscan", result=name, status="success" if name=="Normal" else "danger", confidence=conf, all_probs={CLASS_NAMES[i]: round(float(preds[i])*100,2) for i in range(4)}, image_path=filepath, heatmap_path=heatmap_path, insights=insights, ai_report=ai_report)
    except Exception as e: 
        return render_template("result.html", mode="error", result=str(e))

# ─── VIEW HISTORY ROUTES ──────────────────────────────

@app.route("/record/clinical/<int:record_id>")
@login_required
def view_clinical_record(record_id):
    record = ClinicalRecord.query.get_or_404(record_id)
    if current_user.role == 'patient' and record.user_id != current_user.id:
        return redirect(url_for('dashboard'))
    
    data_list = json.loads(record.input_data)
    num_data = data_list[:14]
    cat_data = data_list[14:]
    
    raw_data_dict = dict(zip(NUM_COLS + CAT_COLS, data_list))
    ai_report = generate_medical_report(record.prediction, raw_data_dict)
    
    return render_template("result.html", mode="tabular", result=record.prediction, status="danger" if record.prediction == "CKD Detected" else "success", confidence=record.confidence, ai_report=ai_report, raw_data=raw_data_dict)

@app.route("/record/ctscan/<int:record_id>")
@login_required
def view_ctscan_record(record_id):
    record = CTScanRecord.query.get_or_404(record_id)
    if current_user.role == 'patient' and record.user_id != current_user.id:
        return redirect(url_for('dashboard'))
        
    insights = get_ctscan_insights(record.prediction)
    all_probs = json.loads(record.class_probabilities)
    
    # --- NEW: Generate Radiology Report for History ---
    ai_report = generate_radiology_report(record.prediction)
    
    return render_template("result.html", mode="ctscan", result=record.prediction, status="success" if record.prediction=="Normal" else "danger", confidence=record.confidence, all_probs=all_probs, image_path=record.image_path, heatmap_path=record.heatmap_path, insights=insights, ai_report=ai_report)

@app.route("/about-model")
def about_model(): return render_template("about_model.html")

@app.route("/about-ckd")
def about_ckd(): return render_template("about_ckd.html")

if __name__ == "__main__":
    app.run(debug=True)