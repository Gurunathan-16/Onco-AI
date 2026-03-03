from flask import Flask, render_template, request, session, redirect, url_for
import os
import sqlite3
from tensorflow.keras.models import load_model
from preprocess import preprocess_image
from database import init_db, save_prediction
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "super_secret_key_change_this"

# ==============================
# 🔹 DATABASE INITIALIZATION
# ==============================

init_db()


def create_admin():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    username = "doctor"
    password = generate_password_hash("1234")

    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )
        conn.commit()
        print("Admin created successfully")
    except:
        print("Admin already exists")

    conn.close()


# 👉 Run once, then you can remove it
create_admin()


# ==============================
# 🔹 LOAD MODELS
# ==============================

breast_model = load_model(r"D:\onco-ai\models\breast_model.h5")
oral_model = load_model(r"D:\onco-ai\models\oral_model.h5")

UPLOAD_FOLDER = r"D:\onco-ai\static\uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==============================
# 🔐 LOGIN REQUIRED DECORATOR
# ==============================

def login_required(f):
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


# ==============================
# 🔹 MAIN PREDICTION PAGE
# ==============================

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    result = ""
    confidence = ""
    image_name = None

    if request.method == "POST":
        file = request.files["file"]
        model_type = request.form["model"]

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = preprocess_image(filepath)

            if model_type == "breast":
                prediction = breast_model.predict(img)[0][0]
                confidence = round(float(prediction) * 100, 2)
                result = "Malignant" if prediction > 0.5 else "Benign"

            elif model_type == "oral":
                prediction = oral_model.predict(img)[0][0]
                confidence = round(float(prediction) * 100, 2)
                result = "Cancer" if prediction > 0.5 else "Normal"

            save_prediction(file.filename, model_type, result, confidence)
            image_name = file.filename

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image=image_name
    )


# ==============================
# 🔐 LOGIN
# ==============================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("predictions.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["user_id"] = user[0]
            session["username"] = user[1]
            return redirect("/")

        return "Invalid username or password"

    return render_template("login.html")


# ==============================
# 📊 ANALYTICS PAGE
# ==============================

@app.route("/analytics")
@login_required
def analytics():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE model_type='breast'")
    breast_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE model_type='oral'")
    oral_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM predictions 
        WHERE result='Cancer' OR result='Malignant'
    """)
    cancer_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM predictions 
        WHERE result='Normal' OR result='Benign'
    """)
    normal_count = cursor.fetchone()[0]

    conn.close()

    return render_template(
        "analytics.html",
        total=total,
        breast=breast_count,
        oral=oral_count,
        cancer=cancer_count,
        normal=normal_count
    )

# ==============================
# 🔓 LOGOUT
# ==============================

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ==============================
# 🚀 RUN APP
# ==============================

if __name__ == "__main__":
    app.run()