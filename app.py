"""
Automobile Defect Detection Portal
===================================
A secure web portal for manufacturing inspection that integrates a YOLO-based
defect detection model, allowing authenticated users to upload images, view
defect localization results, and maintain a historical inspection log.

Technical Stack:
- Backend: Flask (Python web framework)
- ML Model: YOLO (Ultralytics) for defect detection
- Database: SQLite for user management and detection history
- Frontend: HTML + CSS (no external frameworks)

Author: Manufacturing Quality Systems
Version: 1.0.0
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash
from ultralytics import YOLO
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import sqlite3
import os
import cv2
import uuid

# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

app = Flask(__name__)
app.secret_key = "automobile_defect_portal_secret_key_2024"

# Directory paths for file storage
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
MODEL_PATH = "model/defect_model.pt"
DATABASE_PATH = "database.db"

# Create required directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the YOLO model for defect detection
# The model is trained to detect: dent, scratch, lamp_broken, glass_broken, tire_flat
model = YOLO(MODEL_PATH)

# Color mapping for each defect class (BGR format for OpenCV)
# These colors are used to draw bounding boxes on detected defects
DEFECT_COLORS = {
    "dent": (203, 192, 255),        # Pink
    "scratch": (255, 0, 0),          # Blue
    "lamp_broken": (0, 255, 255),    # Yellow
    "glass_broken": (128, 0, 128),   # Purple
    "tire_flat": (0, 0, 255)         # Red
}

# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

def init_database():
    """
    Initialize the SQLite database with required tables.
    
    Tables created:
    1. users - Stores user credentials (username, hashed password)
    2. detection_history - Stores detection results for each user
    
    This function is called at application startup to ensure
    the database schema exists.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create users table for authentication
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create detection_history table for storing inspection results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            original_image TEXT NOT NULL,
            result_image TEXT NOT NULL,
            vehicle_status TEXT NOT NULL,
            defect_classes TEXT,
            confidence_scores TEXT,
            detection_count INTEGER DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    """
    Create and return a database connection.
    Uses row_factory to enable dictionary-like access to rows.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database on startup
init_database()

# =============================================================================
# AUTHENTICATION ROUTES
# =============================================================================

@app.route("/")
def home():
    """
    Home route - redirects to dashboard if logged in, otherwise to login page.
    """
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    User login endpoint.
    
    GET: Display the login form
    POST: Validate credentials and create session
    
    Security: Passwords are verified using werkzeug's check_password_hash
    which implements secure comparison against stored hash.
    """
    # Redirect to dashboard if already logged in
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        # Validate input
        if not username or not password:
            flash("Please enter both username and password.", "error")
            return render_template("login.html")
        
        # Query database for user
        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        conn.close()
        
        # Verify credentials
        if user and check_password_hash(user["password_hash"], password):
            # Create session for authenticated user
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {username}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "error")
    
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """
    User registration endpoint.
    
    GET: Display the registration form
    POST: Create new user account with hashed password
    
    Security: Passwords are hashed using werkzeug's generate_password_hash
    which uses PBKDF2-SHA256 by default.
    """
    # Redirect to dashboard if already logged in
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        
        # Validate input
        if not username or not password:
            flash("Please fill in all fields.", "error")
            return render_template("register.html")
        
        if len(username) < 3:
            flash("Username must be at least 3 characters long.", "error")
            return render_template("register.html")
        
        if len(password) < 4:
            flash("Password must be at least 4 characters long.", "error")
            return render_template("register.html")
        
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html")
        
        # Check if username already exists
        conn = get_db_connection()
        existing_user = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        
        if existing_user:
            conn.close()
            flash("Username already exists. Please choose another.", "error")
            return render_template("register.html")
        
        # Create new user with hashed password
        password_hash = generate_password_hash(password)
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash)
        )
        conn.commit()
        conn.close()
        
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))
    
    return render_template("register.html")


@app.route("/logout")
def logout():
    """
    Log out the current user by clearing the session.
    """
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("login"))

# =============================================================================
# MAIN PORTAL ROUTES
# =============================================================================

@app.route("/dashboard")
def dashboard():
    """
    Main dashboard page for authenticated users.
    
    Displays:
    - Welcome message with username
    - Quick statistics (total inspections, defects found)
    - Recent inspection activity
    """
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    # Get user statistics from database
    conn = get_db_connection()
    
    # Total inspections by this user
    total_inspections = conn.execute(
        "SELECT COUNT(*) as count FROM detection_history WHERE user_id = ?",
        (session["user_id"],)
    ).fetchone()["count"]
    
    # Count of broken vehicles detected
    broken_count = conn.execute(
        "SELECT COUNT(*) as count FROM detection_history WHERE user_id = ? AND vehicle_status = 'Broken'",
        (session["user_id"],)
    ).fetchone()["count"]
    
    # Count of non-broken vehicles
    non_broken_count = conn.execute(
        "SELECT COUNT(*) as count FROM detection_history WHERE user_id = ? AND vehicle_status = 'Non-Broken'",
        (session["user_id"],)
    ).fetchone()["count"]
    
    # Recent inspections (last 5)
    recent_inspections = conn.execute(
        """SELECT * FROM detection_history 
           WHERE user_id = ? 
           ORDER BY timestamp DESC 
           LIMIT 5""",
        (session["user_id"],)
    ).fetchall()
    
    conn.close()
    
    return render_template(
        "dashboard.html",
        username=session["username"],
        total_inspections=total_inspections,
        broken_count=broken_count,
        non_broken_count=non_broken_count,
        recent_inspections=recent_inspections
    )


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """
    Image upload and defect detection endpoint.
    
    GET: Display the upload form
    POST: Process uploaded image through YOLO model
    
    Detection Process:
    1. Save uploaded image to uploads folder
    2. Run YOLO inference on the image
    3. Draw bounding boxes on detected defects
    4. Save annotated image to results folder
    5. Store detection record in database
    6. Display results to user
    """
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    result_data = None
    
    if request.method == "POST":
        file = request.files.get("image")
        
        if not file or file.filename == "":
            flash("Please select an image file to upload.", "error")
            return render_template("upload.html", username=session["username"])
        
        # Validate file type
        allowed_extensions = {"jpg", "jpeg", "png"}
        file_ext = file.filename.rsplit(".", 1)[-1].lower()
        
        if file_ext not in allowed_extensions:
            flash("Invalid file type. Please upload JPG or PNG images only.", "error")
            return render_template("upload.html", username=session["username"])
        
        # Generate unique filename to prevent conflicts
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session['username']}_{timestamp}_{unique_id}.{file_ext}"
        
        # Save paths
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        result_path = os.path.join(RESULT_FOLDER, filename)
        
        # Save the uploaded file
        file.save(image_path)
        
        # Run YOLO inference
        # conf=0.05 is a low threshold to detect subtle defects
        results = model(image_path, conf=0.05)[0]
        
        # Read image for annotation
        img = cv2.imread(image_path)
        
        # Process detection results
        detections = []
        defect_classes = []
        confidence_scores = []
        vehicle_status = "Non-Broken"
        
        if len(results.boxes) > 0:
            vehicle_status = "Broken"
            
            for box in results.boxes:
                # Extract detection information
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                
                # Store detection data
                defect_classes.append(class_name)
                confidence_scores.append(f"{conf:.2%}")
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get color for this defect class
                color = DEFECT_COLORS.get(class_name, (0, 255, 0))
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                
                # Draw label background
                label = f"{class_name} | {conf:.0%}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    img, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1),
                    color, -1
                )
                
                # Draw label text
                cv2.putText(
                    img, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                
                detections.append({
                    "class": class_name,
                    "confidence": f"{conf:.2%}"
                })
        
        # Save annotated image
        cv2.imwrite(result_path, img)
        
        # Store detection in database
        conn = get_db_connection()
        conn.execute(
            """INSERT INTO detection_history 
               (user_id, original_image, result_image, vehicle_status, 
                defect_classes, confidence_scores, detection_count) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session["user_id"],
                filename,
                filename,
                vehicle_status,
                ", ".join(defect_classes) if defect_classes else "None",
                ", ".join(confidence_scores) if confidence_scores else "N/A",
                len(detections)
            )
        )
        conn.commit()
        conn.close()
        
        # Prepare result data for template
        result_data = {
            "result_image": result_path,
            "original_image": image_path,
            "vehicle_status": vehicle_status,
            "detections": detections,
            "detection_count": len(detections)
        }
        
        flash("Image processed successfully!", "success")
    
    return render_template(
        "upload.html",
        username=session["username"],
        result_data=result_data
    )


@app.route("/history")
def history():
    """
    Detection history page.
    
    Displays all past inspections for the current user with:
    - Thumbnail of processed image
    - Vehicle status (Broken/Non-Broken)
    - Detected defect classes
    - Confidence scores
    - Timestamp of inspection
    """
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    # Get all detection records for current user
    conn = get_db_connection()
    records = conn.execute(
        """SELECT * FROM detection_history 
           WHERE user_id = ? 
           ORDER BY timestamp DESC""",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    
    return render_template(
        "history.html",
        username=session["username"],
        records=records
    )

# =============================================================================
# PROFILE ROUTE
# =============================================================================

@app.route("/profile", methods=["GET", "POST"])
def profile():
    """
    User profile page.
    
    Displays:
    - User information (username, account creation date)
    - Inspection statistics
    - Password change functionality
    """
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    # Get user information
    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE id = ?", (session["user_id"],)
    ).fetchone()
    
    # Get user statistics
    total_inspections = conn.execute(
        "SELECT COUNT(*) as count FROM detection_history WHERE user_id = ?",
        (session["user_id"],)
    ).fetchone()["count"]
    
    broken_count = conn.execute(
        "SELECT COUNT(*) as count FROM detection_history WHERE user_id = ? AND vehicle_status = 'Broken'",
        (session["user_id"],)
    ).fetchone()["count"]
    
    non_broken_count = conn.execute(
        "SELECT COUNT(*) as count FROM detection_history WHERE user_id = ? AND vehicle_status = 'Non-Broken'",
        (session["user_id"],)
    ).fetchone()["count"]
    
    # Handle password change
    if request.method == "POST":
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")
        
        # Validate current password
        if not check_password_hash(user["password_hash"], current_password):
            flash("Current password is incorrect.", "error")
        elif len(new_password) < 4:
            flash("New password must be at least 4 characters long.", "error")
        elif new_password != confirm_password:
            flash("New passwords do not match.", "error")
        else:
            # Update password
            new_hash = generate_password_hash(new_password)
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (new_hash, session["user_id"])
            )
            conn.commit()
            flash("Password updated successfully!", "success")
    
    conn.close()
    
    return render_template(
        "profile.html",
        username=session["username"],
        user=user,
        total_inspections=total_inspections,
        broken_count=broken_count,
        non_broken_count=non_broken_count
    )

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run the Flask development server
    # debug=True enables auto-reload and detailed error messages
    print("=" * 60)
    print("Automobile Defect Detection Portal")
    print("=" * 60)
    print("Starting server at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=True)
