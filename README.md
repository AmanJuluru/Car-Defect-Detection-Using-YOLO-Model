# 🚗 Car Defect Detection Using YOLO Model

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3.x-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

**An AI-powered web portal for automobile exterior defect detection using deep learning**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Demo](#-demo) • [Tech Stack](#-tech-stack) • [Documentation](#-documentation)

</div>

---

## 📖 Overview

This project is a **secure web portal** designed for manufacturing inspection teams to detect exterior vehicle defects using state-of-the-art **YOLOv8** deep learning technology. The system enables quality assurance teams to upload vehicle images and receive instant AI-powered defect analysis with visual bounding box annotations.

### 🎯 Key Highlights

- **Real-time Detection**: Sub-second inference time for instant results
- **Offline Operation**: Fully functional without internet connectivity
- **Secure Access**: User authentication with encrypted password storage
- **Historical Tracking**: Complete audit trail of all inspections
- **Visual Results**: Color-coded bounding boxes for easy defect identification

---

## ✨ Features

### 🔍 Defect Detection
The pre-trained YOLO model can detect **5 types** of automobile exterior defects:

| Defect Type | Visual Indicator |
|-------------|------------------|
| 🔴 **Dent** | Pink bounding box |
| 🔵 **Scratch** | Blue bounding box |
| 🟡 **Lamp Broken** | Yellow bounding box |
| 🟣 **Glass Broken** | Purple bounding box |
| ⭕ **Tire Flat** | Red bounding box |

### 🛡️ User Management
- Secure user registration and login
- Password hashing with PBKDF2-SHA256
- Session-based authentication
- User profile management

### 📊 Dashboard & Analytics
- Real-time inspection statistics
- Visual breakdown of broken vs non-broken vehicles
- Recent inspection activity feed
- Quick access navigation

### 📜 Inspection History
- Complete log of all past inspections
- Original and annotated image storage
- Searchable detection records
- Timestamp tracking

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/AmanJuluru/Car-Defect-Detection-Using-YOLO-Model.git
cd Car-Defect-Detection-Using-YOLO-Model
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Model File

Ensure the YOLO model file exists at:
```
model/defect_model.pt
```

### Step 4: Run the Application

```bash
python app.py
```

The server will start at `http://127.0.0.1:5000`

---

## 💻 Usage

### 1. Register/Login
Create a new account or log in with existing credentials.

### 2. Upload Image
Navigate to the **Upload** section and either:
- Drag and drop a vehicle image
- Click to browse and select a file
- Supported formats: **JPG, JPEG, PNG**

### 3. View Results
After upload, the AI processes the image and displays:
- Annotated image with bounding boxes around defects
- Vehicle status: **Broken** or **Non-Broken**
- List of detected defects with confidence scores

### 4. Review History
Access the **History** section to view all past inspections.

---

## 🎬 Demo

### Detection Flow
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Upload Image   │────▶│  AI Processing  │────▶│  View Results   │
│  (JPG/PNG)      │     │  (YOLO Model)   │     │  (Annotated)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Sample Output

When defects are detected, the system:
1. Draws colored bounding boxes around each defect
2. Labels each box with: `defect_type | confidence%`
3. Marks the vehicle as **"Broken"**
4. Stores the result in the database

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Primary programming language |
| **Flask** | Lightweight web framework |
| **SQLite** | Local database for users and history |
| **Werkzeug** | Password hashing and security |

### Machine Learning
| Technology | Purpose |
|------------|---------|
| **Ultralytics YOLOv8** | Object detection framework |
| **OpenCV** | Image processing and annotation |
| **Custom Model** | `defect_model.pt` - Trained on automobile defects |

### Frontend
| Technology | Purpose |
|------------|---------|
| **HTML5 / CSS3** | Structure and styling |
| **JavaScript** | Client-side interactivity |
| **Jinja2** | Server-side templating |

---

## 📁 Project Structure

```
project/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── database.db                 # SQLite database (auto-generated)
├── model/
│   └── defect_model.pt         # Pre-trained YOLO model
├── static/
│   ├── css/
│   │   └── style.css           # Main stylesheet
│   ├── uploads/                # User uploaded images
│   └── results/                # Processed images with annotations
├── templates/
│   ├── base.html               # Base template (navbar, layout)
│   ├── login.html              # Login page
│   ├── register.html           # Registration page
│   ├── dashboard.html          # Main dashboard
│   ├── upload.html             # Image upload and results
│   ├── history.html            # Inspection history
│   └── profile.html            # User profile
├── DOCUMENTATION.md            # Detailed technical documentation
└── README.md                   # This file
```

---

## 🔐 Security Features

| Feature | Implementation |
|---------|----------------|
| **Password Hashing** | PBKDF2-SHA256 algorithm |
| **Session Security** | Flask secure sessions with cryptographic signing |
| **Data Isolation** | Users can only access their own data |
| **Input Validation** | Server-side validation for all inputs |

---

## 📊 API Routes

| Route | Method | Description | Auth Required |
|-------|--------|-------------|---------------|
| `/` | GET | Redirect to dashboard/login | No |
| `/login` | GET, POST | User login | No |
| `/register` | GET, POST | User registration | No |
| `/logout` | GET | Clear session | Yes |
| `/dashboard` | GET | Main dashboard | Yes |
| `/upload` | GET, POST | Image upload & detection | Yes |
| `/history` | GET | View past inspections | Yes |
| `/profile` | GET, POST | User profile & settings | Yes |

---

## 📋 Requirements

### Minimum Hardware
- **CPU**: Intel Core i5 (7th Gen) or equivalent
- **RAM**: 8 GB DDR4
- **Storage**: 2 GB free disk space

### Software Dependencies
See [requirements.txt](requirements.txt) for the complete list of dependencies:
```
Flask>=2.3.0
ultralytics>=8.0.0
opencv-python>=4.8.0
Werkzeug>=2.3.0
```

---

## 📖 Documentation

For detailed technical documentation including:
- Complete API reference
- Database schema
- YOLO model architecture
- Security implementation
- UI/UX design system

Please refer to [DOCUMENTATION.md](DOCUMENTATION.md)

---

## 🔮 Future Enhancements

- [ ] GPU acceleration for faster inference
- [ ] Batch processing for multiple images
- [ ] REST API for external integrations
- [ ] Mobile application
- [ ] PDF report generation
- [ ] Real-time video stream analysis

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Aman Juluru**

- GitHub: [@AmanJuluru](https://github.com/AmanJuluru)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for the Automobile Industry

</div>
