# AI-Based Automated Food Recognition and Price Measurement Application for Canteen Meal Trays

[GitHub Repository](https://github.com/KhaiNghiTranNguyen/AI-Challenge-3ITech)
![image](https://github.com/user-attachments/assets/79992170-a514-4a08-9e50-2f383cc6f48c)

![video](https://youtu.be/w-ruyauu5rc?si=OVgKPpun4WrvD4pU)

# 📋 Overview
An AI-powered application that automatically recognizes food items and calculates prices for canteen meal trays, built with YOLO and CNN models to accurately identify Vietnamese food dishes and calculate their prices and caloric values.
The system uses computer vision and deep learning technologies to automate the payment process in canteens. Through a single image of a food tray, the application can:


Detect and locate individual food items on the tray using YOLOv8

Accurately classify each food item from a set of 41 common Vietnamese dishes using CNN

Calculate the bill based on the recognized items, including pricing and caloric values

Provide a web interface for users to upload images and view results

# ✨ Key Features

- **🔍 Advanced Food Detection**: Uses YOLOv8 to locate and crop individual food items on a tray
- **🍲 Vietnamese Food Classification**: Employs a fine-tuned CNN model to identify 41 different Vietnamese food items
- **💰 Automated Billing**: Calculates the total cost based on detected items
- **🥗 Nutritional Analysis**: Provides calorie content and meal balance feedback
- **🖥️ Responsive Web Interface**: User-friendly design that works on both desktop and mobile devices
- **✏️ Manual Correction**: Allows for easy adjustment of misidentified items
- **📊 Transaction History**: Keeps records of past purchases
- **🌙 Dark/Light Mode**: Interface adapts to user preference

# 🛠️ Technology Stack
**Backend: Python, Flask
**Computer Vision: YOLOv8, OpenCV
**Machine Learning: TensorFlow/Keras with CNN
**Frontend: HTML, CSS, JavaScript
**Data Storage: CSV-based menu system

# 🔧 System Requirements

- Python 3.8, 3.9 or 3.10
- CUDA (recommended for GPU acceleration)
- Libraries listed in requirements.txt
- Web camera or image input source

# 🚀 Installation Guide
bash# Clone repository
git clone https://github.com/KhaiNghiTranNguyen/AI-Challenge-3ITech.git
cd AI-Challenge-3ITech

# Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AI-challenge-UEH-2025/vietnamese-canteen-vision.git
   cd vietnamese-canteen-vision
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the web server:
   ```bash
   python web_server.py
   ```

4. Access the interface at http://localhost:5000

# Docker Installation (Alternative)

```bash
# Build the Docker image
docker build -t canteenvision .

# Run the container
docker run -p 5000:5000 canteenvision
```


# 📊 Project Structure
```
food-recognition-canteen/
├── app.py                # Main Flask application
├── web_server.py         # Web server entry point
├── main.py               # CLI source code
├── requirements.txt      # Required libraries
├── src/
│   ├── detect.py         # Object detection module using YOLO
│   ├── classify.py       # Food classification module using CNN
│   ├── billing.py        # Bill calculation module
│   └── models/           # Directory for trained models
├── data/
│   ├── menu.csv          # Price and calorie data for food items
│   └── training/         # Training data
└── web_ui/
    ├── static/           # CSS, JavaScript, images
    └── templates/        # HTML templates
```


# 🧠 AI Models
The project uses two main AI models:
YOLOv8

For detecting and locating food items on the tray
Trained to recognize food containers and Vietnamese food items
Can accurately locate food items even if they overlap

Custom CNN

For accurate food classification
CNN model trained on a dataset of 41 Vietnamese food items
Provides higher accuracy in classifying individual food items

# 🍲 Supported Food Items
The system can recognize 41 common Vietnamese food items in canteens:

- Rice dishes: rice, banh mi (Vietnamese sandwich)

- Vegetables: boiled cabbage, stir-fried cabbage, tomatoes, carrots, okra, tofu, green beans, cucumber, chili, leafy greens, water spinach, coriander

- Meat dishes: stir-fried beef, fried chicken, braised chicken, pork ribs, stir-fried pork ribs, fried meat, boiled meat, fried eggs, boiled eggs

- Fish dishes: fried fish, braised fish, shrimp

- Soups and liquid dishes: gourd soup, pumpkin soup, vegetable soup, sour soup, seaweed soup, fish sauce, soy sauce

- Others: banana, watermelon, guava

# 💻 Usage
Upload a Food Tray Image:

Click "Select Image" or drag and drop an image of a food tray
Alternatively, use the "Use Camera" button to capture a live image
Analyze the Image:

Click "Analyze Food" to process the image
The system will detect and classify each food item
Review Results:

View detected items, prices, and calorie information
Make corrections to misidentified items if needed
Check the nutritional balance and suggestions
Complete the Order:

Click "Complete Order" to finalize and save the transaction

<div align="center">
  <p>Made with ❤️ for the AI Challenge 3ITECH 2025</p>
  <p>
    <a href="https://github.com/AI-challenge-UEH-2025/vietnamese-canteen-vision/issues">Report Bug</a> ·
    <a href="https://github.com/AI-challenge-UEH-2025/vietnamese-canteen-vision/issues">Request Feature</a>
  </p>
</div>

# 📝 License
This project is distributed under the MIT License. See LICENSE file for more information.
📞 Contact

Khai Nghi Tran Nguyen - GitHub Profile
Project Link: https://github.com/KhaiNghiTranNguyen/AI-Challenge-3ITech
