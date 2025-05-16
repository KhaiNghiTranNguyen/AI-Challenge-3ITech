# AI-Based Automated Food Recognition and Price Measurement Application for Canteen Meal Trays
An AI-powered application that automatically recognizes food items and calculates prices for canteen meal trays, built with YOLO and CNN models to accurately identify Vietnamese food dishes and calculate their prices and caloric values.
GitHub Repository
# 📋 Overview
The system uses computer vision and deep learning technologies to automate the payment process in canteens. Through a single image of a food tray, the application can:

- Detect and locate individual food items on the tray using YOLOv8
- Accurately classify each food item from a set of 41 common Vietnamese dishes using CNN
- Calculate the bill based on the recognized items, including pricing and caloric values
- Provide a web interface for users to upload images and view results

# ✨ Key Features
Multi-food recognition: Can identify multiple food items on a single tray
Accurate classification: Uses CNN for high accuracy in classification
Automatic calculation: Automatically calculates total price and calories
RESTful API: Provides APIs for image analysis and detailed results
Web user interface: Intuitive interface for uploading images and displaying results
Manual edit support: Allows users to adjust classification results if needed

# 🚀 Installation
System Requirements

Python 3.8 or higher
CUDA (recommended for GPU acceleration)


# 🏃‍♂️ Usage
Running the Web Application
bashpython web_server.py
Access the web application through your browser at: http://localhost:5001
Using from Command Line
bashpython main.py --image path/to/your/food_tray_image.jpg
Using the API
Main API: POST /api/analyze
Example using cURL:
bashcurl -X POST -F "image=@path/to/your/food_tray_image.jpg" http://localhost:5001/api/analyze
Or send base64 data:
bashcurl -X POST -d "imageData=base64_encoded_image_data" http://localhost:5001/api/analyze

# 📂 Project Structure
food-recognition-canteen/
├── app.py                  # Main Flask application
├── web_server.py           # Web server entry point
├── main.py                 # CLI source code
├── requirements.txt        # Required libraries
├── src/
│   ├── detect.py           # Object detection module using YOLO
│   ├── classify.py         # Food classification module using CNN
│   ├── billing.py          # Bill calculation module
│   └── models/             # Directory for trained models
├── data/
│   ├── menu.csv            # Price and calorie data for food items
│   └── training/           # Training data
└── web_ui/
    ├── static/             # CSS, JavaScript, images
    └── templates/          # HTML templates
    
# 🧠 AI Models
The project uses two main AI models:

YOLOv8: For detecting and locating food items on the tray

Trained to recognize food containers and Vietnamese food items
Can accurately locate food items even if they overlap


Custom CNN: For accurate food classification

CNN model trained on a dataset of 41 Vietnamese food items
Provides higher accuracy in classifying individual food items


# 📊 Supported Food Items
The system can recognize 41 common Vietnamese food items in canteens:

Rice dishes: rice, banh mi (Vietnamese sandwich)
Vegetables: boiled cabbage, stir-fried cabbage, tomatoes, carrots, okra, tofu, green beans, cucumber, chili, leafy greens, water spinach, coriander
Meat dishes: stir-fried beef, fried chicken, braised chicken, pork ribs, stir-fried pork ribs, fried meat, boiled meat, fried eggs, boiled eggs
Fish dishes: fried fish, braised fish, shrimp
Soups and liquid dishes: gourd soup, pumpkin soup, vegetable soup, sour soup, seaweed soup, fish sauce, soy sauce
Others: banana, pickled vegetables, watermelon, bitter melon, braised meat with black pepper, braised egg, guava, dragon fruit

# 📝 API Documentation
POST /api/analyze
Analyzes a food tray image.
Request:

image: Image file (multipart/form-data)
OR imageData: Base64 string of the image

Response:
json{
  "success": true,
  "detected_items": [
    {
      "id": 0,
      "yolo_class": "class_name",
      "final_class": "food_name",
      "image": "base64_encoded_image"
    }
  ],
  "bill_details": [
    {
      "id": 0,
      "item": "food_name",
      "price": 25000,
      "calories": 350,
      "image": "base64_encoded_image"
    }
  ],
  "total_cost": 75000,
  "total_calories": 1050,
  "items_count": 3
}
GET /api/food-info
Gets information about all available food items.
Response:
json{
  "success": true,
  "food_info": [
    {
      "name": "com",
      "calories": 200,
      "price": 15000,
      "category": "Carbohydrates"
    }
  ]
}
POST /api/update-food-item
Updates a food item classification and recalculates the bill.
Request:
json{
  "itemIndex": 0,
  "newFoodItem": "com",
  "billData": {
    "bill_details": [...]
  }
}
# 🛠️ Technologies Used

Backend: Python, Flask
Computer Vision: OpenCV, YOLOv8 (Ultralytics)
Machine Learning: TensorFlow, PyTorch
Frontend: HTML, CSS, JavaScript
Data Processing: NumPy, pandas, SciPy

# 📄 License
This project is distributed under the MIT License. See LICENSE for more information.
📞 Contact
Khai Nghi Tran Nguyen - GitHub Profile
Project Link: https://github.com/KhaiNghiTranNguyen/AI-Challenge-3ITech
