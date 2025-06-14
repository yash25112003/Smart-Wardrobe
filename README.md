# Smart Wardrobe - AI-Powered Fashion Assistant

An intelligent fashion recommendation system that suggests outfit combinations based on your wardrobe and occasion, leveraging advanced computer vision and machine learning techniques.

## 🌟 Features

* **Personalized Outfit Recommendations**: Generates tailored outfit suggestions based on the user's uploaded wardrobe and specified occasion.

* **Visual Similarity Matching**: Utilizes ResNet50 for robust feature extraction to identify visually similar clothing items within the user's collection.

* **Intuitive Multi-Page Interface**: Provides a seamless user experience with dedicated sections for clothing upload, recommendation viewing, and feedback submission.

* **Smart Clothing Categorization**: Automatically classifies uploaded clothing items by occasion and type, streamlining wardrobe organization.

* **Personalized Shopping Integration**: Offers curated shopping links for new items based on user preferences and identified wardrobe gaps.

* **Secure User Profiles**: Enables users to save, organize, and manage their wardrobe effectively across different occasions.

## 🛠️ Technical Implementation

### 🔍 Core Technologies

* **Computer Vision**: ResNet50 (pre-trained on ImageNet) for deep feature extraction from clothing images.

* **Machine Learning**: k-Nearest Neighbors (k-NN) algorithm for efficient similarity matching and personalized recommendations.

* **Backend Frameworks**: TensorFlow for deep learning operations, scikit-learn for machine learning algorithms, and NumPy for numerical computations.

* **Frontend Interface**: Streamlit for creating an interactive and responsive web application.

* **Image Processing**: OpenCV and PIL (Pillow) for robust image handling, manipulation, and pre-processing.

### 📂 Project Structure

```bash
smart-wardrobe/
├── main.py               # Main application entry point and logic for Streamlit app
├── users/                # Directory for storing user-specific uploads and profiles
├── categories.csv        # CSV file defining clothing category mappings
├── referral_links.csv    # CSV file containing shopping recommendation links
└── requirements.txt      # Lists all Python dependencies for the project
```

## 🚀 Getting Started

Follow these steps to set up and run the Smart Wardrobe application on your local machine.

### Prerequisites

* Python 3.8+

* `pip` package manager

### Installation

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/yourusername/smart-wardrobe.git](https://github.com/yourusername/smart-wardrobe.git)
   cd smart-wardrobe
   ```

   *(Note: Replace `https://github.com/yourusername/smart-wardrobe.git` with your actual repository URL.)*

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: `venv\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   If you do not have a `requirements.txt` file, you can install the necessary packages manually:
   ```bash
   pip install streamlit tensorflow scikit-learn numpy opencv-python Pillow
   ```

## 🖥️ Usage Guide

To launch the application, ensure your virtual environment is active and run:

```bash
streamlit run main.py
```

This will open the application in your default web browser.

### 1. Home Page
    * Enter your name and the occasion for which you need an outfit.
    * The system will process your input to understand your outfit needs.
### 2. Recommendations Engine
    * Upload images of your clothing items.
    * Receive AI-generated outfit recommendations tailored to the occasion.
    * Visualize visually similar items from your existing wardrobe to help combine outfits.
### 3. Feedback Page
    * Rate your experience with the recommendations.
    * Access personalized shopping suggestions based on your preferences and previous interactions.
    * Provide valuable feedback to continuously improve the system's accuracy and relevance.

## 📊 Technical Details

### 1. Input Handling

* Implements secure file upload mechanisms with robust validation to ensure data integrity.

* Includes automatic EXIF orientation correction for uploaded images, ensuring correct display and processing regardless of original camera orientation.

* Organizes uploaded wardrobe items into user-specific directories for efficient management and personalized experiences.

### 2. Feature Extraction

* Utilizes the `ResNet50` convolutional base, pre-trained on the ImageNet dataset, to extract rich, high-level features from clothing images.

* Applies a `GlobalMaxPooling2D` layer to reduce the output of `ResNet50` to compact 2048-dimensional feature vectors.

* Employs L2 normalization on the feature vectors to ensure accurate and reliable similarity comparisons, crucial for the recommendation engine.

### 3. Recommendation Engine

* Employs the `k-Nearest Neighbors` (k-NN) algorithm, with a default `k` value of 3, to find the most similar clothing items.

* Uses the `Euclidean distance` metric to quantify the similarity between feature vectors, identifying close matches for outfit combinations.

* Includes dynamic adjustment logic for the `k` parameter, optimizing recommendations even for users with smaller wardrobe collections.
