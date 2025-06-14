import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pickle
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Set page configuration
st.set_page_config(page_title="Smart Wardrobe", page_icon=":shirt:", layout="wide")

# Define CSS styles
css = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://cdn.openart.ai/published/ueo80BRd2eKE0y1HoO6O/J0e7nWqi_g-wW_raw.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    opacity: 0.8;
}
[data-testid="stSidebar"] {
    background-color: #e0e0e0;  /* Light gray background */
    border-radius: 10px;
    opacity: 0.8;
}
[data-testid="stSidebarNav"] button {
    background-color: #36597d;  /* Darker blue for buttons */
    color: white;
    border: none;
    border-radius: 5px;
    margin: 5px 0;
    padding: 10px 20px;
    cursor: pointer;
}
[data-testid="stSidebarNav"] button:hover {
    background-color: #29496b;  /* Even darker blue on hover */
}
h1 {
    color: #001F3F;  /* Darker blue for better contrast */
    font-family: 'Georgia', serif;
    text-align: center;
}
input[type=text], .stTextInput > div > input {
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #ccc;
    width: 100%;
    box-sizing: border-box;
}
button {
    background-color: #36597d;  /* Same blue as buttons */
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}
button:hover {
    background-color: #29496b;  /* Even darker blue on hover */
}
.file-uploader {
    padding: 20px;
    border: 2px dashed #ccc;
    border-radius: 5px;
    text-align: center;
}
.rating-star {
    color: #ffc107;  /* Keep gold color for rating */
    font-size: 30px;
}
.stMarkdown a {
    color: #FF5733;  /* Change hyperlink color */
    font-weight: bold;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Create multipage structure
pages = ["Home", "Recommendations", "Feedback"]

if 'page' not in st.session_state:
    st.session_state.page = pages[0]

# Function to toggle sidebar visibility
def toggle_sidebar():
    st.session_state.sidebar_visible = not st.session_state.sidebar_visible

# Initialize sidebar visibility state
if 'sidebar_visible' not in st.session_state:
    st.session_state.sidebar_visible = True

# Navigation function
def navigate_to(page):
    st.session_state.page = page

# Navigation buttons
st.sidebar.title("Navigation")
if st.sidebar.button("Toggle Sidebar"):
    toggle_sidebar()

if st.session_state.sidebar_visible:
    for page in pages:
        if st.sidebar.button(page):
            navigate_to(page)

# Function to classify category
def classify_category(input_item, df):
    df = pd.read_csv('categories.csv')
    default_var = 'casual'
    matching_items = df.loc[df['items'].str.lower() == input_item.lower(), 'category']
    if not matching_items.empty:
        category_name = matching_items.iloc[0]
        return category_name
    else:
        return default_var

# Function to save uploaded file
def save_uploaded_file(uploaded_file, user, option):
    try:
        # Create directories if they don't exist
        user_path = os.path.join(root, user)
        option_path = os.path.join(user_path, option)
        os.makedirs(option_path, exist_ok=True)
        
        save_path = os.path.join(option_path, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return print("{uploaded_file.name} saved successfully.")
    except PermissionError as e:
        st.error(f"Permission error: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to create the model
@st.cache_resource  # Cache the model to prevent reloading
def create_model():
    try:
        model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model.trainable = False
        model = tf.keras.Sequential([
            model,
            GlobalMaxPooling2D()
        ])
        return model
    except Exception as e:
        st.error(f"Error creating model: {str(e)}")
        return None

# Function to extract features
def extract_features(img_path, model):
    if model is None:
        st.error("Model not initialized properly")
        return None
        
    if not os.path.isfile(img_path):
        st.error(f"File not found: {img_path}")
        return None
        
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img, verbose=0).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Error extracting features from {os.path.basename(img_path)}: {str(e)}")
        return None

# Function to recommend items
def recommend(features, feature_list, k):
    if features is None or feature_list is None or len(feature_list) == 0:
        st.error("No valid features available for recommendation")
        return None
        
    try:
        feature_list = np.array(feature_list)
        if feature_list.ndim == 1:
            feature_list = feature_list.reshape(1, -1)
        
        if k > len(feature_list):
            k = len(feature_list)
            
        neighbors = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices
    except Exception as e:
        st.error(f"Error in recommendation: {str(e)}")
        return None

root = "/Users/ShahYash/Documents/smart_wardrobe1"

# Function to ensure image orientation is portrait
def ensure_portrait(image_path):
    try:
        img = Image.open(image_path)
        # Check EXIF data for orientation
        try:
            exif = img._getexif()
            if exif is not None:
                orientation = exif.get(274)  # 274 is the orientation tag
                if orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 6:
                    img = img.rotate(270, expand=True)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            # No EXIF data or no orientation info
            pass
        
        width, height = img.size
        if width > height:
            img = img.rotate(-90, expand=True)
        return img
    except Exception as e:
        st.error(f"Error processing image orientation: {e}")
        return Image.open(image_path)  # Return original image if processing fails

# Main logic for each page
if st.session_state.page == "Home":
    st.title("Smart Wardrobe")

    # User input with validation
    user = st.text_input("Enter your name:", st.session_state.get('user', ""))
    option = st.text_input("Enter your Occasion:", st.session_state.get('option', ""))
    dataset_path = os.path.join(root, "categories.csv")
    dataset = pd.read_csv(dataset_path)

    button = st.button("Submit")

    if button:
        # Validate inputs
        if not user.strip():
            st.error("Please enter your name")
        elif not option.strip():
            st.error("Please enter an occasion")
        else:
            user_path = os.path.join(root, user)
            option_path = os.path.join(user_path, option)
            os.makedirs(option_path, exist_ok=True)

            input_item = option
            category = classify_category(input_item, dataset_path)
            st.session_state.user = user
            st.session_state.option = option
            st.session_state.category = category
            navigate_to("Recommendations")

elif st.session_state.page == "Recommendations":
    st.title("Recommendations")

    if 'option' not in st.session_state or 'category' not in st.session_state:
        st.write("It seems you have not selected any option or category. Please go back and enter the details.")
    else:
        st.write(f"Recommendation for {st.session_state.option} will be: {st.session_state.category}")

    st.title("Multiple File Uploader")

    uploaded_files = st.file_uploader("Upload multiple files", accept_multiple_files=True)
    if uploaded_files:
        # Create model once and reuse
        model = create_model()
        if model is None:
            st.error("Failed to initialize the model. Please try again.")
        else:
            filenames = []
            feature_list = []
            
            # Process uploaded files
            for uploaded_file in uploaded_files:
                save_result = save_uploaded_file(uploaded_file, st.session_state.user, st.session_state.option)
                if save_result:
                    st.success(save_result)
                    
            # Extract features from saved files
            user_option_path = os.path.join(root, st.session_state.user, st.session_state.option)
            if os.path.isdir(user_option_path):
                st.info(f"Processing images")
                for file in os.listdir(user_option_path):
                    file_path = os.path.join(user_option_path, file)
                    if os.path.isfile(file_path):
                        features = extract_features(file_path, model)
                        if features is not None:
                            filenames.append(file_path)
                            feature_list.append(features)
                            print(f"Successfully extracted features from {file}")

            if not feature_list:
                st.error('No valid images found or features could not be extracted.')
            else:
                feature_list = np.array(feature_list)
                st.success(f"Successfully processed {len(feature_list)} images")

                # Save features and filenames
                user_directory = os.path.join(root, st.session_state.user)
                os.makedirs(user_directory, exist_ok=True)

                # Determine number of recommendations
                photos_length = len(feature_list)
                k = min(3, max(1, photos_length - 1))  # At least 1, at most 3, but not more than available photos - 1

                if k > 0:
                    recommendation_indices = recommend(feature_list[0], feature_list, k)
                    
                    if recommendation_indices is not None and recommendation_indices.shape[1] > 0:
                        st.success(f"Here are your recommendations!")
                        columns = st.columns(k)
                        
                        # Show recommendations with controlled image size
                        for i in range(k):
                            with columns[i]:
                                if i < recommendation_indices.shape[1]:
                                    img = ensure_portrait(filenames[recommendation_indices[0][i]])
                                    # Resize image to a reasonable width while maintaining aspect ratio
                                    img = img.resize((300, int(300 * img.size[1] / img.size[0])))
                                    st.image(img,
                                           caption=f"Recommendation {i+1}")
                    else:
                        st.error("Could not generate recommendations")
                else:
                    st.error("Not enough images to generate recommendations")

    if st.button("Proceed to Feedback"):
        navigate_to("Feedback")

# Feedback page
elif st.session_state.page == "Feedback":
    st.title('Did you find us helpful')

    def star_rating(rating):
        df = pd.read_csv("/Users/ShahYash/Documents/smart_wardrobe1/referral_links.csv")
        if 'category' not in st.session_state:
            st.error("No category found. Please go back and complete the previous steps.")
            return

        if rating == 0 or rating >= 4:
            matching_rows = df.loc[df['category'].str.lower() == st.session_state.category.lower()]

            if not matching_rows.empty:
                random_row = matching_rows.sample(n=1).iloc[0]
                amazon = random_row['amazon']
                ajio = random_row['ajio']
                myntra = random_row['myntra']

                st.markdown(f"[Amazon]({amazon})")
                st.markdown(f"[Ajio]({ajio})")
                st.markdown(f"[Myntra]({myntra})")
            else:
                st.error("No matching category found.")
        elif rating in [1, 2, 3]:
            user_categoryselection = st.text_input("Please specify a category:")
            matching_rows = df.loc[df['items'].str.lower() == user_categoryselection.lower()]

            if not matching_rows.empty:
                random_row = matching_rows.sample(n=1).iloc[0]
                amazon = random_row['amazon']
                ajio = random_row['ajio']
                myntra = random_row['myntra']

                st.markdown(f"[Amazon]({amazon})")
                st.markdown(f"[Ajio]({ajio})")
                st.markdown(f"[Myntra]({myntra})")
            else:
                st.error("No matching category found.")
        else:
            st.write("Rate us higher to get referral links!")
        return '⭐' * int(rating)

    rating = st.slider("Rate us:", 0, 5, value=0)
    st.write(f"You rated us: {star_rating(rating)}")