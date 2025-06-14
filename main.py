import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
import streamlit as st # type: ignore
from PIL import Image  # type: ignore
from tqdm import tqdm # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.layers import GlobalMaxPooling2D # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore
from numpy.linalg import norm # type: ignore
from scipy import spatial  # type: ignore # Add this import for distance calculations

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

# Get the directory where the script is located and set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
users_dir = os.path.join(script_dir, 'users')  # Create a dedicated users directory
os.makedirs(users_dir, exist_ok=True)  # Ensure users directory exists

# Function to classify category
def classify_category(input_item, df):
    try:
        csv_path = os.path.join(script_dir, 'categories.csv')
        df = pd.read_csv(csv_path)
        default_var = 'casual'
        matching_items = df.loc[df['items'].str.lower() == input_item.lower(), 'category']
        if not matching_items.empty:
            category_name = matching_items.iloc[0]
            return category_name
        else:
            return default_var
    except Exception as e:
        st.error(f"Error reading categories file: {str(e)}")
        return default_var

# Function to save uploaded file
def save_uploaded_file(uploaded_file, user, option):
    try:
        # Create directories if they don't exist
        user_path = os.path.join(users_dir, user)  # Changed from root to users_dir
        option_path = os.path.join(user_path, option)
        os.makedirs(option_path, exist_ok=True)
        
        save_path = os.path.join(option_path, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return f"{uploaded_file.name} saved successfully."  # Fixed string formatting
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
        
        # Apply feature importance weighting
        feature_weights = np.linspace(1.0, 0.5, len(result))  # Higher weight to initial features
        weighted_result = result * feature_weights
        
        # L2 normalization
        normalized_result = weighted_result / np.sqrt(np.sum(weighted_result**2))
        return normalized_result
    except Exception as e:
        st.error(f"Error extracting features from {os.path.basename(img_path)}: {str(e)}")
        return None

# Function to calculate confidence scores
def calculate_confidence(features, recommended_features):
    cosine_similarity = 1 - spatial.distance.cosine(features, recommended_features)
    euclidean_distance = spatial.distance.euclidean(features, recommended_features)
    normalized_euclidean = 1 / (1 + euclidean_distance)  # Convert to similarity score
    
    # Combine scores with weights
    confidence = (0.7 * cosine_similarity + 0.3 * normalized_euclidean) * 100
    return min(100, max(0, confidence))  # Ensure score is between 0 and 100

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
        
        # Compute multiple distance metrics
        cosine_neighbors = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine')
        euclidean_neighbors = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
        
        cosine_neighbors.fit(feature_list)
        euclidean_neighbors.fit(feature_list)
        
        # Get recommendations using both metrics
        cosine_distances, cosine_indices = cosine_neighbors.kneighbors([features])
        euclidean_distances, euclidean_indices = euclidean_neighbors.kneighbors([features])
        
        # Combine and weight the results
        combined_indices = []
        seen_indices = set()
        
        # Prioritize items that appear in both metrics
        for c_idx, e_idx in zip(cosine_indices[0], euclidean_indices[0]):
            if c_idx not in seen_indices:
                combined_indices.append(c_idx)
                seen_indices.add(c_idx)
            if e_idx not in seen_indices:
                combined_indices.append(e_idx)
                seen_indices.add(e_idx)
        
        # Return top k recommendations
        return np.array([combined_indices[:k]])
        
    except Exception as e:
        st.error(f"Error in recommendation: {str(e)}")
        return None

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

# Function to reset and go to home
def reset_and_go_home():
    # Clear all session state except sidebar visibility
    sidebar_visible = st.session_state.sidebar_visible
    st.session_state.clear()
    st.session_state.sidebar_visible = sidebar_visible
    st.session_state.page = "Home"

# Main logic for each page
if st.session_state.page == "Home":
    st.title("Smart Wardrobe")

    # User input with validation
    user = st.text_input("Enter your name:", st.session_state.get('user', ""))
    option = st.text_input("Enter your Occasion:", st.session_state.get('option', ""))
    
    button = st.button("Submit")

    if button:
        # Validate inputs
        if not user.strip():
            st.error("Please enter your name")
        elif not option.strip():
            st.error("Please enter an occasion")
        else:
            user_path = os.path.join(users_dir, user)  # Changed from root to users_dir
            option_path = os.path.join(user_path, option)
            os.makedirs(option_path, exist_ok=True)

            input_item = option
            category = classify_category(input_item, None)  # Remove unused df parameter
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
            user_option_path = os.path.join(users_dir, st.session_state.user, st.session_state.option)  # Changed from root to users_dir
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
                user_directory = os.path.join(users_dir, st.session_state.user)
                os.makedirs(user_directory, exist_ok=True)

                # Determine number of recommendations
                photos_length = len(feature_list)
                k = min(3, max(1, photos_length - 1))  # At least 1, at most 3, but not more than available photos - 1

                if k > 0:
                    recommendation_indices = recommend(feature_list[0], feature_list, k)
                    
                    if recommendation_indices is not None and recommendation_indices.shape[1] > 0:
                        st.success(f"Here are your recommendations!")
                        columns = st.columns(k)
                        
                        # Show recommendations with confidence scores
                        for i in range(k):
                            with columns[i]:
                                if i < recommendation_indices.shape[1]:
                                    img = ensure_portrait(filenames[recommendation_indices[0][i]])
                                    # Calculate confidence score
                                    confidence = calculate_confidence(
                                        feature_list[0], 
                                        feature_list[recommendation_indices[0][i]]
                                    )
                                    
                                    # Resize image
                                    img = img.resize((300, int(300 * img.size[1] / img.size[0])))
                                    st.image(img)
                                    st.caption(f"Recommendation {i+1}")
                                    print("Confidence: {confidence:.1f}%\n")
                    else:
                        st.error("Could not generate recommendations")
                else:
                    st.error("Not enough images to generate recommendations")

    if st.button("Proceed to Feedback"):
        navigate_to("Feedback")

# Feedback page
elif st.session_state.page == "Feedback":
    st.title('How was your experience?')
    
    def star_rating(rating):
        if 'category' not in st.session_state:
            st.warning("⚠️ No recommendations were generated. Please complete the previous steps first.")
            return '⭐' * int(rating)
            
        try:
            csv_path = os.path.join(script_dir, 'referral_links.csv')
            df = pd.read_csv(csv_path)
            
            # Show different messages based on rating
            if rating == 0:
                st.info("Please rate your experience to get personalized shopping recommendations!")
            elif rating >= 4:
                st.success("Thank you for your positive feedback! 🎉")
                st.write("Here are some shopping recommendations based on your preferences:")
                
                matching_rows = df.loc[df['category'].str.lower() == st.session_state.category.lower()]
                if not matching_rows.empty:
                    random_row = matching_rows.sample(n=1).iloc[0]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if random_row['amazon']:
                            st.markdown(f"[![Amazon](https://img.icons8.com/color/48/000000/amazon.png)]({random_row['amazon']})")
                            st.markdown(f"[Shop on Amazon]({random_row['amazon']})")
                    with col2:
                        if random_row['ajio']:
                            st.markdown(f"[![Ajio](https://img.icons8.com/color/48/000000/shopping-bag.png)]({random_row['ajio']})")
                            st.markdown(f"[Shop on Ajio]({random_row['ajio']})")
                    with col3:
                        if random_row['myntra']:
                            st.markdown(f"[![Myntra](https://img.icons8.com/color/48/000000/shopping-cart.png)]({random_row['myntra']})")
                            st.markdown(f"[Shop on Myntra]({random_row['myntra']})")
                else:
                    st.info("We'll add more shopping recommendations for this category soon!")
                    
            elif rating in [1, 2, 3]:
                st.write("We're sorry your experience wasn't perfect. Help us improve!")
                user_categoryselection = st.text_input("What category were you looking for?")
                
                if user_categoryselection.strip():
                    matching_rows = df.loc[df['items'].str.lower() == user_categoryselection.lower()]
                    if not matching_rows.empty:
                        st.success("Here are some alternative recommendations:")
                        random_row = matching_rows.sample(n=1).iloc[0]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if random_row['amazon']:
                                st.markdown(f"[![Amazon](https://img.icons8.com/color/48/000000/amazon.png)]({random_row['amazon']})")
                                st.markdown(f"[Shop on Amazon]({random_row['amazon']})")
                        with col2:
                            if random_row['ajio']:
                                st.markdown(f"[![Ajio](https://img.icons8.com/color/48/000000/shopping-bag.png)]({random_row['ajio']})")
                                st.markdown(f"[Shop on Ajio]({random_row['ajio']})")
                        with col3:
                            if random_row['myntra']:
                                st.markdown(f"[![Myntra](https://img.icons8.com/color/48/000000/shopping-cart.png)]({random_row['myntra']})")
                                st.markdown(f"[Shop on Myntra]({random_row['myntra']})")
                    else:
                        st.info("We'll add recommendations for this category in the future!")
                        
            # Add feedback text area and home button for all ratings except 0
            if rating > 0:
                feedback_text = st.text_area("Any additional feedback for us?", 
                                          placeholder="Your feedback helps us improve!")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Submit Feedback"):
                        if not feedback_text.strip():
                            st.error("Please provide some feedback before submitting")
                        else:
                            st.success("Thank you for your feedback! 🙏")
                            st.session_state.feedback_submitted = True
            
                # Only show Go to Home button after rating is given
                with col2:
                    if st.button("🏠 Start Over", type="primary"):
                        reset_and_go_home()
                    
        except Exception as e:
            st.error(f"An error occurred while processing your feedback. Please try again.")
            
        return '⭐' * int(rating)

    rating = st.slider("Rate your experience:", min_value=0, max_value=5, value=0, 
                      help="Drag the slider to rate your experience")
    st.write(f"Your rating: {star_rating(rating)}")