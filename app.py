# Streamlit setup
import streamlit as st
from streamlit_cropper import st_cropper
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans

# Function to preprocess image (contrast enhancement)
def preprocess_image(image):
    # Convert to grayscale for further processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Histogram equalization for better contrast
    equalized = cv2.equalizeHist(gray_image)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    return blurred

# Function for advanced color distribution using K-means clustering
def advanced_color_analysis(image, n_clusters=3):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)
    labels = kmeans.labels_

    # Calculate the percentage of each cluster
    cluster_counts = np.bincount(labels)
    cluster_percentages = (cluster_counts / len(labels)) * 100

    # Sort clusters by their average intensity
    cluster_centers = kmeans.cluster_centers_
    sorted_indices = np.argsort(cluster_centers.sum(axis=1))

    # Identify black, red, and grey based on intensity
    black_cluster = sorted_indices[0]   # Coke (black)
    red_cluster = sorted_indices[1]     # Cryolite (red)
    grey_cluster = sorted_indices[2]    # Cryolite (grey)

    black_percentage = cluster_percentages[black_cluster]
    red_percentage = cluster_percentages[red_cluster]
    grey_percentage = cluster_percentages[grey_cluster]

    # Cryolite is a combination of red and grey
    cryolite_percentage = red_percentage + grey_percentage

    return black_percentage, cryolite_percentage

# Function to assign grades based on color distribution
def assign_grade(black_percentage, cryolite_percentage):
    if black_percentage > cryolite_percentage:
        return 'Grade C: More coke (black)'
    elif cryolite_percentage > black_percentage:
        return 'Grade A: More cryolite (red/grey)'
    else:
        return 'Grade B: Balanced'

# Main function
def main():
    st.title("Coke-Cryolite Image Analysis with Cropping")

    # Input fields for Pot Number and Anode Change Group
    pot_number = st.text_input("Enter Pot Number")
    anode_group = st.text_input("Enter Anode Change Group")

    # Allow user to upload an image
    uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        # Enhance image contrast for better analysis
        enhancer = ImageEnhance.Contrast(Image.fromarray(image))
        image = np.array(enhancer.enhance(1.5))

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Crop the image using streamlit-cropper
        st.subheader("Crop the area of interest")
        cropped_image = st_cropper(Image.fromarray(image), realtime_update=True, box_color='#FF6347', aspect_ratio=None)
        cropped_image = np.array(cropped_image)

        # Convert the cropped image to OpenCV format
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)

        # Preprocess the cropped image
        preprocessed_image = preprocess_image(cropped_image)

        # Analyze the color distribution using K-means clustering
        black_percentage, cryolite_percentage = advanced_color_analysis(preprocessed_image)

        # Assign a grade based on the distribution
        grade = assign_grade(black_percentage, cryolite_percentage)

        # Display the results
        st.write(f'**Pot Number:** {pot_number}')
        st.write(f'**Anode Change Group:** {anode_group}')
        st.write(f'**Black (Coke) Percentage:** {black_percentage:.2f}%')
        st.write(f'**Cryolite (Red + Grey) Percentage:** {cryolite_percentage:.2f}%')
        st.write(f'**Assigned Grade:** {grade}')

        # Display the cropped and preprocessed image
        st.image(cropped_image, caption="Cropped Image for Analysis", use_column_width=True)

# Run the app
if __name__ == "__main__":
    main()

