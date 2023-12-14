import os
from PIL import Image
import streamlit as st

def find_dataset(directory_name):
    for root, dirs, files in os.walk("."):
        if directory_name in dirs:
            return os.path.join(root, directory_name)
    return None

def load_image(file_path):
    with open(file_path, 'rb') as file:
        image = Image.open(file)
        return image

dataset_directory = find_dataset("Covid19-dataset")

if dataset_directory is not None:
    st.write("Dataset directory found.")

    def load_images(directory_path):
        images = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                    file_path = os.path.join(root, file)
                    image = load_image(file_path)
                    images.append(image)
        return images

    images = load_images(dataset_directory)

    for i, image in enumerate(images):
        st.image(image, caption=f"Image {i + 1}", use_column_width=True)
else:
    st.write("Dataset directory not found.")
