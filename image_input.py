# import streamlit as st
# from PIL import Image

# def main():
#     st.title("Image Input Example")
    
#     # File uploader for image input
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         # Display the selected image
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
        
#         # You can now perform further processing on the image if needed
        
#         # Example: Display image dimensions
#         image_width, image_height = image.size
#         st.write(f"Image Dimensions: {image_width}x{image_height}")
        
# if __name__ == "__main__":
#     main()


import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import preprocessing

# Load your pre-trained MNIST model
model = load_model('mnist_model.h5')

def preprocess_image(img):
    img = img.convert('L')
    img = img.resize((28, 28))
    
    # Normalize the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    return img_array

def main():
    st.title("Hand-Written Digit Recognition")
    
    # File uploader for image input
    uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png","webp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Display the preprocessed image
        st.image(processed_image[0, :, :, 0], caption="Preprocessed Image", use_column_width=True)
        
        # Make prediction using the model
        prediction = model.predict(processed_image)
        predicted_label = np.argmax(prediction)
        st.markdown("Model's Prediction starts from here....")
        st.write(f"It's a : {predicted_label}")
        
if __name__ == "__main__":
    main()
