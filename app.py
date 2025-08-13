import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit_drawable_canvas as st_canvas
import io

# Page configuration
st.set_page_config(
    page_title="Sketch Recognition App",
    page_icon="✏️",
    layout="wide"
)

# Class labels will be generated dynamically based on model output size

@st.cache_resource
def load_model():
    """Load the trained Keras model from file."""
    try:
        model = tf.keras.models.load_model('sketch_classifier.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_class_labels(num_classes):
    """Generate class labels based on the number of classes in the model."""
    # Actual class labels from the training data
    actual_labels = [
        'The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa',
        'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 
        'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 
        'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 
        'baseball bat', 'baseball', 'basket', 'basketball', 'bat', 
        'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 
        'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 
        'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 
        'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 
        'broom', 'bucket', 'bulldozer'
    ]
    
    # If we need more classes, generate generic names
    if num_classes <= len(actual_labels):
        return actual_labels[:num_classes]
    else:
        # Extend with generic class names
        extended_labels = actual_labels.copy()
        for i in range(len(actual_labels), num_classes):
            extended_labels.append(f"Class_{i}")
        return extended_labels

def preprocess_image(image_data):
    """Preprocess the drawn image for model prediction."""
    try:
        # Handle different types of image data
        if isinstance(image_data, np.ndarray):
            # If it's already a numpy array
            if len(image_data.shape) == 3:
                # Convert RGB to grayscale
                image = Image.fromarray(image_data).convert('L')
            else:
                # Already grayscale
                image = Image.fromarray(image_data)
        elif isinstance(image_data, str):
            # If it's a base64 string, decode it
            import base64
            if ',' in image_data:
                image_data = base64.b64decode(image_data.split(',')[1])
            else:
                image_data = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_data))
        else:
            # Try to convert to PIL Image
            image = Image.fromarray(image_data)
        
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Invert the image (make black lines on white background)
        image_array = 1.0 - image_array
        
        # Ensure values are in [0, 1] range
        image_array = np.clip(image_array, 0.0, 1.0)
        
        # Reshape to (1, 28, 28, 1) for model input
        image_array = image_array.reshape(1, 28, 28, 1)
        
        return image_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def get_top_predictions(predictions, class_labels, top_k=3):
    """Get top k predictions with their confidence scores."""
    # Get indices of top k predictions
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    
    # Get corresponding scores and labels
    top_predictions = []
    for idx in top_indices:
        confidence = float(predictions[0][idx] * 100)
        # Safety check to prevent index errors
        if idx < len(class_labels):
            label = class_labels[idx]
        else:
            label = f"Class_{idx}"
        top_predictions.append((label, confidence))
    
    return top_predictions

def main():
    # Initialize session state
    if 'has_drawn' not in st.session_state:
        st.session_state.has_drawn = False
    
    # Header
    st.title("✏️ Sketch Recognition App")
    st.markdown("---")
    
    # Description
    st.markdown("""
    Draw a sketch on the canvas below and the AI will predict what it represents!
    
    **Available classes:** The model can recognize 50 different objects including:
    The Eiffel Tower, The Great Wall of China, The Mona Lisa, aircraft carrier, airplane, 
    alarm clock, ambulance, angel, animal migration, ant, anvil, apple, arm, asparagus, 
    axe, backpack, banana, bandage, barn, baseball bat, baseball, basket, basketball, 
    bat, bathtub, beach, bear, beard, bed, bee, belt, bench, bicycle, binoculars, bird, 
    birthday cake, blackberry, blueberry, book, boomerang, bottlecap, bowtie, bracelet, 
    brain, bread, bridge, broccoli, broom, bucket, bulldozer
    """) 
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check if 'sketch_classifier.h5' exists.")
        return
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎨 Drawing Canvas")
        
        # Create canvas
        canvas_result = st_canvas.st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Orange fill with transparency
            stroke_width=8,
            stroke_color="black",
            background_color="white",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
            update_streamlit=True,
        )
        
        # Check if user has drawn something
        if canvas_result.json_data is not None:
            if len(canvas_result.json_data["objects"]) > 0:
                st.session_state.has_drawn = True
        
        # Add clear button
        if st.button("🗑️ Clear Canvas"):
            st.session_state.has_drawn = False
            st.rerun()
    
    with col2:
        st.subheader("🔍 Predictions")
        
        # Initialize processed_image variable
        processed_image = None
        
        # Only show predictions if user has drawn something
        if st.session_state.has_drawn and canvas_result.image_data is not None:
            # Get the canvas data
            image_data = canvas_result.image_data
            

            
            # Check if the image has any content (not just white background)
            if isinstance(image_data, np.ndarray):
                # Convert to grayscale if it's RGB
                if len(image_data.shape) == 3:
                    gray_image = np.mean(image_data, axis=2)
                else:
                    gray_image = image_data
                
                # Check if there's any drawing (non-white pixels)
                if np.max(gray_image) > 250:  # If mostly white, no drawing
                    st.info("👆 **Draw something on the canvas to see predictions!**")
                    st.markdown("""
                    **Try drawing:**
                    - A simple shape like ⭐ (star) or ❤️ (heart)
                    - An animal like 🐱 (cat) or 🐶 (dog)
                    - An object like 🏠 (house) or 🌲 (tree)
                    """)
                else:
                    # Preprocess the image
                    processed_image = preprocess_image(image_data)
            else:
                st.info("👆 **Draw something on the canvas to see predictions!**")
                st.markdown("""
                **Try drawing:**
                - A simple shape like ⭐ (star) or ❤️ (heart)
                - An animal like 🐱 (cat) or 🐶 (dog)
                - An object like 🏠 (house) or 🌲 (tree)
                """)
                processed_image = None
        else:
            st.info("👆 **Draw something on the canvas to see predictions!**")
            st.markdown("""
            **Try drawing:**
            - A simple shape like ⭐ (star) or ❤️ (heart)
            - An animal like 🐱 (cat) or 🐶 (dog)
            - An object like 🏠 (house) or 🌲 (tree)
            """)
            processed_image = None
        
        # Process predictions if we have a valid image
        if processed_image is not None:
            # Make prediction
            with st.spinner("🤖 Analyzing your drawing..."):
                predictions = model.predict(processed_image, verbose=0)
            
            # Get class labels based on model output size
            num_classes = predictions.shape[1]
            class_labels = get_class_labels(num_classes)
            
            # Get top predictions
            top_predictions = get_top_predictions(predictions, class_labels, top_k=3)
            
            # Display top prediction
            if top_predictions:
                top_label, top_confidence = top_predictions[0]
                
                # Display top prediction with better styling
                st.markdown("### 🎯 **Your Drawing Looks Like:**")
                
                # Create a colored box for the top prediction
                if top_confidence > 70:
                    st.success(f"**{top_label}**")
                elif top_confidence > 40:
                    st.warning(f"**{top_label}**")
                else:
                    st.info(f"**{top_label}**")
                
                # Show confidence with progress bar
                st.progress(top_confidence / 100)
                st.markdown(f"**Confidence: {top_confidence:.1f}%**")
                
                st.markdown("---")
                
                # Display top 3 predictions
                st.markdown("### 📊 **All Predictions:**")
                for i, (label, confidence) in enumerate(top_predictions, 1):
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col1:
                        st.markdown(f"**{i}.**")
                    with col2:
                        st.markdown(f"**{label}**")
                    with col3:
                        st.markdown(f"**{confidence:.1f}%**")
                    st.progress(confidence / 100)
                    st.markdown("")
        elif processed_image is None and st.session_state.has_drawn:
            st.warning("Unable to process the drawing. Please try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and TensorFlow*")

if __name__ == "__main__":
    main() 