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

# Class labels
CLASS_LABELS = [
    'The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa',
    'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 
    'angel', 'animal migration', 'ant'
]

@st.cache_resource
def load_model():
    """Load the trained Keras model from file."""
    try:
        model = tf.keras.models.load_model('sketch_classifier.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image_data):
    """Preprocess the drawn image for model prediction."""
    try:
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(image) / 255.0
        
        # Reshape to (1, 28, 28, 1) for model input
        image_array = image_array.reshape(1, 28, 28, 1)
        
        return image_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def get_top_predictions(predictions, top_k=3):
    """Get top k predictions with their confidence scores."""
    # Get indices of top k predictions
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    
    # Get corresponding scores and labels
    top_predictions = []
    for idx in top_indices:
        confidence = float(predictions[0][idx] * 100)
        label = CLASS_LABELS[idx]
        top_predictions.append((label, confidence))
    
    return top_predictions

def main():
    # Header
    st.title("✏️ Sketch Recognition App")
    st.markdown("---")
    
    # Description
    st.markdown("""
    Draw a sketch on the canvas below and the AI will predict what it represents!
    
    **Available classes:** The Eiffel Tower, The Great Wall of China, The Mona Lisa, 
    aircraft carrier, airplane, alarm clock, ambulance, angel, animal migration, ant
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
        )
        
        # Add clear button
        if st.button("🗑️ Clear Canvas"):
            st.rerun()
    
    with col2:
        st.subheader("🔍 Predictions")
        
        # Check if there's a drawing
        if canvas_result.image_data is not None:
            # Convert canvas data to image
            image_data = canvas_result.image_data
            
            # Preprocess the image
            processed_image = preprocess_image(image_data)
            
            if processed_image is not None:
                # Make prediction
                predictions = model.predict(processed_image, verbose=0)
                
                # Get top predictions
                top_predictions = get_top_predictions(predictions, top_k=3)
                
                # Display top prediction
                if top_predictions:
                    top_label, top_confidence = top_predictions[0]
                    
                    st.markdown("### 🏆 Top Prediction")
                    st.markdown(f"**{top_label}**")
                    st.progress(top_confidence / 100)
                    st.markdown(f"**Confidence: {top_confidence:.1f}%**")
                    
                    st.markdown("---")
                    
                    # Display top 3 predictions
                    st.markdown("### 📊 Top 3 Predictions")
                    for i, (label, confidence) in enumerate(top_predictions, 1):
                        st.markdown(f"**{i}.** {label}")
                        st.progress(confidence / 100)
                        st.markdown(f"*{confidence:.1f}%*")
                        st.markdown("")
            else:
                st.warning("Unable to process the drawing. Please try again.")
        else:
            st.info("👆 Draw something on the canvas to see predictions!")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and TensorFlow*")

if __name__ == "__main__":
    main() 