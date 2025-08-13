# ✏️ Sketch Recognition App

A Streamlit web application that uses a trained Keras model to recognize hand-drawn sketches in real-time.

## Features

- **Interactive Drawing Canvas**: Draw sketches with a 280x280 white canvas and black pen (width 8)
- **Real-time Prediction**: Get instant predictions as you draw
- **Top 3 Predictions**: See the model's confidence scores for the top 3 most likely classes
- **Clean UI**: Modern, responsive interface with clear visual feedback

## Available Classes

The model can recognize 50 Classes

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure the model file is present:**
   - Make sure `sketch_classifier.h5` is in the same directory as `app.py`

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   - The app will automatically open in your default browser
   - Usually at `http://localhost:8501`

3. **Start drawing:**
   - Use the drawing canvas on the left
   - Draw any of the available classes
   - See real-time predictions on the right

4. **Clear canvas:**
   - Click the "🗑️ Clear Canvas" button to start over

## Technical Details

- **Model**: Keras model loaded from `sketch_classifier.h5`
- **Preprocessing**: 
  - Converts drawing to grayscale
  - Resizes to 28x28 pixels
  - Normalizes pixel values to [0,1]
  - Reshapes to (1, 28, 28, 1) for model input
- **Caching**: Model loading is cached for better performance
- **Error Handling**: Graceful error handling for model loading and image processing

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- Streamlit 1.28+
- Other dependencies listed in `requirements.txt`

## Troubleshooting

- **Model loading error**: Ensure `sketch_classifier.h5` is in the correct directory
- **Canvas not working**: Check if `streamlit-drawable-canvas` is properly installed

- **Memory issues**: The model is cached, so it only loads once per session 
