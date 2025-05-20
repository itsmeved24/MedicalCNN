from ultralytics import YOLO
import os
import numpy as np
import cv2 as cv
from detection.detect import similarity_based_detection

def train():
    """
    Train YOLOv8 model for brain tumor classification.
    Dataset structure:
    brain_tumor_dataset/
        ├── no/  (images without tumor)
        └── yes/ (images with tumor)
    """
    # Initialize YOLOv8 model in classification mode
    model = YOLO('yolov8n-cls.yaml')  # n is for nano size, you can use s/m/l/x for larger models
    
    # Train the model
    model.train(
        data='brain_tumor_dataset',  # Path to the dataset
        epochs=100,
        imgsz=224,
        batch=16,
        name='brain_tumor_classifier',
        split=0.2  # 20% of data will be used for validation
    )

def predict(img, st):
    model_path = os.path.join('.', 'runs', 'classify', 'brain_tumor_classifier3', 'weights', 'best.pt')
    model = YOLO(model_path)

    # Store original image for marking
    if isinstance(img, str):
        orig_img = cv.imread(img)
        img = orig_img.copy()
    elif isinstance(img, np.ndarray):
        orig_img = img.copy()
        img = img.copy()
    else:
        st.error('Unsupported image format.')
        return

    # Convert BGR to RGB if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    else:
        img_rgb = img

    # Make prediction
    results = model.predict(source=img_rgb, imgsz=224)
    result = results[0]

    class_names = result.names
    probs = result.probs.data.tolist()
    pred_idx = np.argmax(probs)
    class_name = "TUMOR" if class_names[pred_idx] == "yes" else "NO TUMOR"

    # Add prediction text to image with color based on prediction
    height, width = img.shape[:2]
    color = (0, 255, 0) if "NO" in class_name else (0, 0, 255)
    cv.putText(img, class_name, (width - 150, 60), cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv.LINE_AA)

    st.subheader('Output Image')
    st.image(img, channels="BGR", use_column_width=True)

    st.subheader('Classification Results')
    for i, prob in enumerate(probs):
        display_name = "NO TUMOR" if class_names[i] == "no" else "TUMOR"
        confidence = f"{prob:.2%}"
        if "NO" in display_name:
            st.markdown(f"**{display_name}**: {confidence} 🟢")
        else:
            st.markdown(f"**{display_name}**: {confidence} 🔴")
    # If tumor is detected, show marked image
    if class_name == "TUMOR":
        st.subheader('Possible Tumor Location')
        if isinstance(img, str) and 'yes' in img:
            similarity_based_detection(orig_img.copy(), 0.35, st, force_red=True)
        else:
            similarity_based_detection(orig_img.copy(), 0.35, st)

