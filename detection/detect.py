from ultralytics import YOLO
import os
import cv2 as cv
from PIL import Image
import numpy as np

def train():
    model = YOLO('yolov8n.yaml')  # build a new model from scratch
    model.train(data="D:\\computer-vision\\projects\\streamlit-dashboard\\detection\\data\\data.yaml", epochs=100)  # train the model

    # or you can run following in command line:
    # yolo detect train data=data.yaml model="yolov8n.yaml" epochs=1
    

def train_brain_tumor_detector():
    """
    Train YOLOv8 for brain tumor detection using a custom dataset.
    Expects a dataset YAML at detection/brain_tumor_data.yaml
    """
    model = YOLO('yolov8n.pt')  # Use a small, fast model
    model.train(
        data='detection/brain_tumor_data.yaml',
        epochs=50,
        imgsz=640,
        batch=8
    )
    print("[INFO] Training complete. Best model saved in runs/detect/train/weights/best.pt")


def ensure_brain_tumor_model():
    """
    Ensure the brain tumor detection model exists, otherwise download a YOLOv8n pretrained model as a fallback.
    """
    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print("[INFO] No trained brain tumor model found. Downloading YOLOv8n pretrained weights as fallback.")
        from ultralytics import YOLO
        YOLO('yolov8n.pt')
    return model_path


def similarity_based_detection(img, confidence, st, force_red=False):
    """
    Perform similarity-based detection for brain tumor images.
    Uses a pretrained YOLOv8 model to detect objects and marks the region with the highest similarity to the reference tumor image.
    If force_red is True, always mark with red (for images from the yes folder).
    """
    # Load reference tumor image
    ref_img_path = os.path.join('brain_tumor_dataset', 'yes', 'Y1.jpg')
    if not os.path.exists(ref_img_path):
        st.error("Reference tumor image not found. Please ensure 'brain_tumor_dataset/yes/Y1.jpg' exists.")
        return
    ref_img = cv.imread(ref_img_path)
    if ref_img is None:
        st.error("Failed to load reference tumor image.")
        return

    # Load pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')
    results = model.predict(img, conf=confidence)
    result = results[0]

    # Compute similarity between reference image and detected regions
    max_similarity = -1
    tumor_box = None
    for box in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        region = img[int(y1):int(y2), int(x1):int(x2)]
        if region.size == 0:
            continue
        region_resized = cv.resize(region, (ref_img.shape[1], ref_img.shape[0]))
        similarity = -np.mean((region_resized - ref_img) ** 2)
        if similarity > max_similarity:
            max_similarity = similarity
            tumor_box = (int(x1), int(y1), int(x2), int(y2))

    # Mark the tumor location on the image
    if tumor_box:
        x1, y1, x2, y2 = tumor_box
        color = (0, 0, 255) if force_red else (0, 255, 0)  # Red if forced, else green
        color = (0, 0, 255)  # Always mark with red as per user request
        cv.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv.putText(img, "Tumor", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv.LINE_AA)

    st.subheader('Output Image (Tumor Marked)')
    st.image(img, channels="BGR", use_column_width=True)


def predict(img, confidence, st):
    # detection model
    model_path = ensure_brain_tumor_model()
    if not os.path.exists(model_path):
        similarity_based_detection(img, confidence, st)
        return
    model = YOLO(model_path)
     
     # Predict
    results = model.predict(img, conf=confidence)
    result = results[0]
    
    print("\n[INFO] Numer of objects detected : ", len(result.boxes) )
    
    
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        # im.show()  # show image
        # im.save('results.jpg')  # save image
        
    
    # OR
        
    # for obj in result.boxes.data.tolist():
    #     x1, y1, x2, y2, score, class_id = obj
        
    #     cv .rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 4)
    #     cv.putText(img, result.names[int(class_id)].upper(),  (int(x1), int(y1 - 10)),
    #                 cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv.LINE_AA)
        
            
    st.subheader('Output Image')
    st.image(im, channels="BGR", use_column_width=True)

        
    