import streamlit as st
from PIL import Image
import io
from models.detector import YOLOObjectDetection  # import your detector function
from models.ROI_Points import get_roi_points  # import your ROI points function
from segment_anything_2.segmentation_mask import get_segmentation_mask  # import your segmentation function
import numpy as np
import cv2
from PIL import Image as PILImage
import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'segment-anything-2')))
# from segmentation_mask import get_segmentation_mask  # import your segmentation function 


def main():
    st.title("DressifyAI")
    st.write("Welcome to DressifyAI! Upload an image to get started.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)


        # Store uploaded image in session state
        if 'pil_image' not in st.session_state:
            st.session_state['pil_image'] = pil_image

        if st.button("Run Detection") or ('detected_img' in st.session_state and 'box' in st.session_state):
            if 'detected_img' not in st.session_state or 'box' not in st.session_state:
                detected_img, box = YOLOObjectDetection(st.session_state['pil_image'])
                st.session_state['detected_img'] = detected_img
                st.session_state['box'] = box
            else:
                detected_img = st.session_state['detected_img']
                box = st.session_state['box']

            st.image(detected_img, caption="Detected Objects", use_column_width=True)
            st.write("Bounding Box Coordinates:")

            if st.button("Show ROI Points") or ('roi_image' in st.session_state and 'roi_points' in st.session_state):
                st.write(f"X: {box[0]}, Y: {box[1]}, Width: {box[2]}, Height: {box[3]}")
                if 'roi_image' not in st.session_state or 'roi_points' not in st.session_state:
                    roi_image, roi_points = get_roi_points(st.session_state['pil_image'], box)
                    st.session_state['roi_image'] = roi_image
                    st.session_state['roi_points'] = roi_points
                else:
                    roi_image = st.session_state['roi_image']
                    roi_points = st.session_state['roi_points']
                st.image(roi_image, caption="Image with ROI Points", use_column_width=True)

                if roi_points is not None:
                    segmentation_mask = get_segmentation_mask(st.session_state['pil_image'], roi_points)
                    st.image(segmentation_mask, caption="Segmentation Mask", use_column_width=True)

if __name__ == "__main__":
    main()