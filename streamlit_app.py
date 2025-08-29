import streamlit as st
from PIL import Image
import io
from models.detector import YOLOObjectDetection  # import your detector function
#from models.ROI_Points import get_roi_points  # import your ROI points function
from models.segmentation_mask import segmentation_mask
from models.image_generation import image_generation
import numpy as np
import cv2
from PIL import Image as PILImage



def main():
    st.title("DressifyAI")
    st.write("Welcome to DressifyAI! Upload an image to get started.")
    
    # Prompt input
    prompt = st.text_input("Enter your prompt:", value="")
    st.session_state['prompt'] = prompt

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        st.session_state['pil_image'] = pil_image
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)


        # Store uploaded image in session state
        if 'pil_image' not in st.session_state:
            st.session_state['pil_image'] = pil_image

        if prompt and (st.button("Run Detection") or ('detected_img' in st.session_state and 'box' in st.session_state)):
            if 'detected_img' not in st.session_state or 'box' not in st.session_state:
                detected_img, box = YOLOObjectDetection(st.session_state['pil_image'])
                st.session_state['detected_img'] = detected_img
                st.session_state['box'] = box
            else:
                detected_img = st.session_state['detected_img']
                box = st.session_state['box']

            st.image(detected_img, caption="Detected Objects", use_column_width=True)
            st.write("Bounding Box Coordinates:")

            # Uncomment and fix ROI Points logic if needed
            # if st.button("Show ROI Points") or ('roi_image' in st.session_state and 'roi_points' in st.session_state):
            #     st.write(f"X: {box[0]}, Y: {box[1]}, Width: {box[2]}, Height: {box[3]}")
            #     if 'roi_image' not in st.session_state or 'roi_points' not in st.session_state:
            #         roi_image, roi_points = get_roi_points(st.session_state['pil_image'], box)
            #         st.session_state['roi_image'] = roi_image
            #         st.session_state['roi_points'] = roi_points
            #     else:
            #         roi_image = st.session_state['roi_image']
            #         roi_points = st.session_state['roi_points']
            #     st.image(roi_image, caption="Image with ROI Points", use_column_width=True)

            # Assuming roi_points is defined elsewhere in your logic
            #roi_points = st.session_state.get('roi_points', None)
            if box is not None:
                if st.button("Show Segmentation Mask"):
                    # Call segmentation_mask with the PIL image and ROI points
                    mask = segmentation_mask(st.session_state['pil_image'], box)
                    st.session_state['segmentation_mask'] = mask

                # Always show the segmentation mask if available
                if 'segmentation_mask' in st.session_state:
                    st.image(st.session_state['segmentation_mask'], caption="Segmentation Mask", use_column_width=True)
                    if st.button("Generate Image"):
                        # Call image_generation with image, mask, and prompt
                        generated_image = image_generation(
                            st.session_state['pil_image'],
                            st.session_state['segmentation_mask'],
                            st.session_state['prompt']
                        )
                        st.image(generated_image, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()