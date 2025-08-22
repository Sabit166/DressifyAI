import cv2
from PIL import Image as PILImage
from PIL import Image as PILImage



def get_roi_points(image, box):
    x1, y1, x2, y2 = box
    centre_x = int((x1 + x2) // 2)
    centre_y = int((y1 + y2) // 2)

    color = (0, 255, 0)
    radius = 30

    # Convert PIL image to OpenCV format
    import numpy as np
    image_cv = np.array(image)
    image_cv = image_cv[:, :, ::-1].copy()  # RGB to BGR

    # Draw the points on the image.
    p1 = (centre_x, centre_y)
    p2 = (centre_x, int((centre_y + y1) // 2))
    p3 = (centre_x, int((centre_y + y2) // 2))
    p4 = (int((centre_x + x1) // 2), centre_y)
    p5 = (int((centre_x + x2) // 2), centre_y)

    points = [p1, p2, p3, p4, p5]

    import cv2
    cv2.circle(image_cv, p1, radius, color, -1)
    cv2.circle(image_cv, p2, radius, color, -1)
    cv2.circle(image_cv, p3, radius, color, -1)
    cv2.circle(image_cv, p4, radius, color, -1)
    cv2.circle(image_cv, p5, radius, color, -1)

    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    roi_output = PILImage.fromarray(image_rgb)
    # Optionally display or save the image here
    # roi_output.save(output_path)
    # display(roi_output)
    return roi_output, points