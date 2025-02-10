import cv2
import numpy as np
import os



def border_maker(filename):
    img = cv2.imread(filename)
    save = "border.txt"
    save_path = os.path.join(save)
    # imgflipped = cv2.flip(img, 0)

    image_height, image_width, _ = img.shape


    # Define the color ranges for each color of interest for creating masks.
    COLOR1_RANGE = [(209, 152, 116), (289, 232, 196)]  # Blue in BGR, [(low), (high)].
    COLOR2_RANGE = [(158, 186, 147), (238, 266, 227)]  # Green in BGR, [(low), (high)].
    COLOR3_RANGE = [(209, 152, 116), (289, 232, 196)]   # Blue in BGR, [(low), (high)].
    COLOR4_RANGE = [(0, 0, 0), (20, 20, 20)]  # Green in BGR, [(low), (high)].



    # Create masks:
    color1_mask = cv2.inRange(img, COLOR1_RANGE[0], COLOR1_RANGE[1])
    color2_mask = cv2.inRange(img, COLOR2_RANGE[0], COLOR2_RANGE[1])
    color3_mask = cv2.inRange(img, COLOR3_RANGE[0], COLOR3_RANGE[1])
    color4_mask = cv2.inRange(img, COLOR4_RANGE[0], COLOR4_RANGE[1])

    # Adjust according to your adjacency requirement.
    kernel = np.ones((3, 3), dtype=np.uint8)

    # Dilating masks to expand boundary.
    color1_mask = cv2.dilate(color1_mask, kernel, iterations=1)
    color2_mask = cv2.dilate(color2_mask, kernel, iterations=1)
    color3_mask = cv2.dilate(color3_mask, kernel, iterations=1)
    color4_mask = cv2.dilate(color4_mask, kernel, iterations=1)

    # Required points now will have both color's mask val as 255.
    common1 = cv2.bitwise_and(color1_mask, color2_mask)
    common2 = cv2.bitwise_and(color3_mask, color4_mask)
    cv2.imwrite("test1.jpg", common1)
    cv2.imwrite("test2.jpg", common2)

    # Common is binary np.uint8 image, min = 0, max = 255.
    # SOME_THRESHOLD can be anything within the above range. (not needed though)
    # Extract/Use it in whatever way you want it.
    intersection_points1 = np.where(common1 > 120)
    intersection_points2 = np.where(common2 > 120)

    # Create a copy of the original image to draw on


    # Say you want these points in a list form, then you can do this.
    pts_list1 = [[r, c] for r, c in zip(*intersection_points1)]
    pts_list2 = [[r, c] for r, c in zip(*intersection_points2)]

    cleaned_coordinates = [[int(x) for x in pair] for pair in (pts_list1 + pts_list2)]
    # Fix the scaling by using y,x instead of x,y since the points are stored as [row,col]
    scaled_coordinates = [[int((y / image_width) * 100), int((x / image_height) * 100)] for x, y in cleaned_coordinates]

    with open(save_path, "w") as file:
        file.write(f"ob = {scaled_coordinates}\n")

    result_image = img.copy()

    # Draw intersection points on the image, converting from scaled coordinates back to image coordinates
    for coord in scaled_coordinates:
        # Convert from percentage back to pixel coordinates
        y = int((coord[0] * image_width) / 100)
        x = int((coord[1] * image_height) / 100)
        cv2.circle(result_image, (x, y), 1, (0, 0, 255), -1)  # Red dots for all intersections

    # Save the result
    cv2.imwrite("intersection_points.jpg", result_image)

    



border_maker("map.png")




