import cv2
import numpy as np

im = cv2.imread('abdomen.png')
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
w = im.shape[1]
h = im.shape[0]

mask = np.zeros([h, w], np.uint8)

def region_growing(image, seed_point, threshold):
    # 0 - not visited pixels
    # 1 - visited pixels
    visited = np.zeros_like(image)
    # Pixel value at the starting point
    seed_value = image[seed_point[1], seed_point[0]]
    stack = [seed_point]

    while stack:
        x, y = stack.pop()

        # Checking whether the pixel has already been visited
        if visited[y, x] == 1:
            continue

        # Checking if the pixel value is similar to the starting point value
        if abs(image[y, x] - seed_value) < threshold:
            visited[y, x] = 1

            # Adding neighboring pixels to the stack
            if x > 0:
                stack.append((x - 1, y))
            if x < image.shape[1] - 1:
                stack.append((x + 1, y))
            if y > 0:
                stack.append((x, y - 1))
            if y < image.shape[0] - 1:
                stack.append((x, y + 1))

    # Marking a designated area in the resulting image
    segmented_image = np.zeros_like(image)
    segmented_image[visited == 1] = 255

    return segmented_image

def mouse_callback(event, x, y, flags, param):
    global seed_point
    if event == cv2.EVENT_LBUTTONDOWN:
        seed_point = (x, y)
        segmented_image = region_growing(im_gray, seed_point, threshold=110)
        cv2.imshow('image', segmented_image)

cv2.imshow('image', im_gray)
cv2.setMouseCallback('image', mouse_callback)
cv2.waitKey()
cv2.destroyAllWindows()
