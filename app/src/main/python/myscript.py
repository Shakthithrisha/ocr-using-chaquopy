import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import base64
import io

# Match contours to license plate or character template


def find_contours(dimensions, img):
    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    ii = cv2.imread('contour.jpg')

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intx, inty, intwidth, intheight = cv2.boundingRect(cntr)

        # checking the dimensions of the contour to filter out the characters by contour's size
        if lower_width < intwidth < upper_width and lower_height < intheight < upper_height:
            x_cntr_list.append(
                intx)  # stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44, 24))
            # extracting each character using the enclosing rectangle's coordinates.
            character = img[inty:inty + intheight, intx:intx + intwidth]
            character = cv2.resize(character, (20, 40))

            cv2.rectangle(ii, (intx, inty), (intwidth + intx, inty + intheight), (50, 21, 200), 2)
            plt.imshow(ii, cmap='gray')

            #  Make result formatted for classification: invert colors
            character = cv2.subtract(255, character)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = character
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)  # List that stores the character's binary image (unsorted)

    # Return characters on ascending order with respect to the x-coordinate (most-left character first)

    plt.show()
    # arbitrary function that stores sorted list of character indices
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])  # stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res


def extract_plate(image):
    plate_img = image.copy()

    plate_cascade = cv2.CascadeClassifier('C:\\indian_license_plate.xml')

    # detects number plates and returns the coordinates and dimensions of detected license plate's contours

    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.3, minNeighbors=7)

    for (x, y, w, h) in plate_rect:
        a, b = (int(0.02 * img.shape[0]), int(0.025 * img.shape[1]))  # parameter tuning
        plate = plate_img[y + a:y + h - a, x + b:x + w - b, :]
        # finally representing the detected contours by drawing rectangles around the edges.
        cv2.rectangle(plate_img, (x, y), (x + w, y + h), (51, 51, 255), 3)
    # plt.im show(plate, c map='gray')
    # plt.show()
    # cv2.im write('plate.jpg',plate)

    return plate  # returning the processed image


# Find characters in the resulting images


def segment_characters(image):
    image = extract_plate(image)
    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    lp_width = img_binary_lp.shape[0]
    lp_height = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [lp_width / 6, lp_width / 2, lp_height / 10, 2 * lp_height / 3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.show()
    cv2.imwrite('contour.jpg', img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list


def main(data):
    decoded_data = base64.b64decode(data)
    np_data = np.fromstring(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    pil_im = Image.fromarray(segment_characters(img))

    buff = io.BytesIO
    pil_im.save = (buff, format("JPEG"))
    img_str = base64.b64encode(buff.getvalue())
    return""+str(img_str, "utf-8")