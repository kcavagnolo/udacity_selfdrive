import numpy as np
import cv2
from scipy import signal


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        derivative = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        derivative = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    else:
        print("Error: orient must be either x or y.")
        derivative = 0

    # 3) Take the absolute value of the derivative or gradient
    abs_derivative = np.absolute(derivative)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_derivative / np.max(abs_derivative))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    # So there are 1s where #s are within our thresholds and 0s otherwise.
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return grad_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    abs_grad_dir = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(abs_grad_dir)
    dir_binary[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return dir_binary


def mag_thresh(img, sobel_kernel=9, mag_thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # 3) Calculate the magnitude
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)

    # 5) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))

    # 6) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    # 7) Return this mask as your binary_output image
    return mag_binary


def get_pixel_in_window(img, x_center, y_center, size):
    """
    returns selected pixels inside a window.
    :param img: binary image
    :param x_center: x coordinate of the window center
    :param y_center: y coordinate of the window center
    :param size: size of the window in pixel
    :return: x, y of detected pixels
    """
    half_size = size // 2
    window = img[int(y_center - half_size):int(y_center + half_size), int(x_center - half_size):int(x_center + half_size)]
    x, y = (window.T == 1).nonzero()
    x = x + x_center - half_size
    y = y + y_center - half_size
    return x, y


def add_pixels_given_centres(x_centres, y_centres, x_array, y_array, image, window_radius):
    for x_centre, y_centre in zip(x_centres, y_centres):
        x_additional, y_additional = get_pixel_in_window(image, x_centre,
                                                         y_centre, window_radius)
        return x_array.append(x_additional), y_array.append(y_additional)


def evaluate_poly(indep, poly_coeffs):
    return poly_coeffs[0]*indep**2 + poly_coeffs[1]*indep + poly_coeffs[2]


def highlight_lane_line_area(mask_template, left_poly, right_poly, start_y=0, end_y =720):
    area_mask = mask_template
    for y in range(start_y, end_y):
        left = evaluate_poly(y, left_poly)
        right = evaluate_poly(y, right_poly)
        area_mask[y][int(left):int(right)] = 1
    return area_mask


def add_figures_to_image(img, c, v, mc):
    font = cv2.FONT_HERSHEY_SIMPLEX
    lr = "left" if v < 0 else "right"
    cv2.putText(img, 'Curvature: {:.0f} [m]'.format(c), (50, 50), font, 1, (255, 255, 255), 2)
    cv2.putText(img, 'Centerline: {:.2f} [m] {}'.format(np.abs(v), lr), (50, 100), font, 1, (255, 255, 255), 2)
    cv2.putText(img, 'Min. Curvature: {:.0f} [m]'.format(mc), (50, 150), font, 1, (255, 255, 255), 2)


def lane_poly(yval, poly_coeffs):
    """Returns x value for poly given a y-value.
    Note here x = Ay^2 + By + C."""
    return poly_coeffs[0]*yval**2 + poly_coeffs[1]*yval + poly_coeffs[2]


def draw_poly(img, poly, poly_coeffs, steps, color=[255, 0, 0], thickness=10, dashed=False):
    img_height = img.shape[0]
    pixels_per_step = img_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start, poly_coeffs=poly_coeffs)), start)
        end_point = (int(poly(end, poly_coeffs=poly_coeffs)), end)

        if dashed == False or i % 2 == 1:
            img = cv2.line(img, end_point, start_point, color, thickness)
    return img


def binarize(image, gray_thresh=(20, 255), s_thresh=(170, 255), l_thresh=(30, 255), sobel_kernel=3):
    # first we take a copy of the source iamge
    image_copy = np.copy(image)

    # convert RGB image to HLS color space.
    # HLS more reliable when it comes to find out lane lines
    hls = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]

    # Next, we apply Sobel operator in X direction and calculate scaled derivatives.
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Next, we generate a binary image based on gray_thresh values.
    thresh_min = gray_thresh[0]
    thresh_max = gray_thresh[1]
    sobel_x_binary = np.zeros_like(scaled_sobel)
    sobel_x_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Next, we generated a binary image using S component of our HLS color scheme and
    # provided S threshold
    s_binary = np.zeros_like(s_channel)
    s_thresh_min = s_thresh[0]
    s_thresh_max = s_thresh[1]
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Next, we generated a binary image using S component of our HLS color scheme and
    # provided S threshold
    l_binary = np.zeros_like(l_channel)
    l_thresh_min = l_thresh[0]
    l_thresh_max = l_thresh[1]
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    # finally, return the combined binary image
    binary = np.zeros_like(sobel_x_binary)
    binary[((l_binary == 1) & (s_binary == 1) | (sobel_x_binary == 1))] = 1
    binary = 255 * binary.astype('uint8')  # single channel
    #binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')  # three channel

    return noise_reduction(binary)


def noise_reduction(image, threshold=4):
    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    nb_neighbours = cv2.filter2D(image, ddepth=-1, kernel=k)
    image[nb_neighbours < threshold] = 0
    return image


def fit_lanes(warped):

    # build histo
    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
    midpoint = np.int(histogram.shape[0] / 2)
    
    # get left and right halves of the histogram
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # set windows
    nwindows = 12
    window_height = np.int(warped.shape[0] / nwindows)
    margin = 75
    min_num_pixels = 35
    
    # Extracts x and y coordinates of non-zero pixels
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set current x coordinated for left and right
    leftx_current = leftx_base
    rightx_current = rightx_base

    # save pixel ids in these two lists
    left_lane_inds = []
    right_lane_inds = []

    # iterate over windows
    for window in range(nwindows):
        win_y_low = warped.shape[0] - (window + 1) * window_height
        win_y_high = warped.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > min_num_pixels:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > min_num_pixels:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the ndarrays of indices
    left_lane_array = np.concatenate(left_lane_inds)
    right_lane_array = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_array]
    lefty = nonzeroy[left_lane_array]
    rightx = nonzerox[right_lane_array]
    righty = nonzeroy[right_lane_array]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    fity = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    fit_leftx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
    fit_rightx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

    return fit_leftx, left_fit, fit_rightx, right_fit

def road_info(size, left_coeffs, right_coeffs, left_fit, right_fit):

    # first we calculate the intercept points at the bottom of our image
    left_intercept = left_coeffs[0] * size[0] ** 2 + left_coeffs[1] * size[0] + left_coeffs[2]
    right_intercept = right_coeffs[0] * size[0] ** 2 + right_coeffs[1] * size[0] + right_coeffs[2]
    
    # Next take the difference in pixels between left and right interceptor points
    road_width_in_pixels = right_intercept - left_intercept
    assert road_width_in_pixels > 0, 'Road width in pixel can not be negative'
        
    # Since average highway lane line width in US is about 3.7m
    # Source: https://en.wikipedia.org/wiki/Lane#Lane_width
    # we calculate length per pixel in meters
    meters_per_pixel_x_dir = 3.7 / road_width_in_pixels
    meters_per_pixel_y_dir = 60 / road_width_in_pixels

    # Recalculate road curvature in X-Y space
    ploty = np.linspace(0, 719, num=720)
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * meters_per_pixel_y_dir, left_fit * meters_per_pixel_x_dir, 2)
    right_fit_cr = np.polyfit(ploty * meters_per_pixel_y_dir, right_fit * meters_per_pixel_x_dir, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * meters_per_pixel_y_dir + left_fit_cr[1]) ** 2) ** 1.5) / \
                        np.absolute(2 * left_fit_cr[0])

    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * meters_per_pixel_y_dir + right_fit_cr[1]) ** 2) ** 1.5) / \
                         np.absolute(2 * right_fit_cr[0])
    curvature = (left_curverad + right_curverad) / 2.0
    min_curvature = min(left_curverad, right_curverad)

    # Next, we can lane deviation
    centerline = (left_intercept + right_intercept) / 2.0
    lane_deviation = (centerline - (size[0] / 2.0)) * meters_per_pixel_x_dir

    return curvature, min_curvature, lane_deviation