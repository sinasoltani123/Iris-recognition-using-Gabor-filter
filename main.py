import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from skimage import exposure
from skimage.filters import gaussian
from scipy import ndimage

def preprocess_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5),1.2)
    return blurred_image,image
    
def detect_iris_inner(image, center, radius=50):
    
    # Create a mask with the same dimensions as the image, initialized to zeros (black)
    mask = np.zeros_like(image)

    # Draw a white filled circle on the mask where the circle should be
    cv2.circle(mask, center, radius, (255, 255, 255), -1)

    # Mask the inner regions with white color
    white_inner_region = np.ones_like(image) * 255
    inner_regions_white = cv2.bitwise_and(white_inner_region, mask)

    # Mask the outer region (outside the circle) with the original image
    outer_region = cv2.bitwise_and(image, cv2.bitwise_not(mask))

    # Combine the outer regions and the inner region
    result = cv2.add(outer_region, inner_regions_white)

    return result

def detect_iris_outer(image):
    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 4000,
    param1=40, param2=60,
    minRadius=40, maxRadius=200)
    

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw the circle in the output image
            cv2.circle(image, (x, y), r, (255, 0, 0), 2)
            # Draw a small circle (of radius 3) to show the center
            cv2.circle(image, (x, y), 3, (0, 255, 0), 3)
        return image, circles
    return image, None

def plot_images(original, localized_iris_outer,crop_Iris, localized_iris_inner,normalized_iris,gaussian,
                title1='Original Image', title2='Outer Iris detected',title3="Crop Outer Iris",
                title4='Crop Inner Iris', title5="normalized",title6=" enhanced &\ndenoised"):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 6, 1)
    plt.title(title1)
    plt.imshow(original, cmap='gray')
    
    plt.subplot(1, 6, 2)
    plt.title(title2)
    plt.imshow(localized_iris_outer, cmap='gray')

    plt.subplot(1, 6, 3)
    plt.title(title3)
    plt.imshow(crop_Iris, cmap='gray')

    plt.subplot(1, 6, 4)
    plt.title(title4)
    plt.imshow(localized_iris_inner, cmap='gray')

    plt.subplot(1, 6, 5)
    plt.title(title5)
    plt.imshow(normalized_iris, cmap='gray')

    plt.subplot(1, 6, 6)
    plt.title(title6)
    plt.imshow(gaussian, cmap='gray')
    plt.show()

def crop_Iris(image, center, radius):

    # Create a mask with the same dimensions as the image, initialized to zeros (black)
    mask = np.zeros_like(image)

    # Draw a white filled circle on the mask where the circle should be
    cv2.circle(mask, center, radius, (255, 255, 255), -1)

    # Create the inverse of the mask
    mask_inv = cv2.bitwise_not(mask)

    # Mask the outer regions with white color
    white_background = np.ones_like(image) * 255
    outer_regions_white = cv2.bitwise_and(white_background, mask_inv)

    # Mask the inner region (the circle) with the original image
    inner_region = cv2.bitwise_and(image, mask)

    # Combine the outer regions and the inner region
    result = cv2.add(outer_regions_white, inner_region)

    return result

def daugman_normalizaiton(image, cx, cy, rp, ri, num_radial_points=64, num_angular_points=360):
    '''
    Normalize the iris region using Daugman's rubber sheet model. fixed size = (Height: 64 and Width: 360)
    Radius of the pupil = rp
    Radius of the iris = ri
    Center of the pupil = cx , cy
    '''
    theta = np.linspace(0, 2 * np.pi, num_angular_points)
    r = np.linspace(rp, ri, num_radial_points)
    normalized_iris = np.zeros((num_radial_points, num_angular_points), dtype=image.dtype)

    for i in range(num_radial_points):
        for j in range(num_angular_points):
            x = int(cx + r[i] * np.cos(theta[j]))
            y = int(cy + r[i] * np.sin(theta[j]))
            normalized_iris[i, j] = image[y, x]
    
    return normalized_iris


def apply_gabor_filter(image, ksize=31, sigma=4.0, theta=0, lambd=10.0, gamma=0.5, psi=0):
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
    return filtered_image

    
def apply_local_histogram_equalization_low_pass_Gaussian_filter(image):
    # Apply local histogram equalization
    image_eq = exposure.equalize_adapthist(image, clip_limit=0.03)

    # Apply low-pass Gaussian filter
    image_gaussian = gaussian(image_eq, sigma=2)
    return image_gaussian


def gabor_filter_bank(ksize=21, sigma=5.0, lambd=10.0, gamma=0.5):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 8):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        filters.append(kern)
    return filters

def apply_gabor_filters(image, filters):
    responses_real = []
    responses_imag = []
    for kern in filters:
        fimg_real = cv2.filter2D(image, cv2.CV_32F, kern)
        fimg_imag = cv2.filter2D(image, cv2.CV_32F, np.imag(np.exp(1j * kern)))
        responses_real.append(fimg_real)
        responses_imag.append(fimg_imag)
    return responses_real, responses_imag

def generate_feature_vector(responses_real, responses_imag):
    combined_response = np.stack(responses_real + responses_imag, axis=-1)
    binary_vector = (combined_response > 0).astype(int)
    #To reduce this to 2048 bits, downsampling or selecting a subset of the bits can be performed. But I used the whole response.
    return binary_vector.flatten()

def hamming_distance(vec1, vec2):
    # print(len(vec1),'\n',len(vec2))
    return np.sum(vec1 != vec2) / len(vec1)


def is_match(distance, threshold):
    return distance <= threshold

def before_Gabor(image_path):
    preprocessed_image,original_image = preprocess_image(image_path)
    localized_iris_outer, circles_outer = detect_iris_outer(preprocessed_image)
    
    (x,y,radius) =circles_outer[0]
    center = (x,y)
    cropped_image =crop_Iris(localized_iris_outer, center, radius)
    

    localized_iris_inner = detect_iris_inner(cropped_image,center)

    # rubber_sheet_normalization daugman's method
    normalized_iris = daugman_normalizaiton(localized_iris_inner,x,y,50,radius)
    
    Gaussian =apply_local_histogram_equalization_low_pass_Gaussian_filter(normalized_iris)


    # Plot the results
    plot_images(original_image, localized_iris_outer,cropped_image,localized_iris_inner,normalized_iris,Gaussian)
    return Gaussian


# Example usage
# image_path = 'UBIRIS_800_600\Sessao_1/1/Img_1_1_2.jpg'

# Create the database
iris_database = {}
# Parameters
threshold = 0.3  # Example threshold value, adjust based on your analysis
filters = gabor_filter_bank()

iris_images =[]
#loop on a subset of MMU database for creating the database of Iris codes
for i in range(1,22):

    folder_path = "images/{}/".format(i)

    #creating images pathes 
    path_to_image_1= folder_path+"1.jpg"
    path_to_image_2= folder_path+"2.jpg"
    iris_images.append((i,path_to_image_1,path_to_image_2))
    
    Image_before_Gabor = before_Gabor(path_to_image_1)

    # Apply Gabor filters
    filters = gabor_filter_bank()
    responses_real, responses_imag = apply_gabor_filters(Image_before_Gabor, filters)

    # Generate feature vector
    feature_vector = generate_feature_vector(responses_real, responses_imag)
    # print(len(feature_vector))
    iris_database[i] = feature_vector
    

# Test the images using Hamming distance
for i in range(1,22):
    folder_path = "images/{}/".format(i)

    path_to_image_2= folder_path+"2.jpg"
    
    Image_before_Gabor = before_Gabor(path_to_image_2)

    # Apply Gabor filters
    filters = gabor_filter_bank()
    responses_real, responses_imag = apply_gabor_filters(Image_before_Gabor, filters)

    # Generate feature vector
    feature_vector = generate_feature_vector(responses_real, responses_imag)

    # Compare with the database
    stored_feature_vector = iris_database[i]
    distance = hamming_distance(feature_vector, stored_feature_vector)
    match = is_match(distance, threshold)

    print(f"Person {i}: {'Match' if match else 'No Match'} with a Hamming distance of {distance:.4f}")


