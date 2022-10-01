from scipy import ndimage
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Helper Functions

def BGR2RGB(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image
    
def vis_hybrid_image(hybrid_image):
    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    # counting how many color channels the input has
    num_colors = hybrid_image.shape[2]
    output = hybrid_image
    cur_image = hybrid_image

    for i in range(2, scales):
        # add padding
        output = np.concatenate(
            (output, np.ones((original_height, padding, num_colors), dtype=int)), axis=1)
        # dowsample image;
        width = int(cur_image.shape[1] * scale_factor)
        height = int(cur_image.shape[0] * scale_factor)
        dim = (width, height)
        cur_image = cv2.resize(cur_image, dim, interpolation=cv2.INTER_LINEAR)
        # pad the top and append to the output
        tmp = np.concatenate((np.ones(
            (original_height-cur_image.shape[0], cur_image.shape[1], num_colors)), cur_image), axis=0)
        output = np.concatenate((output, tmp), axis=1)

    output = toUint8(output)
    return output


def read_image_as_float(image_path):
    img1 = cv2.imread(image_path)
    return img1.astype(np.float64)


def gaussian_2D_filter(filter_size, sigma):
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size//2
    n = filter_size//2
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2 * sigma**2))
            gaussian_filter[x+m, y+n] = (1/x1)*x2

    return gaussian_filter


def imgfilter(image, filter):
    img = cv2.filter2D(image, -1, filter)
    return img


def log_mag_FFT(image):
    output = np.log(np.abs(np.fft.fftshift(np.fft.fft2(image))))
    return output


def FFT(image):
    dft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift


def IFFT(image):
    f_ishift = np.fft.ifftshift(image)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return img_back


def toUint8(image):
    img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)  # normalize the data to 0 - 1
    return img


def read_image_as_gray(image_path):
    img1 = cv2.imread(image_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    return img1.astype(np.float64)


def applyGfilter(image, cutoff_frequency):
    filter_size = cutoff_frequency*4+1
    filter = gaussian_2D_filter(filter_size, cutoff_frequency)
    img = imgfilter(image, filter)
    return img


def pyramidsGL(image, num_levels, cutoff_frequency):
    ''' Creates Gaussian (G) and Laplacian (L) pyramids of level "num_levels" from image im. 
    G and L are list where G[i], L[i] stores the i-th level of Gaussian and Laplacian pyramid, respectively. '''

    L = []

    G = [image]

    for i in range(num_levels):
        G_smooth = applyGfilter(G[i], cutoff_frequency)
        G.append(cv2.resize(
            G_smooth, (G[i].shape[1]//2, G[i].shape[0]//2), interpolation=cv2.INTER_NEAREST))
        G_upsample = cv2.resize(
            G[i+1], (G[i].shape[1], G[i].shape[0]), interpolation=cv2.INTER_NEAREST)
        G_upsample_smooth = applyGfilter(G_upsample, cutoff_frequency)
        L.append(G[i]-G_upsample_smooth)

    return G, L


def displayPyramids(G, L):
    '''Role of this function is to display intensity and Fast Fourier Transform (FFT) images of pyramids.'''
    length = len(G)
    rows = 2
    columns = 6
    result_path = "output/GL_pyramid"
    fig = plt.figure(figsize=(20, 6))

    for i in range(length):
      
      fig.add_subplot(rows, columns, i + 1)
      plt.imshow(G[i], cmap='gray')
      plt.axis('off')
      if i < len(G) - 1:
        fig.add_subplot(rows, columns, i + 1 + length)
        plt.imshow(L[i], cmap='gray')
        plt.axis('off')
    fig.show()
    fig.savefig(f'{result_path}/G_L_Pyramid.png')

    fig = plt.figure(figsize=(20, 6))

    for i in range(length):
      fig.add_subplot(rows, columns, i + 1)
      plt.imshow(log_mag_FFT(G[i]))
      plt.axis('off')
      if i < len(G) - 1:
        fig.add_subplot(rows, columns, i + 1 + length)
        plt.imshow(log_mag_FFT(L[i]))
        plt.axis('off')
    fig.show()
    fig.savefig(f'{result_path}/FFT_oF_G_L_Pyramid.png')

    return


def reconstructLaplacianPyramid(G, L, cutoff_frequency):
    '''Given a Laplacian Pyramid L, reconstruct an image img.'''
    for i in range(len(G)-1):
        G_upsample = cv2.resize(
            G[-1-i], (L[-1-i].shape[1], L[-1-i].shape[0]), interpolation=cv2.INTER_NEAREST)
        G_upsample_smooth = applyGfilter(G_upsample, cutoff_frequency)
        img = L[-1-i] + G_upsample_smooth
    return img


# Edge Detection

def gradientMagnitude(im, sigma):
    '''
    im: input image
    sigma: standard deviation value to smooth the image

    outputs: gradient magnitude and gradient direction of the image
    '''
    # smooth the RGB image
    # im = applyGfilter(im, sigma)
    x_grad = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=np.float64)

    y_grad = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])

    gaussain = gaussian_2D_filter(3, sigma)

    kernel_x = gaussain*x_grad
    kernel_y = gaussain*y_grad

    image_r = im[:, :, 2]
    image_g = im[:, :, 1]
    image_b = im[:, :, 0]

    image_2D_shape = image_r.shape

    image_r_x_gradient = imgfilter(image_r, kernel_x).flatten()
    image_g_x_gradient = imgfilter(image_g, kernel_x).flatten()
    image_b_x_gradient = imgfilter(image_b, kernel_x).flatten()

    image_r_y_gradient = imgfilter(image_r, kernel_y).flatten()
    image_g_y_gradient = imgfilter(image_g, kernel_y).flatten()
    image_b_y_gradient = imgfilter(image_b, kernel_y).flatten()

    r_norm = np.zeros_like(image_r_x_gradient, dtype=np.uint8)
    g_norm = np.zeros_like(image_g_x_gradient, dtype=np.uint8)
    b_norm = np.zeros_like(image_b_x_gradient, dtype=np.uint8)

    img_gradient = np.zeros_like(r_norm, dtype=np.uint8)
    img_orientation = np.zeros_like(r_norm, dtype=np.float64)

    for i in range(image_r_x_gradient.shape[0]):
        r_norm[i] = np.sqrt((image_r_x_gradient[i]**2) +
                            (image_r_y_gradient[i]**2))
        g_norm[i] = np.sqrt((image_g_x_gradient[i]**2) +
                            (image_g_y_gradient[i]**2))
        b_norm[i] = np.sqrt((image_b_x_gradient[i]**2) +
                            (image_b_y_gradient[i]**2))

        max_mag = np.argmax(np.array([r_norm[i], g_norm[i], b_norm[i]]))

        if max_mag == 0:
            img_gradient[i] = r_norm[i]
            img_orientation[i] = np.arctan2(
                image_r_y_gradient[i], image_r_x_gradient[i])
        if max_mag == 1:
            img_gradient[i] = g_norm[i]
            img_orientation[i] = np.arctan2(
                image_g_y_gradient[i], image_g_x_gradient[i])
        if max_mag == 2:
            img_gradient[i] = b_norm[i]
            img_orientation[i] = np.arctan2(
                image_b_y_gradient[i], image_b_x_gradient[i])

    grad_magnitude = img_gradient.reshape(image_2D_shape)
    grad_orientation = img_orientation.reshape(image_2D_shape)

    return grad_magnitude, grad_orientation


def edgeGradient(magnitude, orientation, canny):
    '''
    im: input image

    output: a soft boundary map of the image
    '''
    if canny:
        img = cv2.Canny(magnitude, 10, 130)

        return img

    else:
        #Non-maxima suppression
        magnitude = magnitude / magnitude.max() * 255
        non_maxima = np.zeros_like(magnitude, dtype=np.uint8)
        orientation = orientation * 180 / np.pi
        orientation[orientation < 0] += 180

        for i in range(magnitude.shape[0]):
            for j in range(magnitude.shape[1]):
                q = 255
                r = 255
                try:
                    # check for angle around zero
                    if (0 <= orientation[i, j] <= 22.5) or (157.5 <= orientation[i, j] <= 180.0):
                        # print("hi")
                        q = magnitude[i, j+1]
                        r = magnitude[i, j-1]

                    # check for 45 degrees
                    elif (22.5 <= orientation[i, j] < 67.5):
                        q = magnitude[i+1, j-1]
                        r = magnitude[i-1, j+1]

                    # check for 90 degrees
                    elif (67.5 <= orientation[i, j] < 112.5):
                        q = magnitude[i+1, j]
                        r = magnitude[i-1, j]

                    # check for 135 degrees
                    elif (112.5 <= orientation[i, j] < 157.5):
                        q = magnitude[i+1, j+1]
                        r = magnitude[i-1, j-1]

                    if magnitude[i, j] >= q and magnitude[i, j] >= r:

                        non_maxima[i, j] = magnitude[i, j]
                    else:
                        non_maxima[i, j] = 0
                except:
                    continue

        # cv2.imshow('edge', non_maxima)
        # cv2.waitKey(0)

        return non_maxima


# oriented edge detection:


def orientedFilterMagnitude(im, sigma):
    '''
    im: input image

    outputs: gradient magnitude and gradient direction of the image
    '''
    x_grad = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=np.float64)

    gaussain = gaussian_2D_filter(3, sigma)

    kernel_0 = gaussain*x_grad
    kernel_45 = ndimage.rotate(kernel_0, 45, reshape=False)
    kernel_90 = ndimage.rotate(kernel_0, 90, reshape=False)
    kernel_135 = ndimage.rotate(kernel_0, 135, reshape=False)

    image_r = im[:, :, 2]
    image_g = im[:, :, 1]
    image_b = im[:, :, 0]
    image_2D_shape = image_r.shape

    im_r_res_to_0 = imgfilter(image_r, kernel_0).flatten()
    im_g_res_to_0 = imgfilter(image_g, kernel_0).flatten()
    im_b_res_to_0 = imgfilter(image_b, kernel_0).flatten()

    im_r_res_to_45 = imgfilter(image_r, kernel_45).flatten()
    im_g_res_to_45 = imgfilter(image_g, kernel_45).flatten()
    im_b_res_to_45 = imgfilter(image_b, kernel_45).flatten()

    im_r_res_to_90 = imgfilter(image_r, kernel_90).flatten()
    im_g_res_to_90 = imgfilter(image_g, kernel_90).flatten()
    im_b_res_to_90 = imgfilter(image_b, kernel_90).flatten()

    im_r_res_to_135 = imgfilter(image_r, kernel_135).flatten()
    im_g_res_to_135 = imgfilter(image_g, kernel_135).flatten()
    im_b_res_to_135 = imgfilter(image_b, kernel_135).flatten()

    magnitude = np.zeros_like(im_r_res_to_0, dtype=np.uint8)
    orientation = np.zeros_like(im_r_res_to_0, dtype=np.uint8)

    r_norm_0_90 = np.zeros_like(im_r_res_to_0, dtype=np.uint8)
    r_norm_45_135 = np.zeros_like(im_r_res_to_0, dtype=np.uint8)
    g_norm_0_90 = np.zeros_like(im_r_res_to_0, dtype=np.uint8)
    g_norm_45_135 = np.zeros_like(im_r_res_to_0, dtype=np.uint8)
    b_norm_0_90 = np.zeros_like(im_r_res_to_0, dtype=np.uint8)
    b_norm_45_135 = np.zeros_like(im_r_res_to_0, dtype=np.uint8)

    for i in range(im_r_res_to_0.shape[0]):
        # response = np.array([im_r_res_to_0[i], im_g_res_to_0[i], im_b_res_to_0[i], im_r_res_to_45[i], im_g_res_to_45[i], im_b_res_to_45[i],
        #                    im_r_res_to_90[i], im_g_res_to_90[i], im_b_res_to_90[i], im_r_res_to_135[i], im_g_res_to_135[i], im_b_res_to_135[i]])
        # max = np.argmax(response)
        # magnitude[i] = response[max]
        
        r_norm_0_90[i] = np.sqrt((im_r_res_to_0[i]**2) +(im_r_res_to_90[i]**2))
        r_norm_45_135[i] = np.sqrt((im_r_res_to_45[i]**2) +(im_r_res_to_135[i]**2))

        g_norm_0_90[i] = np.sqrt((im_g_res_to_0[i]**2) +(im_g_res_to_90[i]**2))
        g_norm_45_135[i] = np.sqrt((im_g_res_to_45[i]**2) +(im_g_res_to_135[i]**2))

        b_norm_0_90[i] = np.sqrt((im_b_res_to_0[i]**2) +(im_b_res_to_90[i]**2))
        b_norm_45_135[i] = np.sqrt((im_b_res_to_45[i]**2) +(im_b_res_to_135[i]**2))

        norms = np.array([r_norm_0_90[i],r_norm_45_135[i],g_norm_0_90[i],g_norm_45_135[i],b_norm_0_90[i],b_norm_45_135[i]])
        max = np.argmax(norms)

        magnitude[i] = norms[max]
        if max == 0:
            orientation[i] = np.arctan2(im_r_res_to_90[i],im_r_res_to_0[i])   
        if max == 1:
            orientation[i] = np.arctan2(im_r_res_to_135[i],im_r_res_to_45[i])  
        if max == 2:
            orientation[i] = np.arctan2(im_g_res_to_90[i],im_g_res_to_0[i])   
        if max == 4:
            orientation[i] = np.arctan2(im_g_res_to_135[i],im_g_res_to_45[i]) 
        if max == 5:
            orientation[i] = np.arctan2(im_b_res_to_90[i],im_b_res_to_0[i])   
        if max == 6:
            orientation[i] = np.arctan2(im_b_res_to_135[i],im_b_res_to_45[i]) 

    magnitude = np.reshape(magnitude,image_2D_shape)
    orientation = np.reshape(orientation,image_2D_shape)

    return magnitude


def edgeOrientedFilters(im):
    '''
    im: input image

    output: a soft boundary map of the image
    '''
    img = cv2.Canny(im, 10, 130)

    return img
    


def ssd_matching(waldo, puzzle):

    puzzle = (cv2.cvtColor(puzzle, cv2.COLOR_BGR2GRAY)).astype(np.float64)
    waldo = (cv2.cvtColor(waldo, cv2.COLOR_BGR2GRAY)).astype(np.float64)

    k, l = waldo.shape
    ssd = (np.ones((puzzle.shape[0], puzzle.shape[1])))*255

    waldo_f = waldo.flatten()
    k, l = waldo.shape

    ssd = (np.ones((puzzle.shape[0]-k, puzzle.shape[1]-l)))*255
    for i in range(0, puzzle.shape[0]-k):
        for j in range(0, puzzle.shape[1]-l):
            window = puzzle[i:i+k, j:j+l].flatten()
            ssd[i, j] = np.sum((window-waldo_f)**2)

    return ssd
