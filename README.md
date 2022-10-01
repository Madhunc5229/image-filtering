# image-filtering
## This repository contains programs related to hybrid images, Gaussian and Laplacian pyramids, edge detection pipeline and template mathcing

### 1. Hybrid Images
#### To generate a hybrid image, we take low frequency of one image and high frequency of another image and add both the images 

Original Image 1             |  Original Image 2
:-------------------------:|:-------------------------:
![makeup_before](https://user-images.githubusercontent.com/61328094/193397429-6223bca1-7e98-4754-8612-e726aa088694.jpg)  |  ![makeup_after](https://user-images.githubusercontent.com/61328094/193397437-a728eb55-f5b9-431d-8847-aff47e4e4e48.jpg)

Low Frequency of Image 1             |  High Frequency of Image 2  | Hybrid Image
:-------------------------:|:-------------------------:|:-------------------------:|
![low_frequency_image](https://user-images.githubusercontent.com/61328094/193397724-2e94cb7c-8246-4ca5-bc1e-8350e62c2804.png)  |  !![high_frequency_image](https://user-images.githubusercontent.com/61328094/193397731-c96d361a-9039-419a-aac1-663e503f10ea.png)  |  ![hybrid_image](https://user-images.githubusercontent.com/61328094/193397742-9c814c0a-7616-4005-8352-0b8591c51730.png)


### 2. Gaussian and Laplacian Pyramid for lossless rescontsruction
#### Constructing a Gaussian and Laplacian pyramids help preserve the important features of an image which helps in lossless reconstruction
Original Image            | 
:-------------------------:|
![reconstructedImage](https://user-images.githubusercontent.com/61328094/193397880-01609ef8-346e-453c-9cd6-bd4885eb5949.png) | 

Gaussian and Laplacian Pyramid:
![G_L_Pyramid](https://user-images.githubusercontent.com/61328094/193397904-746efa69-1e00-49e5-8a18-6e3578736f85.png)  

FFT magnitude of Gaussian and Laplacian Pyramid:
![FFT_oF_G_L_Pyramid](https://user-images.githubusercontent.com/61328094/193397918-76291a85-afc1-4850-a82b-3973be7bdc0b.png)


