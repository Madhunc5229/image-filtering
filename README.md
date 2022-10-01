# image-filtering
 This repository contains programs related to:  
 hybrid images  
 Gaussian and Laplacian pyramids  
 Edge detection pipeline  
 Template mathcing

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

Reconstructed Image          | 
:-------------------------:|
![reconstructedImage](https://user-images.githubusercontent.com/61328094/193397880-01609ef8-346e-453c-9cd6-bd4885eb5949.png) | 


### 3. Edge detection pipeline
#### In this edge detection pipeline, the gradient magnitude was computed two ways, one using only x and y Gaussian derivative and the other using 4 oriented filters followed by non-maxima suppression. 
Original Image         | 
:-------------------------:|
![o](https://user-images.githubusercontent.com/61328094/193398046-997f6255-670e-48f4-8edd-2b238b40bf2a.png) | 

Magnitude using x and y Gaussian derivative            |  After Non-maxima suppresion
:-------------------------:|:-------------------------:
![m1](https://user-images.githubusercontent.com/61328094/193398095-291fd70e-f002-4147-bda4-6eea349fccd6.png) |  ![c1](https://user-images.githubusercontent.com/61328094/193398316-3a39d427-4056-47e1-8438-7b6127f44558.png)

Magnitude using oriented filters          |  After Non-maxima suppresion
:-------------------------:|:-------------------------:
![m2](https://user-images.githubusercontent.com/61328094/193398127-73375e4c-62f8-4d4c-9e1d-82461ba17dbe.png) |  ![c2](https://user-images.githubusercontent.com/61328094/193398134-8a082ffa-bdb2-4588-84b5-d41338a47054.png)

### 4. Template matching (finding waldo)
#### The goal in this program is to find waldo in a puzzle using SSD(sum of squared differences) as the matching criteria for template.

Waldo  
![waldo](https://user-images.githubusercontent.com/61328094/193398250-8c1caeae-deb3-4f70-a2f4-a1e3cee8cce7.png)  

Found in map using template matching        | 
:-------------------------:|
![foundP2](https://user-images.githubusercontent.com/61328094/193398476-35bf004e-037d-4c42-bdfe-faa7ec74ff58.png) | 







