import tifffile
import numpy as np
import matplotlib.pyplot as plt
import math
import time

def bilinear_maibo(input_file, dim):
    # Deal with the input parameter and produce np data type
    image = tifffile.imread(input_file)
    image_data = np.array(image)
    dim_temp = [round(x) for x in dim]

    # Initialize the output image
    dim_out = np.array(dim_temp)
    Output_image = np.zeros(dim_out)

    # The bilinear algorithm
    h_factor = (np.shape(image_data)[0] - 1) / (dim_temp[0] - 1)
    w_factor = (np.shape(image_data)[1] - 1) / (dim_temp[1] - 1)
    for i in range(dim_temp[0]):
        for j in range(dim_temp[1]):
            x = i * h_factor
            y = j * w_factor
            x_deci, x_int = math.modf(x)
            y_deci, y_int = math.modf(y)
            x1 = int(x_int)
            x2 = int(x_int) + (x_deci != 0)
            y1 = int(y_int)
            y2 = int(y_int) + (y_deci != 0)
            if (x1 == x2) & (y1 == y2):
                Output_image[i][j] = image[int(x)][int(y)]
            elif (x1 == x2) & (y1 != y2):
                Output_image[i][j] = (image_data[int(x)][y1].astype(np.uint16) + image_data[int(x)][y2].astype(np.uint16)) / 2
            elif (x1 != x2) & (y1 == y2):
                Output_image[i][j] = (image_data[x1][int(y)].astype(np.uint16) + image_data[x2][int(y)].astype(np.uint16)) / 2
            else:
                Output_image[i][j] = (image_data[x1][y1] * (x2 - x) * (y2 - y) +
                                      image_data[x2][y1] * (x - x1) * (y2 - y) +
                                      image_data[x1][y2] * (x2 - x) * (y - y1) +
                                      image_data[x2][y2] * (x - x1) * (y - y1))

    return Output_image


# The following is the Main function for testing
# 1. Prepare the input
image = tifffile.imread('rice.tif')
image_data = np.array(image)
Input_vector1 = (256*1.7, 256*1.7)
Input_vector2 = (256*0.7, 256*0.7)

# 2. The time for the program to run
start_time = time.perf_counter()  
Output_image1 = bilinear_maibo('rice.tif', Input_vector1)
Output_image2 = bilinear_maibo('rice.tif', Input_vector2)
end_time = time.perf_counter() 
elapsed_time = end_time - start_time  
print(f"Bilinear运行时间: {elapsed_time} 秒")

# 3. The result plotting
fig, axes = plt.subplots(1, 3)
axes[0].imshow(image_data, cmap='Greys')
axes[0].set_title('The original image')
axes[1].imshow(Output_image1, cmap='Greys')
axes[1].set_title('The enlarged image after bilinear interpolation')
axes[2].imshow(Output_image2, cmap='Greys')
axes[2].set_title('The shrunk image after bilinear interpolation')
plt.tight_layout()    
plt.show()

# 4. File saving
Output_image1 = np.asarray(Output_image1, dtype=np.uint8)
Output_image2 = np.asarray(Output_image2, dtype=np.uint8)
tifffile.imwrite('enlarged_bilinear_maibo.tif', Output_image1)
tifffile.imwrite('shrunk_bilinear_maibo.tif', Output_image2)
