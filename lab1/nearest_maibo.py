import tifffile
import numpy as np
import matplotlib.pyplot as plt
import time

def nearest_maibo(input_file, dim):
    # Deal with the input parameter and produce np data type
    image = tifffile.imread(input_file)
    image_data = np.array(image)
    dim_temp = [round(x) for x in dim]

    # Initialize the output image
    dim_out = np.array(dim_temp)
    Output_image = np.zeros(dim_out)

    # The nearest algorithm
    h_factor = (np.shape(image_data)[0] - 1) / (dim_temp[0] - 1)
    w_factor = (np.shape(image_data)[1] - 1) / (dim_temp[1] - 1)
    for i in range(dim_temp[0]):
        for j in range(dim_temp[1]):
            x_index = round(i * h_factor)
            y_index = round(j * w_factor)
            Output_image[i][j] = image_data[x_index][y_index]

    return Output_image


# The following is the Main function for testing
# 1. Prepare the input
image = tifffile.imread('rice.tif')
image_data = np.array(image)
Input_vector1 = (256*1.7, 256*1.7)
Input_vector2 = (256*0.7, 256*0.7)

# 2. The time for the program to run
start_time = time.perf_counter()  
Output_image1 = nearest_maibo('rice.tif', Input_vector1)
Output_image2 = nearest_maibo('rice.tif', Input_vector2)
end_time = time.perf_counter() 
elapsed_time = end_time - start_time  
print(f"Nearest运行时间: {elapsed_time} 秒")

# 3. The result plotting
fig, axes = plt.subplots(1, 3)
axes[0].imshow(image_data, cmap='Greys')
axes[0].set_title('The original image')
axes[1].imshow(Output_image1, cmap='Greys')
axes[1].set_title('The enlarged image after nearest interpolation')
axes[2].imshow(Output_image2, cmap='Greys')
axes[2].set_title('The shrunk image after nearest interpolation')
plt.tight_layout()   
plt.show()

# 4. File saving
Output_image1 = np.asarray(Output_image1, dtype=np.uint8)
Output_image2 = np.asarray(Output_image2, dtype=np.uint8)
tifffile.imwrite('enlarged_nearest_maibo.tif', Output_image1)
tifffile.imwrite('shrunk_nearest_maibo.tif', Output_image2)