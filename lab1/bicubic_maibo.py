import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import time

def bicubic_maibo(input_file, dim):
    # Deal with the input parameter and produce np data type
    image = tifffile.imread(input_file)
    image_data = np.array(image)
    dim_temp = [round(x) for x in dim]

    # The bicubic algorithm
    h_factor = (np.shape(image_data)[0] - 1) / (dim_temp[0] - 1)
    w_factor = (np.shape(image_data)[1] - 1) / (dim_temp[1] - 1)
    h = np.arange(np.shape(image_data)[0])
    w = np.arange(np.shape(image_data)[1])
    temp = interpolate.interp2d(h,w,image_data,kind='cubic')
    dim_out_x = np.arange(dim_temp[0])
    dim_out_y = np.arange(dim_temp[1])
    dim_out_x = dim_out_x * h_factor
    dim_out_y = dim_out_y * w_factor
    Output_image_temp = temp(dim_out_x,dim_out_y)
    Output_image_temp = np.asarray(Output_image_temp, dtype=np.uint8)
    Output_image = Output_image_temp

    return Output_image


# The following is the Main function for testing
# 1. Prepare the input
image = tifffile.imread('rice.tif')
image_data = np.array(image)
Input_vector1 = (256*1.7, 256*1.7)
Input_vector2 = (256*0.7, 256*0.7)

# 2. The time for the program to run
start_time = time.perf_counter()  
Output_image1 = bicubic_maibo('rice.tif', Input_vector1)
Output_image2 = bicubic_maibo('rice.tif', Input_vector2)
end_time = time.perf_counter() 
elapsed_time = end_time - start_time  
print(f"Bicubic运行时间: {elapsed_time} 秒")

# 3. The result plotting
fig, axes = plt.subplots(1, 3)
axes[0].imshow(image_data, cmap='Greys')
axes[0].set_title('The original image')
axes[1].imshow(Output_image1, cmap='Greys')
axes[1].set_title('The enlarged image after bicubic interpolation')
axes[2].imshow(Output_image2, cmap='Greys')
axes[2].set_title('The shrunk image after bicubic interpolation')
plt.tight_layout()  
plt.show()

# 4. File saving
tifffile.imwrite('enlarged_bicubic_maibo.tif', Output_image1)
tifffile.imwrite('shrunk_bicubic_maibo.tif', Output_image2)