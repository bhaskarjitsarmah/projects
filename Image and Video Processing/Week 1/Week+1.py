
# coding: utf-8

# ## Reading images

# In[54]:

# loading the libraries

import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[55]:

# reading the images

img = cv2.imread('Lenna.png') # cv2 reads images in BGR mode
b, g, r = cv2.split(img) # extracting pixels in three different channels B, G and R
img2 = cv2.merge([r, g, b]) # merging the pixels in RGB mode


# In[56]:

# plotting the images in both mode

plt.subplot(121); plt.title("BGR Mode"); plt.imshow(img)
plt.subplot(122); plt.title("RGB Mode"); plt.imshow(img2)
plt.show()


# ## Reducing number of intensity level from 256 to 2 

# In[57]:

# reducing the pixel intensity by a factor of 8

n = 8 # desired number of intensity level

r_n = np.uint8(np.floor(r/n) * n) # np.uint8 is used to convert float to integer
b_n = np.uint8(np.floor(b/n) * n)
g_n = np.uint8(np.floor(g/n) * n)


# In[58]:

img3 = cv2.merge([r_n, b_n, g_n])
plt.imshow(img3)
plt.show()


# ## Spatial average of image pixels

# In[59]:

for n in range(0, 512):
    r_avg[n][n] = np.mean([r[n-1][n+1], r[n][n+1], r[n+1][n+1], 
                  r[n-1][n], r[n][n], r[n+1][n],
                  r[n-1][n-1], r[n][n-1], r[n+1][n-1]])


# In[ ]:

np.mean([1, 2, 3])


# In[ ]:

plt.subplot(131); plt.title("BGR Mode"); plt.imshow(img)
plt.subplot(132); plt.title("RGB Mode"); plt.imshow(img2)
plt.subplot(133); plt.title("RGB with low intensity"); plt.imshow(img3)
plt.show()


# In[ ]:



