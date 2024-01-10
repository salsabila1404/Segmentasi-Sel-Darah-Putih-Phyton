#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import cv2


# In[26]:


img = cv2.imread('4.jpg')
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)


# In[27]:


react = (50,25,171,290)
cv2.grabCut(img,mask,react,bgdModel,fgdModel,15,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
bilateral = cv2.bilateralFilter(mask2, 12, 125, 125)
img2 = img*bilateral[:,:,np.newaxis]


# In[28]:


cv2.imshow("Citra Asli", img)
cv2.imshow("Hasil Segmentasi", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




