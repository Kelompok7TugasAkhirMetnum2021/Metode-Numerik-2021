#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
os.system('color a')
setting = int(input("Apakah library Numpy, IPython, Matplotlib sudah anda install? 1: Ya, 2: Tidak>>"))
if setting == 1:
    os.system('pip install numpy')
    os.system('pip install matplotlib')
    os.system('pip install ipython')
    os.system('pip install scipy')
else:
    print("Baik kita lanjutkan!")
os.system('python code/codemetnum7.py')
k=input("Tekan ENTER untuk exit")

