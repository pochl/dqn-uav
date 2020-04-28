#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:48:27 2020

@author: thepoch
"""
from __future__ import print_function
import numpy as np
from base64 import b64decode
import cv2
import time

class transfer_data():
    """Class for transfering data to and from Unity"""
    def __init__(self, sock, InputType, InputDim, DeptEstSpeed, truestate):        
        self.sock = sock
        self.InputType = InputType
        self.InputDim = InputDim
        self.DeptEstSpeed = DeptEstSpeed
        self.truestate = truestate
    
    def ConvertDataReceived(self, data_received):
        """Convert string of data from Unity to processable list"""
        split_list = data_received.split(" ")
        if self.InputType == "Visual":
            return list(map(float, split_list[:-1])) + [split_list[-1]]
        elif self.InputType == "LiDAR":
            return list(map(float, split_list))
    
    def decode_image(self, base64_img):
        """Decode base64 from Unity to image"""
        a = b64decode(base64_img)
        np_arr = np.frombuffer(a, np.uint8)
        img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image = np.round(transfer_data.rgb2gray(img_np))
        image = cv2.resize(image,(self.InputDim[1],self.InputDim[0]), 
                           interpolation = cv2.INTER_AREA)
        
        """Time delay to simulate slow computational time 
        of depth estimation algorithm"""
        time.sleep(max(0,(self.DeptEstSpeed)))
        return image/255 #Standardise the pixel value into 0 - 1
    
    def rgb2gray(rgb):
        """Turn RGB type of image into gray-scale image"""
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]) 
      
    def ReceiveData(self, image_old):
        """Recieve raw data from Unity and process into state"""
        data_received = self.sock.recv(1024).decode("utf-8")    
        data = transfer_data.ConvertDataReceived(self, data_received)
        
        if self.InputType == "Visual":
            """
            If data contains image, try to decode it. 
            If error occurs, continue by using previous image and do not store
            this experience in the replay memory. The error does not occur
            often.
            """
            img_start = -1
            try:
                image = transfer_data.decode_image(self, data[-1])
                rem = True
            except:
                print('error')
                image = self.image_old
                rem = False 
        
        elif self.InputType == "LiDAR":
            img_start = len(data) - np.prod(self.InputDim)
            image = np.array(data[img_start:])
            image = image.reshape([self.InputDim[0],self.InputDim[1]])
            rem = True
        
        state = data[(img_start - self.truestate):img_start] + \
                list(image.flatten())   
        return data, state, image, rem
    
    def SendData(self, data):
        """
        Send action and command to whether to reset the environment to Unity
        """
        data_send = str(data)
        self.sock.sendall(data_send.encode("utf-8"))