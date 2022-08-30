#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:48:27 2020

@author: thepoch
"""
from __future__ import print_function

import time
from base64 import b64decode
import socket

import cv2
import numpy as np


class Communicator:
    """Class for transferring data to and from Unity"""

    def __init__(self, InputType, InputDim, DeptEstSpeed):
        self.InputType = InputType
        self.InputDim = InputDim
        self.DeptEstSpeed = DeptEstSpeed
        self.image_old = np.ones(InputDim)

        self._sock = None

    def connect(self):
        """Establishes connection with Unity"""

        host, port = "127.0.0.1", 25001  # Must be identical to the ones in Unity code
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((host, port))

    def disconnect(self):
        """Terminates connection with Unity"""

        self._sock.close()

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
        image = np.round(Communicator.rgb2gray(img_np))
        image = cv2.resize(
            image, (self.InputDim[1], self.InputDim[0]), interpolation=cv2.INTER_AREA
        )

        """Time delay to simulate slow computational time 
        of depth estimation algorithm"""
        time.sleep(max(0, (self.DeptEstSpeed)))
        return image / 255  # Standardise the pixel value into 0 - 1

    def rgb2gray(rgb):
        """Turn RGB type of image into gray-scale image"""
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def ReceiveData(self):
        """Recieve raw data from Unity and process into state"""
        data_received = self._sock.recv(1024).decode("utf-8")
        data = Communicator.ConvertDataReceived(self, data_received)

        if self.InputType == "Visual":
            """
            If data contains image, try to decode it.
            If error occurs, continue by using previous image and do not store
            this experience in the replay memory. The error does not occur
            often.
            """
            try:
                image = Communicator.decode_image(self, data[-1])
                self.image_old = image
                rem = True
            except:
                print("error")
                image = self.image_old
                rem = False

            data = data[:-1] + list(image.flatten())

        elif self.InputType == "LiDAR":
            rem = True

        else:
            raise ValueError("Invalid input type.")

        return data, rem

    def SendData(self, data):
        """
        Send action and command to whether to reset the environment to Unity
        """
        data_send = str(data)
        self._sock.sendall(data_send.encode("utf-8"))
