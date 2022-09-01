import socket
import time
from base64 import b64decode
from typing import List, Tuple, Union

import cv2
import numpy as np

from src.libs.utils import rgb2gray


class Communicator:
    """Communicator class, which is used for establishing the connection with Unity to send and receive the data.

    Attributes:
        input_type (str): The type of input data. Either 'Visual' or 'LiDAR'.
        input_dim (Tuple[int, int]): (vertical_dimension, horizontal_dimension).
        dept_est_speed (Union[int, float]): Amount of time delay in second to delay each loop. This is used to mimic the
            real-life setting where depth estimation algorithm takes a bit of time to compute. This parameter only has
            an effect when input_type is set to 'Visual'.
        _image_previous (np.array): Depth image from the previous step.
        _sock (socket.socket): Socket object.
    """

    def __init__(
        self,
        input_type: str,
        input_dim: Tuple[int, int],
        dept_est_speed: Union[int, float],
    ):
        self.input_type = input_type
        self.input_dim = input_dim
        self.dept_est_speed = dept_est_speed

        self._image_previous = np.ones(input_dim)
        self._sock = None

    def connect(self):
        """Establishes connection with Unity."""

        host, port = "127.0.0.1", 25001  # Must be identical to the ones in Unity code
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((host, port))

    def disconnect(self):
        """Terminates connection with Unity."""

        self._sock.close()

    def convert_received_data(self, data_received: str) -> List[Union[int, float, str]]:
        """Convert string of data from Unity to processable list.

        Args:
            data_received (str): Raw received data from Unity.

        Returns:
            List[Union[int, float, str]]: Converted data.

        Raises:
            ValueError: Invalid Input Type.
        """
        split_list = data_received.split(" ")
        if self.input_type == "Visual":
            return [
                float(elm)
                for i, elm in enumerate(split_list)
                if (i != len(split_list) - 1)
            ]

        if self.input_type == "LiDAR":
            return list(map(float, split_list))

        raise ValueError("Invalid Input Type.")

    def decode_image(self, base64_img: str) -> np.ndarray:
        """Decodes base64 from Unity to image.

        Args:
            base64_img (str): Base64 image string.

        Returns:
            np.array: Decoded image in gray scale, ranging from 0 to 1.
        """

        np_arr = np.frombuffer(b64decode(base64_img), np.uint8)
        img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image = np.round(rgb2gray(img_np))
        image = cv2.resize(
            image, (self.input_dim[1], self.input_dim[0]), interpolation=cv2.INTER_AREA
        )

        # Time delay to simulate slow computational time of depth estimation algorithm.
        time.sleep(max(0, self.dept_est_speed))
        return image / 255  # Standardise the pixel value into 0 - 1

    # pylint: disable=[W0702(bare-except)]
    def receive_data(self) -> List[Union[int, float]]:
        """Recieves raw data from Unity and process into state

        Returns:
            List[Union[int, float]]: State.

        Raises:
            ValueError: Invalid input type.
        """
        data_received = self._sock.recv(1024).decode("utf-8")
        data = Communicator.convert_received_data(self, data_received)

        if self.input_type == "Visual":

            # If data contains image, try to decode it. If error occurs, continue by using previous image.
            try:
                image = Communicator.decode_image(self, data[-1])
                self._image_previous = image
            except:
                image = self._image_previous

            return data[:-1] + list(image.flatten())

        if self.input_type == "LiDAR":
            return data

        raise ValueError("Invalid input type.")

    def send_data(self, data: List[int]):
        """Sends action and command to whether to reset the environment to Unity.

        Args:
            data (List[int]): [action, to_reset].
        """

        data_send = str(data)
        self._sock.sendall(data_send.encode("utf-8"))
