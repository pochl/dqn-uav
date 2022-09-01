import re

import numpy as np
import torch
import yaml


def load_yaml(path: str) -> dict:
    """Loads yaml file into dict.

    Args:
        path (str): Path of the yaml file

    Returns:
        dict: Dictionary.

    """
    with open(path, encoding="UTF-8") as file:
        yaml_dict = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_dict


def write_yaml(data: dict, path: str):
    """Writes dict to yaml file.

    Args:
        data (dict): Dictionary.
        path (str): Path to the yaml file.
    """

    with open(path, "w", encoding="UTF-8") as file:
        yaml.dump(data, file)


def read_spec(path: str) -> dict:
    """Reads specification from Unity

    Args:
        path (str): Path to the spec.txt file.

    Returns:
        dict: Dictionary.
    """

    with open(path, encoding="UTF-8") as file:
        lines = [line.split(' ') for line in file.read().splitlines()]

    return {k: (int(v) if i > 0 else v) for i, (k, v) in enumerate(zip(lines[0], lines[1]))}


def get_moving_average(period: int, values: torch.Tensor) -> np.ndarray:
    """Calculates simple moving average from tensor.

    Args:
        period (int): Period to calculate moving average.
        values (torch.Tensor): Tensor.

    Returns:
        np.array: Array of moving average values.
    """

    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = (
            values.unfold(dimension=0, size=period, step=1)
            .mean(dim=1)
            .flatten(start_dim=0)
        )
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
        return moving_avg.numpy()

    moving_avg = torch.zeros(len(values))
    return moving_avg.numpy()


def extract_numeric(name: str) -> list:
    """Extracts numbers from string.

    Args:
        name (str): File name.

    Returns:
        list: List of splitted elements.
    """
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(name)
    parts[1::2] = map(int, parts[1::2])
    return parts


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """Turns RGB image into gray-scale image.

    Args:
        rgb (np.array): RGB image.

    Returns:
        np.array: Gray-scale image.
    """

    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
