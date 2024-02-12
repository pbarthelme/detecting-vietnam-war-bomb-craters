"""Utility functions

A collection of utility functions that are used in multiple notebooks.

This script requires the h5py and pyyaml packages to be installed within 
the Python environment you are running this script in.

This file contains the following functions:

    * create_dir - create a new directory
    * load_config - load a YAML configuration file
    * apply_min_max_scaling - apply min-max scaling to a numpy array
    * load_data - load data from a h5py file
    * rect_from_bound - return rectangle coordinates from a list of bounds
"""

import os

import h5py
import yaml


def create_dir(directory, is_file=False):
    """
    Create a directory at the specified path if it does not already exist.

    Parameters:
    - directory (str): The path of the directory to be created. If `is_file` is True,
      the directory part of the full path is considered.
    - is_file (bool, optional): If True, `directory` is treated as a file path,
      and the directory part of the path will be used. Default is False.

    Returns:
    None
    """
    if is_file:
        # Get the directory part of the full path
        directory = os.path.dirname(directory)

    # Check if the directory already exists
    if os.path.exists(directory):
        print(f"Directory already exists: {directory}")
        return

    # Create the directory and any missing parent directories
    try:
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    except OSError as e:
        print(f"Error creating directory: {directory}")
        print(e)


def load_config(path, study_area=None, raster_id=None):
    """
    Load a YAML configuration file with optional string substitution.

    Args:
        path (str): The path to the YAML configuration file.
        study_area (str, optional): The study area string to substitute.
        raster_id (str, optional): The raster ID string to substitute.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(path, encoding="utf-8") as file:
        content = file.read()

        if study_area is not None:
            content = content.replace("{study_area}", study_area)

        if raster_id is not None:
            content = content.replace("{raster_id}", raster_id)

        config = yaml.safe_load(content)

    return config


def apply_min_max_scaling(x, axis=(1, 2, 3)):
    """
    Apply min-max scaling to a numpy array along specified axes.

    Parameters:
        x (numpy.ndarray): The input numpy array.
        axis (int or tuple, optional): The axes along which to compute the min and max values.
            Default is (1, 2, 3).

    Returns:
        numpy.ndarray: The scaled numpy array with values between 0 and 1.
    """
    min_vals = x.min(axis=axis, keepdims=True)
    max_vals = x.max(axis=axis, keepdims=True)
    x_scaled = (x - min_vals) / (max_vals - min_vals)
    return x_scaled


def load_data(path, *subsets):
    """
    Load data from an HDF5 file at the specified path for the specified subsets.

    Parameters:
    - path (str): The path to the HDF5 file containing the data.
    - subsets (str): One or more strings representing the subsets of data to load.
      These should correspond to the keys in the HDF5 file.

    Returns:
    list: A list containing the loaded data arrays for each specified subset.

    """
    res = list()
    with h5py.File(path, "r") as hf:
        for subset in subsets:
            data = hf[f"{subset}"][:]
            res.append(data)
    return res


def rect_from_bound(xmin, xmax, ymin, ymax):
    """Returns list of (x,y)'s for a rectangle"""
    xs = [xmax, xmin, xmin, xmax, xmax]
    ys = [ymax, ymax, ymin, ymin, ymax]
    return [(x, y) for x, y in zip(xs, ys)]
