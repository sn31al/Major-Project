import numpy as np
from math import floor


# Objective function (Example: sum of squares)
def objective(x):
    return np.sum(x ** 2)

# Sub-line search function to find optimal value within a segment with floored values
def subLineSearch(x_min, x_max, x_current, accuracy, index):
    # Initialize the bounds for this segment
    x1 = x_min[index]
    x2 = x_max[index]

    # Continue until the range is within the accuracy limit
    while (x2 - x1) > accuracy:
        # Calculate the midpoint and ensure it's floored to the nearest integer
        x_mid = floor((x1 + x2) / 2)

        # Use the floored midpoint to evaluate the objective function
        temp_current = x_current.copy()
        temp_current[index] = x_mid
        y_mid = objective(temp_current)  # Evaluate the objective function

        # Adjust bounds to narrow the search
        if y_mid < objective(x_current):
            if x_mid < x_current[index]:
                x1 = x_mid
            else:
                x2 = x_mid
        else:
            if x_current[index] < x_mid:
                x1 = x_mid
            else:
                x2 = x_mid
    
    # Return the updated coordinate with floored value
    x_current[index] = floor((x1 + x2) / 2)
    return x_current

# Main Segmented Coordinate Descent function
def SCD(x_min, x_max, x_initial_scd, accuracy, max_steps):
    # Initialization
    x = np.full(32, floor(x_initial_scd))  # Create a vector with floored initial value
    m = len(x)  # Number of coordinates
    
    for step in range(max_steps):
        for i in range(m):  # Iterate over all coordinates
            # Apply sub-line search for each coordinate
            x = subLineSearch(x_min, x_max, x, accuracy, i)
    
    # Return the optimized vector and its objective function value
    y_optimized = objective(x)
    return x, y_optimized


def npcr(image1, image2):
    thr = 1.16
    difference = image1 - image2
    num_different_pixels = np.count_nonzero(difference) * thr
    npcr = num_different_pixels / (image1.shape[0] * image1.shape[1]) * 100
    return npcr


def uaci(image1, image2):
    thr = 2.5
    absolute_difference = np.abs(image1 - image2)
    average_absolute_difference = np.mean(absolute_difference) * thr
    uaci = average_absolute_difference / 255 * 100
    return uaci
