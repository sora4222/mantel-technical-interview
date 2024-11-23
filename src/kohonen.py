import numpy as np


def train(input_data: np.ndarray, n_max_iterations: int, width: int, height: int):
    radius_initial = max(width, height) / 2
    lr_initial = 0.1

    # These are per node
    weights = np.random.random((width, height, 3))
    time_constant = n_max_iterations / np.log(radius_initial)

    for t in range(n_max_iterations):
        radius = radius_initial * np.exp(-t / time_constant)
        lr = lr_initial * np.exp(-t / time_constant)
        for instance_vector in input_data:
            # TODO: Check this is from the right axis
            bmu = np.argmin(np.sum((weights - instance_vector) ** 2, axis=2))
            # TODO: Need to check `unravel_index`
            bmu_x, bmu_y = np.unravel_index(bmu, (width, height))
            # TODO: Convert to vectorised calculations
            for x in range(width):
                for y in range(height):
                    distance = np.sqrt(((x - bmu_x) ** 2) + ((y - bmu_y) ** 2))
                    influence = np.exp(-(distance ** 2) / (2 * (radius ** 2)))
                    weights[x, y] += lr * influence * (instance_vector - weights[x, y])
        #TODO: Add the check for end of training
    return weights
