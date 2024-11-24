import numpy as np
from numpy import signedinteger


class Kohonen:
    def __init__(self, input_data: np.ndarray, width: int, height: int):
        self._radius_initial = max(width, height) / 2
        if self._radius_initial == 0:
            raise Exception('Kohonen initial radius cannot be one')

        self._input_data = input_data
        self._lr_initial = 0.1
        self._width = width
        self._height = height

        # These are per node
        self.weights = np.random.random((width, height, 3))

    def train(self, n_max_iterations: int):
        time_constant = n_max_iterations / np.log(self._radius_initial)

        for t in range(n_max_iterations):
            radius = self._radius_initial * np.exp(-t / time_constant)
            lr = self._lr_initial * np.exp(-t / time_constant)
            for instance_vector in self._input_data:
                bmu_x, bmu_y = self._best_matching_unit(instance_vector)
                # TODO: Convert to vectorised calculations
                for x in range(self._width):
                    for y in range(self._height):
                        distance = np.sqrt(((x - bmu_x) ** 2) + ((y - bmu_y) ** 2))
                        influence = np.exp(-(distance ** 2) / (2 * (radius ** 2)))
                        self.weights[x, y] += lr * influence * (instance_vector - self.weights[x, y])
            # TODO: Add the check for end of training
        return self.weights

    def _best_matching_unit(self, instance_vector: np.ndarray) -> tuple[signedinteger, ...]:
        bmu = np.argmin(np.sum((self.weights - instance_vector) ** 2, axis=2))
        return np.unravel_index(bmu, (self._width, self._height))
