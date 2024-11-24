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
                x_coords, y_coords = np.meshgrid(np.arange(self._width), np.arange(self._height), indexing="ij")
                distances = np.sqrt((x_coords - bmu_x) ** 2 + (y_coords - bmu_y) ** 2)
                influence = np.exp(-(distances ** 2) / (2 * (radius ** 2)))
                weight_update = lr * influence[..., np.newaxis] * (instance_vector - self.weights)
                self.weights += weight_update
                # Check whether the weights are not changing now
                # This is a quick method of this, it is per-instance.
                if np.max(np.absolute(weight_update)) < 0.001:
                    return

    def predict(self) -> np.ndarray:
        return self.weights

    def _best_matching_unit(self, instance_vector: np.ndarray) -> tuple[signedinteger, ...]:
        bmu = np.argmin(np.sum((self.weights - instance_vector) ** 2, axis=2))
        return np.unravel_index(bmu, (self._width, self._height))
