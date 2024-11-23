import matplotlib.pyplot as plt
import numpy as np

def train(input_data, n_max_iterations, width, height):
    radius_initial = max(width, height) / 2
    lr_initial = 0.1

    # These are per node
    weights = np.random.random((width, height, 3))
    time_constant = n_max_iterations / np.log(radius_initial)


    for t in range(n_max_iterations):
        radius = radius_initial * np.exp(-t/time_constant)
        lr = lr_initial * np.exp(-t/time_constant)
        for instance_vector in input_data:
            bmu = np.argmin(np.sum((weights - instance_vector) ** 2, axis=2))
            bmu_x, bmu_y = np.unravel_index(bmu, (width, height))
            for x in range(width):
                for y in range(height):
                    distance = np.sqrt(((x - bmu_x) ** 2) + ((y - bmu_y) ** 2))
                    influence = np.exp(-(distance ** 2) / (2*(radius ** 2)))
                    weights[x, y] += lr * influence * (instance_vector - weights[x, y])
    return weights

if __name__ == '__main__':
    # Generate data
    input_data = np.random.random((10,3))
    image_data = train(input_data, 100, 10, 10)

    plt.imsave('100.png', image_data)

    # Generate data
    input_data = np.random.random((10,3))
    image_data = train(input_data, 1000, 100, 100)

    plt.imsave('1000.png', image_data)
