from src.kohonen import Kohonen
import numpy as np


def test_bmu_calculated():
    input_data = np.array([
        [0.75, 0.75, 0.75],
        [0.25, 0.25, 0.25]
    ])
    som = Kohonen(input_data, 4, 4)
    som.weights = np.zeros_like(som.weights)
    som.weights[0][0] = np.array([0.75, 0.75, 0.75])
    som.weights[3][3] = np.array([0.25, 0.25, 0.25])
    bmu = som._best_matching_unit(input_data[0])
    assert bmu == (0, 0)

def test_single_cycle():
    input_data = np.array([
        [0.75, 0.75, 0.75],
    ])
    som = Kohonen(input_data, 4, 4)
    som.weights = np.zeros_like(som.weights)
    som.weights[0][0] = np.array([0.75,0.75,0.75])
    som.train(1)
    results = som.predict()
    assert np.all(results[0][0] ==[0.75,0.75,0.75])
