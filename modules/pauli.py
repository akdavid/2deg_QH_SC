import tinyarray

# Pauli matrices for electron-hole degrees of freedom
sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_1 = tinyarray.array([[0, 1], [1, 0]])
sigma_2 = tinyarray.array([[0, -1j], [1j, 0]])
sigma_3 = tinyarray.array([[1, 0], [0, -1]])
