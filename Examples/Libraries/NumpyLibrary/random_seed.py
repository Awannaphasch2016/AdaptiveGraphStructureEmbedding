import numpy as np
def xx():
    # np.random.seed(999)
    return np.random.choice([1,2,3,4,5])

for i in range(10):
    np.random.seed(999)
    x = np.random.choice([1, 2, 3, 4, 5])
    print(x)
    print(xx())


