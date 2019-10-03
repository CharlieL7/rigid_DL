def pos(eta, xi, nodes):
    """
    position in a triangle as a function of eta and xi

    Parameters:
        eta : parametric coordinate, scalar
        xi : paramteric coordinate, scalar
        nodes : three nodes of triangle as columns in 3x3 ndarray
    Returns:
        x : output position (3,) ndarray
    """
    x = (1. - eta - xi) * nodes[:, 0] + eta * nodes[:, 1] + xi * nodes[:, 2]
    return x
