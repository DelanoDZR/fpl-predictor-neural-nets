class Result:
    filters: int
    kernel_size: int
    mse: float

    def __init__(self, fil, ker, mse):
        self.filters = fil
        self.kernel_size = ker
        self. mse = mse
