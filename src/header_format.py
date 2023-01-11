def benchmark(value) -> str:
    """Format for the benchmark headers"""
    return f'BM: {value}'


def low_res(value) -> str:
    """Format for low_resolution headers"""
    return f'LR: {value}'


def super_res(lr_val, sr_val) -> str:
    """Format for high_resolution headers"""
    return f'SR: {lr_val}-{sr_val}'
