class Header:
    """Contains the formats and header information for the dataframes used in this project."""

    def benchmark_headers(self, hr_values: list[int]) -> list[str]:
        return [
            self.benchmark_format(val)
            for val in hr_values
        ]

    def lr_headers(self, lr_values: list[int]) -> list[str]:
        return [
            self.low_res_format(val)
            for val in lr_values
        ]

    def sr_headers(self, lr_values: list[int], sr_values: list[int]) -> list[str]:
        from .helpers import combos
        return [
            self.super_res_format(lr, sr)
            for lr, sr in combos(lr_values, sr_values)
        ]

    @staticmethod
    def benchmark_format(value) -> str:
        """Format for the benchmark headers"""
        return f'BM: {value}'

    @staticmethod
    def low_res_format(value) -> str:
        """Format for low_resolution headers"""
        return f'LR: {value}'

    @staticmethod
    def super_res_format(lr_val, sr_val) -> str:
        """Format for high_resolution headers"""
        return f'SR: {lr_val}-{sr_val}'
