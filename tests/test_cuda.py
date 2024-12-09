import pytest

class TestCudaImports:
    def test_import_cuda(self):
        try:
            import cuda
        except ImportError:
            pytest.fail("Failed to import cuda")

    def test_import_cudf(self):
        try:
            import cudf
        except ImportError:
            pytest.fail("Failed to import cudf")
    def test_import_polars(self):
        try:
            import polars
        except ImportError:
            pytest.fail("Failed to import polars")


if __name__ == '__main__':
    pytest.main()
