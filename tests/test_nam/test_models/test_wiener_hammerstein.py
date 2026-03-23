import pytest as _pytest

from nam.models import wiener_hammerstein as _wiener_hammerstein

from .base import Base as _Base


class TestWienerHammerstein(_Base):
    @classmethod
    def setup_class(cls):
        C = _wiener_hammerstein.WienerHammerstein
        args = ()
        kwargs = {
            "stages": [
                {
                    "filter_length": 4,
                    "hidden_sizes": [4],
                    "activation": "Tanh",
                    "context": 1,
                },
            ],
            "post_filter_length": 4,
        }
        super().setup_class(C, args, kwargs)

    @_pytest.mark.parametrize(
        "stages,post_filter_length",
        [
            # Single stage
            (
                [{"filter_length": 8, "hidden_sizes": [4, 4], "activation": "Tanh", "context": 1}],
                8,
            ),
            # Multi-stage
            (
                [
                    {"filter_length": 4, "hidden_sizes": [4], "activation": "Tanh", "context": 1},
                    {"filter_length": 4, "hidden_sizes": [4], "activation": "Tanh", "context": 1},
                ],
                4,
            ),
            # Context > 1
            (
                [{"filter_length": 4, "hidden_sizes": [4], "activation": "Tanh", "context": 3}],
                4,
            ),
        ],
    )
    def test_init(self, stages, post_filter_length):
        super().test_init(kwargs={"stages": stages, "post_filter_length": post_filter_length})

    @_pytest.mark.parametrize(
        "stages,post_filter_length",
        [
            (
                [{"filter_length": 4, "hidden_sizes": [4], "activation": "Tanh", "context": 1}],
                4,
            ),
            (
                [
                    {"filter_length": 4, "hidden_sizes": [4], "activation": "Tanh", "context": 1},
                    {"filter_length": 4, "hidden_sizes": [4], "activation": "Tanh", "context": 1},
                ],
                4,
            ),
        ],
    )
    def test_export(self, stages, post_filter_length):
        super().test_export(kwargs={"stages": stages, "post_filter_length": post_filter_length})


if __name__ == "__main__":
    _pytest.main()
