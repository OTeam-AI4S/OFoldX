from __future__ import annotations

import chanfig
import pytest

from ofoldx.model import Model
from ofoldx.models.registry import ModelRegistry
from ofoldx.processor import Processor


class Config(chanfig.Config):
    pass


class OtherConfig(chanfig.Config):
    pass


class ConfigSubclass(Config):
    pass


class RegisteredModel(Model):
    config_class = Config


class ConflictingModel(Model):
    config_class = Config


class OtherModel(Model):
    config_class = OtherConfig


class RegisteredProcessor(Processor):
    pass


class OtherProcessor(Processor):
    pass


class ProcessorSubclass(RegisteredProcessor):
    pass


class IncompatibleProcessor(Processor):
    def validate_model_input_layout(self, model: Model) -> None:
        raise ValueError("incompatible layout")


def _registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("model", model=RegisteredModel, processor=RegisteredProcessor)
    registry.register("model.variant", model=RegisteredModel, processor=OtherProcessor)
    registry.register("model.incompatible", model=RegisteredModel, processor=IncompatibleProcessor)
    return registry


class TestModelRegistry:
    def test_lookup(self) -> None:
        registration = _registry().lookup_model("model")

        assert registration.id == "model"
        assert registration.model is RegisteredModel
        assert registration.processor is RegisteredProcessor

    def test_config(self) -> None:
        assert _registry().model_for_config(Config()) is RegisteredModel

    def test_pair(self) -> None:
        registration = _registry().for_pair(RegisteredModel(Config()), OtherProcessor())

        assert registration.id == "model.variant"

    def test_layout(self) -> None:
        with pytest.raises(ValueError, match="incompatible layout"):
            _registry().for_pair(RegisteredModel(Config()), IncompatibleProcessor())

    def test_exact_id(self) -> None:
        with pytest.raises(KeyError):
            _registry().lookup_model("MODEL")

    def test_exact_config(self) -> None:
        with pytest.raises(KeyError):
            _registry().model_for_config(ConfigSubclass())

    def test_exact_pair(self) -> None:
        with pytest.raises(KeyError):
            _registry().for_pair(RegisteredModel(Config()), ProcessorSubclass())

    @pytest.mark.parametrize(
        "registration",
        [
            pytest.param(
                ("model", OtherModel, RegisteredProcessor),
                id="id",
            ),
            pytest.param(
                ("duplicate.pair", RegisteredModel, RegisteredProcessor),
                id="pair",
            ),
            pytest.param(
                ("conflicting.config", ConflictingModel, OtherProcessor),
                id="config",
            ),
            pytest.param(("", OtherModel, OtherProcessor), id="empty-id"),
        ],
    )
    def test_conflicts(self, registration) -> None:
        id, model, processor = registration

        with pytest.raises(ValueError):
            _registry().register(id, model=model, processor=processor)
