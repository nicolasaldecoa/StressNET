from pathlib import Path

import pytest

import stressnet.models as models
import stressnet.net.stressnet as stressnet_net
from stressnet.models._registry import get_entry, list_models


def test_registry_contains_notebook_models() -> None:
    names = set(list_models())
    assert 'STRESSNET-PRETRAINED-A' in names
    assert 'STRESSNET-PRETRAINED-B' in names
    assert 'STRESSNET-PRETRAINED-C' in names
    assert 'STRESSNET-PRETRAINED-D' in names
    assert 'STRESSNET-PRETRAINED-E' in names
    assert 'STRESSNET-FINETUNED-MYOSIN-88-A' in names


def test_get_entry_is_case_insensitive() -> None:
    entry = get_entry('stressnet-pretrained-a')
    assert entry.filename == 'StressNET_weights_pretrained_A.h5'
    assert entry.points_per_edge == 9


def test_get_entry_unknown_lists_available_models() -> None:
    with pytest.raises(KeyError) as exc_info:
        get_entry('missing-model')

    message = str(exc_info.value)
    assert 'Unknown model' in message
    assert 'STRESSNET-PRETRAINED-A' in message


def test_load_model_uses_registry_and_builds_with_downloaded_weights(monkeypatch, tmp_path) -> None:
    weights_path = tmp_path / 'weights.h5'
    calls = {}
    sentinel_model = object()

    def fake_download_model(name: str, *, force: bool = False) -> Path:
        calls['download'] = {'name': name, 'force': force}
        return weights_path

    def fake_get_model(**kwargs):
        calls['get_model'] = kwargs
        return sentinel_model

    monkeypatch.setattr(models, 'download_model', fake_download_model)
    monkeypatch.setattr(stressnet_net, 'get_model', fake_get_model)

    model = models.load_model(force_download=True, device='/CPU:0', custom_kwarg='value')

    assert model is sentinel_model
    assert calls['download'] == {'name': 'STRESSNET-PRETRAINED-A', 'force': True}
    assert calls['get_model'] == {
        'weights_path': str(weights_path),
        'points_per_edge': 9,
        'device': '/CPU:0',
        'custom_kwarg': 'value',
    }


def test_build_stressnet_rejects_unknown_fine_tune_layer(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(stressnet_net.Model, 'load_weights', lambda self, path: None)
    weights_path = tmp_path / 'weights.h5'

    with pytest.raises(ValueError, match='Values in fine_tune_layers') as exc_info:
        stressnet_net.build_stressnet(
            load_weights_path=str(weights_path),
            fine_tune_layers=['reg_head_fc_2', 'reg_head_ln_2reg_head_out'],
        )

    message = str(exc_info.value)
    assert 'reg_head_ln_2reg_head_out' in message
    assert 'reg_head_ln_2' in message
    assert 'reg_head_out' in message


def test_build_stressnet_freezes_all_layers_except_fine_tune_layers(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(stressnet_net.Model, 'load_weights', lambda self, path: None)
    weights_path = tmp_path / 'weights.h5'
    fine_tune_layers = {'reg_head_fc_2', 'reg_head_ln_2', 'reg_head_out'}

    model = stressnet_net.build_stressnet(
        load_weights_path=str(weights_path),
        fine_tune_layers=fine_tune_layers,
    )

    for layer in model.layers:
        assert layer.trainable is (layer.name in fine_tune_layers)
