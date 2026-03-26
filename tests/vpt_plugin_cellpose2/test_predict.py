from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import pytest

from vpt_core.io.image import ImageSet

from tests.vpt_plugin_cellpose2 import TEST_DATA_ROOT
from vpt_plugin_cellpose2 import CellposeSegParameters, CellposeSegProperties
from vpt_plugin_cellpose2 import predict


@dataclass(frozen=True)
class Circle:
    x: int
    y: int
    radius: int


def generate_images(image_size: int, cells: List[Circle]) -> Tuple[ImageSet, str, str]:
    dapi = np.ones((image_size, image_size), dtype=np.uint8)
    cellbound3 = np.ones((image_size, image_size), dtype=np.uint8)
    cellbound1 = np.ones((image_size, image_size), dtype=np.uint8)
    for cell in cells:
        cv2.circle(dapi, (cell.x, cell.y), int(cell.radius * 0.8), (255, 255, 255), -1)
        cv2.circle(cellbound3, (cell.x, cell.y), int(cell.radius * 1.5), (255, 255, 255), -1)
        cv2.circle(cellbound1, (cell.x, cell.y), int(cell.radius * 1.4), (255, 255, 255), -1)

    red, green, blue = "Cellbound1", "Cellbound3", "DAPI"
    images = ImageSet()
    images[red] = {i: cellbound1 for i in range(7)}
    images[green] = {i: cellbound3 for i in range(7)}
    images[blue] = {i: dapi for i in range(7)}

    return images, blue, red


class FakeCellposeModel:
    init_kwargs = []
    eval_kwargs = []
    eval_shapes = []

    def __init__(self, gpu: bool = False, pretrained_model: str | None = None, model_type: str | None = None) -> None:
        FakeCellposeModel.init_kwargs.append(
            {
                "gpu": gpu,
                "pretrained_model": pretrained_model,
                "model_type": model_type,
            }
        )

    def eval(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        FakeCellposeModel.eval_kwargs.append(kwargs)
        FakeCellposeModel.eval_shapes.append(image.shape)
        mask = np.zeros(image.shape[:3], dtype=np.int32)
        height, width = mask.shape[1], mask.shape[2]
        labels = [
            (slice(10, min(30, height)), slice(10, min(30, width))),
            (slice(45, min(70, height)), slice(45, min(70, width))),
            (slice(90, min(120, height)), slice(20, min(50, width))),
            (slice(140, min(170, height)), slice(140, min(170, width))),
        ]
        for label_index, (row_slice, col_slice) in enumerate(labels, start=1):
            mask[:, row_slice, col_slice] = label_index
        return (mask.reshape(-1),)


@pytest.fixture(autouse=True)
def reset_fake_model() -> None:
    FakeCellposeModel.init_kwargs.clear()
    FakeCellposeModel.eval_kwargs.clear()
    FakeCellposeModel.eval_shapes.clear()


@pytest.mark.parametrize(
    "seg_props",
    [
        {
            "model_dimensions": "2D",
            "channel_map": {"red": "Cellbound1", "green": "Cellbound3", "blue": "DAPI"},
            "model": "cyto2",
            "custom_weights": None,
        },
        {
            "model_dimensions": "2D",
            "channel_map": {"red": "Cellbound1", "green": "Cellbound3", "blue": "DAPI"},
            "model": None,
            "custom_weights": str(TEST_DATA_ROOT / "CP_20230830_093420"),
        },
    ],
)
def test_run_prediction(monkeypatch: pytest.MonkeyPatch, seg_props) -> None:
    cells = [Circle(20, 15, 10), Circle(30, 100, 10), Circle(100, 20, 15), Circle(210, 100, 15)]
    images, nuc, fill = generate_images(513, cells)
    monkeypatch.setattr(predict.models, "CellposeModel", FakeCellposeModel)

    properties = CellposeSegProperties(
        seg_props["model_dimensions"],
        seg_props["channel_map"],
        seg_props["model"],
        seg_props["custom_weights"],
    )
    parameters = CellposeSegParameters(nuc, fill, 30, 0.95, 0.0, 0)
    mask = predict.run(images, properties, parameters)

    assert mask.shape == (7, 513, 513)
    for i in images.z_levels():
        labels = np.unique(mask[i, :, :])
        assert labels.tolist() == [0, 1, 2, 3, 4]

    expected_model = {
        "gpu": False,
        "pretrained_model": seg_props["custom_weights"],
        "model_type": None if seg_props["custom_weights"] else "cyto2",
    }
    assert FakeCellposeModel.init_kwargs == [expected_model]
    assert FakeCellposeModel.eval_shapes == [(7, 513, 513, 3)]
    assert FakeCellposeModel.eval_kwargs == [
        {
            "z_axis": 0,
            "channels": [1, 3],
            "channel_axis": 3,
            "diameter": 30,
            "flow_threshold": 0.95,
            "cellprob_threshold": 0.0,
            "resample": False,
            "min_size": 0,
            "do_3D": False,
        }
    ]


def test_run_prediction_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    cells = [Circle(20, 15, 10), Circle(30, 100, 10), Circle(100, 20, 15), Circle(210, 100, 15)]
    images, nuc, fill = generate_images(256, cells)
    monkeypatch.setattr(predict.models, "CellposeModel", FakeCellposeModel)
    properties = CellposeSegProperties(
        "2D",
        {"red": "Cellbound1", "green": "Cellbound3", "blue": "DAPI"},
        None,
        str(TEST_DATA_ROOT / "CP_20230830_093420"),
    )
    parameters = CellposeSegParameters(nuc, fill, 30, 0.95, 0.0, 0)
    mask = predict.run(images, properties, parameters)

    assert mask.shape == (7, 256, 256)
    for i in images.z_levels():
        labels = np.unique(mask[i, :, :])
        assert labels.tolist() == [0, 1, 2, 3, 4]

    assert FakeCellposeModel.eval_shapes == [(7, 256, 256, 3)]
