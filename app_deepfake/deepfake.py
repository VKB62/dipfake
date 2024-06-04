import cv2
import numpy as np
from keras.applications import (
    ResNet101,
    InceptionV3,
    InceptionResNetV2,
)
from keras.applications.resnet import preprocess_input as input_resnet
from keras.applications.inception_v3 import preprocess_input as input_inception_v3
from keras.applications.inception_resnet_v2 import (
    preprocess_input as input_inception_resnet_v2,
)
from time import time

from typing import Callable


def time_calculation(
    function: Callable[..., float]
) -> Callable[..., tuple[float, float]]:
    def wrapper(*args, **kwargs):
        start = time()
        result: float = function(*args, **kwargs)
        return result, time() - start

    wrapper.__name__ = function.__name__
    return wrapper


class DeepfakeDetector:
    def __init__(self, model: str) -> None:
        model, func, model_threshold = MODELS.get(model)
        self.model = model(
            weights="imagenet",
            include_top=False,
        )
        self.func = getattr(self, func.__name__)
        self.model_threshold = model_threshold

    def __call__(self, path, *args, **kwargs) -> tuple[bool, float, float]:
        arr: np.ndarray = self.__extract_frames(path)
        model_output, time_work = self.func(arr)
        scaled = self.scale_to_threshold(model_output, self.model_threshold)
        return scaled < 0.4, scaled, time_work

    @staticmethod
    def __extract_frames(
        video_path: str,
        frame_size: tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        frames: list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        return np.array(frames)

    @time_calculation
    def resnet_101(self, arr: np.ndarray) -> float:
        processed_frames_resnet: list = [
            input_resnet(cv2.resize(frame, (224, 224))) for frame in arr
        ]
        features: list = [
            self.model.predict(frame.reshape(1, 224, 224, 3), verbose=0)
            for frame in processed_frames_resnet
        ]
        features_flat = [feat.flatten() for feat in features]
        av = np.mean(features_flat, axis=0)
        print(f"resnet_101 {np.mean(av)}")
        return np.mean(av)

    @time_calculation
    def inception_resnet_v2(self, arr: np.ndarray) -> float:
        processed_frames_inception_resnet_v2: list = [
            input_inception_resnet_v2(cv2.resize(frame, (224, 224))) for frame in arr
        ]
        features: list = [
            self.model.predict(frame.reshape(1, 224, 224, 3), verbose=0)
            for frame in processed_frames_inception_resnet_v2
        ]
        features_flat = [feat.flatten() for feat in features]
        av = np.mean(features_flat, axis=0)
        print(f"inception_resnet_v2 {np.mean(av)}")
        return np.mean(av)

    @time_calculation
    def inception_v3(self, arr: np.ndarray) -> float:
        processed_frames_inception_v3: list = [
            input_inception_v3(cv2.resize(frame, (224, 224))) for frame in arr
        ]
        features: list = [
            self.model.predict(frame.reshape(1, 224, 224, 3), verbose=0)
            for frame in processed_frames_inception_v3
        ]
        features_flat = [feat.flatten() for feat in features]
        av = np.mean(features_flat, axis=0)
        print(f"inception_v3 {np.mean(av)}")
        return np.mean(av)

    @staticmethod
    def scale_to_threshold(
        model_output: float, model_threshold: float, common_threshold: float = 0.4
    ) -> float:
        scale_factor: float = common_threshold / model_threshold
        scaled_output: float = model_output * scale_factor
        return scaled_output


MODELS: dict[str, tuple] = {
    "Модель ResNet101": (ResNet101, DeepfakeDetector.resnet_101, 0.4),
    "Модель InceptionResNetV2": (
        InceptionResNetV2,
        DeepfakeDetector.inception_resnet_v2,
        0.55,
    ),
    "Модель InceptionV3": (InceptionV3, DeepfakeDetector.inception_v3, 0.5),
}
