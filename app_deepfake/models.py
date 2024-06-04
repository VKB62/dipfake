from django.db import models
from enum import Enum


class DetectionResult(Enum):
    """
    Перечисление, представляющее результаты обнаружения глубоких подделок (deepfakes).

    Атрибуты:
    - FAKE (str): Обозначает результат обнаружения как подделку.
    - NOT_FAKE (str): Обозначает результат обнаружения как не подделку.
    """

    FAKE: str = "FAKE"
    NOT_FAKE: str = "НЕ FAKE"


MODEL_OPTIONS_LIST: list[str] = [
    "Модель ResNet101",
    "Модель InceptionV3",
    "Модель InceptionResNetV2",
]


class Deepfake(models.Model):
    """
    Модель Django для хранения информации о deepfakes.

    Атрибуты:
    - MODEL_OPTIONS (list[tuple[str, str]]): Перечень доступных моделей для обнаружения deepfakes.
    - RESULT_OPTIONS (list[tuple[str, str]]): Перечень возможных результатов обнаружения.
    - id (models.AutoField): Автоматически увеличиваемый первичный ключ.
    - model (models.CharField): Поле для хранения выбранной модели обнаружения.
    - result (models.CharField): Поле для хранения результата обнаружения.
    - upload_at (models.DateTimeField): Время загрузки записи.
    - video (models.FileField): Файл загруженного видео.

    Методы:
    - __str__(self): Возвращает строковое представление объекта.
    """

    MODEL_OPTIONS: list[tuple[str, str]] = [
        (option, option) for option in MODEL_OPTIONS_LIST
    ]

    RESULT_OPTIONS: list[tuple[str, str]] = [
        (option.value, option.value) for option in DetectionResult
    ]

    id = models.AutoField(primary_key=True)
    model = models.CharField(
        max_length=30,
        choices=MODEL_OPTIONS,
        default=MODEL_OPTIONS_LIST[0],
        verbose_name="Выберите модель",
    )
    result = models.CharField(
        max_length=10,
        choices=RESULT_OPTIONS,
        default=DetectionResult.FAKE.value,
    )

    upload_at = models.DateTimeField(auto_now_add=True)
    video = models.FileField(verbose_name="Выберите видео")

    def __str__(self) -> str:
        return f"Deepfake {self.id} - {self.result}"

    class Meta:
        db_table: str = "deepfake"
