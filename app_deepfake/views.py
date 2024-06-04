import cv2
import os
import numpy as np
from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from .forms import DeepfakeForm
from .models import Deepfake, DetectionResult
from .deepfake import DeepfakeDetector


def main_page(request: HttpRequest) -> HttpResponse:
    """
    Обрабатывает GET и POST запросы на главной странице.

    При POST запросе, форма DeepfakeForm заполняется данными из запроса и файла. Если форма
    валидна, создается запись Deepfake, которая сохраняется в базе данных. Затем вызывается
    DeepfakeDetector для анализа загруженного видео. Результат анализа сохраняется и отображается
    на странице.

    При GET запросе, отображается пустая форма DeepfakeForm.

    Аргументы:
    - request (HttpRequest): HTTP запрос от пользователя.

    Возвращает:
    - HttpResponse: Ответ сервера, содержащий HTML страницы с формой и результатами обработки.

    Переменные:
    - message (dict): Словарь для хранения сообщений, которые будут отображаться на странице.
    - form (DeepfakeForm): Форма для загрузки данных о Deepfake.
    - deepfake (Deepfake): Объект модели Deepfake, представляющий загруженное видео и его анализ.
    - result (str): Результат обнаружения Deepfake, возвращаемый DeepfakeDetector.
    - scaled, time_work (Any): Дополнительные данные, возвращаемые DeepfakeDetector.
    """
    message: dict[str, bool | str | float] = {}
    if request.method == "POST":
        form: DeepfakeForm = DeepfakeForm(request.POST, request.FILES)
        if form.is_valid():
            deepfake: Deepfake = form.save(commit=False)
            deepfake.save()

            result: bool
            scaled: float
            time_work: float
            result, scaled, time_work = DeepfakeDetector(deepfake.model)(
                deepfake.video.path
            )
            text_lines: list[str] = ["GOOD" if result else "FAKE", f"{scaled}%"]
            font_color: tuple[int, int, int] = (0, 255, 0) if result else (0, 0, 255)
            result: str = (
                DetectionResult.NOT_FAKE.value if result else DetectionResult.FAKE.value
            )
            deepfake.result = result
            deepfake.save()
            message: dict[str, bool | float] = {"result": result, "scaled": scaled, "time_work": time_work}

            cap: cv2.VideoCapture = cv2.VideoCapture(deepfake.video.path)
            fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")
            width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_filename: str = f"{deepfake.video.name.split('.')[0]}_result.mp4"
            output_path: str = os.path.join(settings.MEDIA_ROOT, output_filename)
            out: cv2.VideoWriter = cv2.VideoWriter(
                output_path, fourcc, 20.0, (width, height)
            )
            font: int = cv2.FONT_HERSHEY_SIMPLEX
            font_scale: float = 1.5
            font_thickness: int = 3
            padding: int = 10
            frame_thickness: int = 3

            while True:
                ret: bool
                frame: np.ndarray
                ret, frame = cap.read()
                if not ret:
                    break

                y_offset: int = height // 2
                text_width_max: int = 0
                text_height_total: int = 0

                for line in text_lines:
                    (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
                    text_width_max = max(text_width_max, text_width)
                    text_height_total += text_height + baseline

                start_x: int = (width - text_width_max) // 2 - padding
                start_y: int = y_offset - text_height_total // 2 - padding
                end_x: int
                end_y: int
                end_x, end_y = start_x + text_width_max + padding * 2, start_y + text_height_total + padding * 2

                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), font_color, frame_thickness)

                y_text_start: int = start_y + padding
                for line in text_lines:
                    text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
                    x_text: int = start_x + (text_width_max - text_size[0]) // 2
                    cv2.putText(frame, line, (x_text, y_text_start + text_size[1]), font, font_scale, font_color,
                                font_thickness)
                    y_text_start += text_height + baseline

                out.write(frame)
            cap.release()
            out.release()
    else:
        form: DeepfakeForm = DeepfakeForm()

    return render(request, "main.html", {"form": form, "message": message})
