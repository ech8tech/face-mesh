import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import time

# Инициализация MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Индексы ключевых точек для глаз (MediaPipe Face Mesh)
# Стандартный метод EAR использует 6 точек на каждый глаз
# Левый глаз (индексы в MediaPipe Face Mesh)
LEFT_EYE_POINTS = [
    33,   # левый внешний уголок (p1)
    160,  # верхняя точка (внешняя) (p2)
    158,  # верхняя точка (средняя) (p3)
    153,  # нижняя точка (средняя) (p4)
    144,  # нижняя точка (внешняя) (p5)
    133,  # правый внешний уголок (p6)
]

# Правый глаз (индексы в MediaPipe Face Mesh)
RIGHT_EYE_POINTS = [
    362,  # левый внешний уголок (p1)
    385,  # верхняя точка (внешняя) (p2)
    387,  # верхняя точка (средняя) (p3)
    373,  # нижняя точка (средняя) (p4)
    380,  # нижняя точка (внешняя) (p5)
    263,  # правый внешний уголок (p6)
]


def calculate_eye_aspect_ratio(landmarks, eye_points):
    """
    Вычисляет соотношение сторон глаза (Eye Aspect Ratio - EAR)
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    где p1-p6 - это 6 ключевых точек глаза
    """
    try:
        # Получаем координаты точек глаза
        points = []
        for idx in eye_points:
            if idx < len(landmarks.landmark):
                point = landmarks.landmark[idx]
                points.append([point.x, point.y])
        
        if len(points) != 6:
            return None
        
        points = np.array(points)
        
        # Вычисляем расстояния для EAR
        # Стандартная формула EAR: (вертикальное_1 + вертикальное_2) / (2 * горизонтальное)
        # где:
        # - вертикальное_1 = расстояние между верхней и нижней точками (внешние)
        # - вертикальное_2 = расстояние между верхней и нижней точками (средние)
        # - горизонтальное = расстояние между уголками глаза
        
        # Вертикальные расстояния (верх-низ)
        # p2 (верхняя внешняя) и p5 (нижняя внешняя)
        vertical_1 = np.linalg.norm(points[1] - points[4])
        # p3 (верхняя средняя) и p4 (нижняя средняя)
        vertical_2 = np.linalg.norm(points[2] - points[3])
        
        # Горизонтальное расстояние (лево-право)
        # p1 (левый уголок) и p6 (правый уголок)
        horizontal = np.linalg.norm(points[0] - points[5])
        
        if horizontal == 0:
            return None
        
        # EAR формула
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    except Exception:
        return None


def is_eye_open(landmarks, threshold=0.25):
    """
    Определяет, открыты ли глаза
    threshold - пороговое значение EAR (обычно 0.2-0.3)
    При открытых глазах EAR выше, при закрытых - ниже
    """
    left_ear = calculate_eye_aspect_ratio(landmarks, LEFT_EYE_POINTS)
    right_ear = calculate_eye_aspect_ratio(landmarks, RIGHT_EYE_POINTS)
    
    if left_ear is None or right_ear is None:
        return None
    
    # Используем среднее значение для обоих глаз
    avg_ear = (left_ear + right_ear) / 2.0
    
    return avg_ear > threshold


def play_beep(frequency=800, duration=0.1, sample_rate=44100):
    """
    Воспроизводит звуковой сигнал (beep)
    frequency - частота в Гц (по умолчанию 800)
    duration - длительность в секундах (по умолчанию 0.1)
    """
    try:
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t)
        # Применяем плавное затухание для избежания щелчков
        fade_samples = int(sample_rate * 0.01)
        wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
        wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        sd.play(wave, sample_rate)
    except Exception as e:
        print(f"Ошибка воспроизведения звука: {e}")

def draw_eye_ear_points(frame, landmarks, eye_points, color=(0, 255, 255)):
    """
    Рисует точки EAR на глазу и соединяет их линиями для визуализации
    frame - кадр для отрисовки
    landmarks - landmarks лица от MediaPipe
    eye_points - список индексов точек глаза
    color - цвет для отрисовки (BGR формат)
    """
    h, w = frame.shape[:2]
    
    # Получаем координаты точек в пикселях
    pixel_points = []
    for idx in eye_points:
        if idx < len(landmarks.landmark):
            landmark = landmarks.landmark[idx]
            # Преобразуем нормализованные координаты в пиксели
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            pixel_points.append((x, y))
    
    if len(pixel_points) != 6:
        return
    
    # Рисуем точки
    point_radius = 4
    point_thickness = -1  # Залитые круги
    for point in pixel_points:
        cv2.circle(frame, point, point_radius, color, point_thickness)
        cv2.circle(frame, point, point_radius + 1, (255, 255, 255), 1)  # Белая обводка
    
    # Рисуем линии для вычисления EAR
    line_thickness = 2
    
    # Горизонтальная линия (p1 - p6): левый уголок - правый уголок
    cv2.line(frame, pixel_points[0], pixel_points[5], (255, 0, 0), line_thickness)  # Синяя
    
    # Вертикальная линия 1 (p2 - p5): верхняя внешняя - нижняя внешняя
    cv2.line(frame, pixel_points[1], pixel_points[4], (0, 255, 0), line_thickness)  # Зеленая
    
    # Вертикальная линия 2 (p3 - p4): верхняя средняя - нижняя средняя
    cv2.line(frame, pixel_points[2], pixel_points[3], (0, 255, 0), line_thickness)  # Зеленая
    
    # Дополнительно: рисуем контур глаза (соединяем все точки по порядку)
    contour_color = (255, 255, 0)  # Голубой
    for i in range(len(pixel_points)):
        next_i = (i + 1) % len(pixel_points)
        cv2.line(frame, pixel_points[i], pixel_points[next_i], contour_color, 1)


def print_on_screen(frame, label, value, y_offset, x_offset=10, spacing=10):
    """
    Выводит метку и значение на экран, автоматически вычисляя позицию для значения
    frame - кадр для отрисовки
    label - текст метки
    value - значение для вывода
    y_offset - вертикальная позиция
    x_offset - горизонтальная позиция начала (по умолчанию 10)
    spacing - отступ между меткой и значением (по умолчанию 10)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (0, 255, 0)
    
    # Формируем текст метки
    label_text = f'{label}:'
    
    # Вычисляем размер текста метки
    (label_width, label_height), baseline = cv2.getTextSize(
        label_text, font, font_scale, thickness
    )
    
    # Рисуем метку
    cv2.putText(
        frame, 
        label_text, 
        (x_offset, y_offset), 
        font,
        font_scale,
        color,
        thickness
    )
    
    # Вычисляем позицию для значения (справа от метки с отступом)
    value_x = x_offset + label_width + spacing
    
    # Рисуем значение
    cv2.putText(
        frame, 
        str(value), 
        (value_x, y_offset), 
        font,
        font_scale,
        color,
        thickness
    )

def main():
    # Открываем камеру
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return
    
    print("Камера запущена. Нажмите 'q' для выхода.")
    print("Определение состояния глаз...")
    
    # Параметры для фильтрации мигания
    CLOSED_FRAMES_THRESHOLD = 8  # Количество кадров подряд для подтверждения закрытых глаз
    OPEN_FRAMES_THRESHOLD = 3    # Количество кадров подряд для подтверждения открытых глаз
    
    closed_frames_count = 0  # Счетчик кадров с закрытыми глазами
    open_frames_count = 0    # Счетчик кадров с открытыми глазами
    previous_state = None
    last_sound_time = 0       # Время последнего звукового сигнала
    SOUND_COOLDOWN = 0.5      # Минимальный интервал между звуками (секунды)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: Не удалось получить кадр с камеры")
            break
        
        # Переворачиваем изображение горизонтально для зеркального эффекта
        frame = cv2.flip(frame, 1)
        
        # Конвертируем BGR в RGB для MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Обрабатываем кадр
        results = face_mesh.process(rgb_frame)

        # Определяем состояние глаз
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                print_on_screen(frame, 'x', face_landmarks.landmark[0].x, 60)
                print_on_screen(frame, 'y', face_landmarks.landmark[0].y, 90)
                print_on_screen(frame, 'z', face_landmarks.landmark[0].z, 120)  

                # Визуализируем точки EAR на глазах
                draw_eye_ear_points(frame, face_landmarks, LEFT_EYE_POINTS, color=(0, 255, 255))  # Голубой для левого глаза
                draw_eye_ear_points(frame, face_landmarks, RIGHT_EYE_POINTS, color=(255, 0, 255))  # Пурпурный для правого глаза

                # Определяем, открыты ли глаза
                eyes_open = is_eye_open(face_landmarks, threshold=0.25)
                
                if eyes_open is not None:
                    # Обновляем счетчики кадров
                    if eyes_open:
                        open_frames_count += 1
                        closed_frames_count = 0
                    else:
                        closed_frames_count += 1
                        open_frames_count = 0

                    print_on_screen(frame, 'open_frames_count', open_frames_count, 150)
                    print_on_screen(frame, 'closed_frames_count', closed_frames_count, 180)
                    
                    # Определяем финальное состояние с учетом фильтрации мигания
                    if closed_frames_count >= CLOSED_FRAMES_THRESHOLD:
                        # Глаза действительно закрыты (не мигание)
                        state = "Eyes closed"
                        color = (0, 0, 255)  # Красный
                        
                        # Воспроизводим звук (с ограничением частоты)
                        current_time = time.time()
                        if current_time - last_sound_time >= SOUND_COOLDOWN:
                            play_beep(frequency=600, duration=0.15)
                            last_sound_time = current_time
                            
                    elif open_frames_count >= OPEN_FRAMES_THRESHOLD:
                        # Глаза открыты
                        state = "Eyes opened"
                        color = (0, 255, 0)  # Зеленый
                    else:
                        # Переходное состояние (мигание) - сохраняем предыдущее состояние
                        if previous_state:
                            state = previous_state
                            color = (0, 255, 0) if state == "Eyes opened" else (0, 0, 255)
                        else:
                            state = "Eyes opened"  # По умолчанию считаем открытыми
                            color = (0, 255, 0)
                    
                    # Выводим только при изменении состояния
                    if state != previous_state:
                        print(state)
                        previous_state = state
                    
                    # Рисуем результат на кадре
                    cv2.putText(
                        frame,
                        state,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2
                    )
                else:
                    # Сбрасываем счетчики при неопределенном состоянии
                    closed_frames_count = 0
                    open_frames_count = 0
                    if previous_state is not None:
                        print("Face detected but eyes state unclear")
                        previous_state = None
        else:
            # Сбрасываем счетчики при отсутствии лица
            closed_frames_count = 0
            open_frames_count = 0
            if previous_state is not None:
                print("No face detected")
                previous_state = None
        
        # Показываем кадр
        cv2.imshow('Eyes Checker', frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()
    print("Приложение завершено.")


if __name__ == "__main__":
    main()

