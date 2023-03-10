import uvicorn
from fastapi import FastAPI, File
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Создание объекта класса приложения FastAPI
app = FastAPI()

# Загрузка модели TensorFlow
model = load_model('my_model.h5')

# Названия классов
class_names = ['самолеты', 'автомобили', 'птицы', 'кошки', 'олени', 'собаки', 'лягушки', 'лошади', 'корабли',
               'грузовики']


# Функция для обработки изображения
def process_image(image):
    # Изменение размера изображения до 32x32
    resized = cv2.resize(image, (32, 32))
    # Нормализация значений пикселей
    normalized = resized / 255.0
    # Добавление измерения для батча
    batched = np.expand_dims(normalized, axis=0)
    return batched


# Функция для классификации изображения
def classify_image(image):
    # Обработка изображения
    processed_image = process_image(image)
    # Классификация изображения
    predictions = model.predict(processed_image)
    # Поиск индекса наибольшего значения предсказания
    predicted_index = np.argmax(predictions)
    # Возвращение названия класса
    return class_names[predicted_index]

# Создание маршрута API для классификации изображения
@app.post("/predict")
async def predict(file: bytes = File(...)):
    # Загрузка изображения
    image = cv2.imdecode(np.fromstring(file, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None: return "error"
    # Классификация изображения
    predicted_class = classify_image(image)
    # Возвращение результата
    return predicted_class

if __name__ == "__main__":
    # Запуск сервера
    uvicorn.run(app, host="0.0.0.0", port=8000)
