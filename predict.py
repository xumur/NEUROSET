import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import tensorflow as tf
import numpy as np
import cv2

# Загрузка модели TensorFlow
model = tf.keras.models.load_model('my_model.h5')

# Названия классов
class_names = ['самолеты', 'автомобили', 'птицы', 'кошки', 'олени', 'собаки', 'лягушки', 'лошади', 'корабли', 'грузовики']


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


# Функция для выбора файла
def choose_file():
    file_path = filedialog.askopenfilename().replace("/", "\\")
    print(file_path)
    if file_path:
        # Загрузка изображения
        image = cv2.imread(file_path)
        if image is None: print('Ошибка загрузки изображения')
        # Классификация изображения
        predicted_class = classify_image(image)
        # Отображение результата
        messagebox.showinfo('Результат классификации', f'Изображение относится к классу "{predicted_class}".')


if __name__ == "__main__":
    # Создание графического интерфейса
    root = tk.Tk()
    root.title('Классификация изображений')
    root.geometry('300x100')

    # Создание кнопки для выбора файла
    button = tk.Button(root, text='Выбрать файл', command=choose_file)
    button.pack(pady=10)

    # Запуск главного цикла
    root.mainloop()
