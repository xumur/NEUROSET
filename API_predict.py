from requests import post
from tkinter import filedialog

# Выбор файла
file = filedialog.askopenfilename().replace("/", "\\")

# Открываем файл с изображением
with open(file, 'rb') as f:
    image_data = f.read()

# Отправляем POST-запрос на URL-адрес вашего API
response = post('http://localhost:8000/predict', files={'file': image_data})

# Получаем название предсказанного класса
print(response.text[1:-1])
