# NEUROSET
### Пример использования TensorFlow для обучения нейронной сети.

## Описание проекта
Данный проект представляет собой решение задачи распознавания изображений на примере датасета CIFAR-10. Для решения задачи была использована сверточная нейронная сеть, реализованная с помощью фреймворка TensorFlow.
## Установка и запуск проекта

### Клонируйте репозиторий:
    git clone https://github.com/xumur/NEUROSET.git

### Создайте виртуальное окружение и активируйте его:
    python -m venv venv
#### Для Windows: 
    venv\Scripts\activate
#### Для Linux/MacOS:
    source venv/bin/activate

### Установите зависимости:
    pip install -r requirements.txt

### Запустите проект:
!Выбирая нужный файл необходимо учитывать, что в пути файла не должно быть кириллицы, можно использовать только латинские символы и цифры
    
    python predict.py

###### Дообучение модели:
    python fit.py

## Примеры использования
Вы можете использовать этот проект в качестве примера для обучения своей нейронной сети на датасете CIFAR-10.

## Как работает модель
Модель, используемая в данном проекте, состоит из нескольких свёрточных слоев и слоев пулинга, а также полносвязных слоев для классификации. Обучение модели происходило на датасете CIFAR-10, который содержит 60000 изображений в 10 классах: самолеты, автомобили, птицы, кошки, олени, собаки, лягушки, лошади, корабли и грузовики.

## Лицензия
Данный проект лицензируется под лицензией MIT.

## Ссылки
### [Документация TensorFlow](https://www.tensorflow.org/)
### [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
## Контактная информация
Если у вас есть вопросы или предложения по улучшению проекта, свяжитесь со мной по электронной почте:
#### [nanaimo0555@mail.ru]()
