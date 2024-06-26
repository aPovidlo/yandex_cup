# Yandex Cup 2023 - ML: Nowcasting

==============================

## Описание задачи

Прогноз осадков с высокой временной и пространственной детализацией важен как в повседневной жизни, так для различных сфер бизнеса. Для эффективного решения этой задачи часто используются данные, полученные от метеорологических радиолокаторов, способных регистрировать наличия влаги в атмосфере на большой площади и с хорошим разрешением. Эти устройства позволяют определить интенсивность осадков и их характеристики. В рамках данной задачи вам предстоит на основе данных, собранных одним из таких метеорологических радаров, прогнозировать объем осадков, который выпадет в ближайшие несколько часов в конкретное время и в конкретном месте.

Данные и код бейзлайна доступны по ссылке: https://disk.yandex.ru/d/bLIfjz-8n5InYA

## Данные

В папке `train` лежат hdf5-файлы с измерениями одного из российских метеорологических радаров за 2021 год с разбивкой по месяцам. В файлах для каждого момента времени (шаг 10 минут) хранятся данные про интенсивность осадков, отражаемость облаков и их радиальную скорость, погодные явления.

Обученную модель надо будет проверить на данных из файла `2021-test-public.hdf5`, в котором хранятся часть измерений с метеорологического радара за 2022 год.

Пример с визуализацией данных находится в файле Jupyter Notebook `draw-samples.ipynb`.

Данные измерений метеорологического радара предоставлены Яндекс Погодой и ФГБУ ЦАО.

### Общая информация про формат данных

Радар делает измерения каждые 10 минут с пространственным разрешением 2 км и возвращает изображение с несколькими каналами. Таким образом, каждый пиксель изображения соответствует квадрату 2х2 км с данными различных измерений атмосферы по высоте.

Есть два спец. кода:

- `-2e6` — нет данных измерений в точке
- `-1e6` — нет каких-либо погодных явлений (например, нет осадков для поля `intensity`)

### Интенсивность осадков

Интенсивность осадков в мм/ч — величина, которую требуется прогнозировать. Рассчитывается по формуле Маршала-Палмера [1] из отражаемости, которая описана в следующем разделе. Показывает сколько выпало бы осадков в мм, если бы осадки шли с такой интенсивностью целый час.

### Отражаемость

Отражаемость — величина, измеряемая радаром. Значение скоррелировано с количеством влаги в атмосфере в некоторой точке. Измерения происходят на 10 уровнях высоты от 1км до 10 км с шагом 1 км и аппроксимируются на полный круг измерения радара.

### Радиальная скорость

Радиальная скорость — ещё одна величина, измеряемая радаром в м/с на10 уровнях высоты. По пространственной форме соответствует отражаемости. В отличие от отражаемости значения не апроксимируются на полный круг радара, а остаются как есть.

### Погодные явления

Расчётные значения погодных явлений на полном круге радара, которые могут принимать следующие значения:

0. Нет облачности
1. Обл. в ср. яр.
2. Слоист. обл.
3. Осадки слабые
4. Осадки умеренные
5. Осадки сильные
6. Кучевая облачность
7. Ливень слабый
8. Ливень умеренный
9. Ливень сильный
10. Гроза, вероятность 30-70%
11. Гроза, вероятность 70-90%
12. Гроза, вероятность > 90%
13. Град слабый
14. Град умеренный
15. Град сильный
16. Шквал слабый
17. Шквал умеренный
18. Шквал сильный
19. Смерч

## Код

В папке `sources` лежат три файла:

- `main.py` — файл для запуска обучения и теста с некоторой моделью
- `datasets.py` — класс для чтения данных радара, в пригодном для PyTorch формата
- `models.py` — две бейзлайновых модели
  - `PersistantModel`— модель, которая считает, что в ближайшие два часа осадки будут как в последние 10 минут
  - `ConvLSTMModel` — простая нейронная модель на базе ConvLSTM [2]

Пример запуска:

```
$ python ./main.py --model persistant
```

## Метрика

В задаче наукастинга осадков необходимо минимизировать отклонение спрогнозированных мм от истинного. За основу берётся метрика RMSE [3], но с небольшими изменениями. Финальная формула выглядит так:

$$
\frac{1}{T}\sum_{t=1}^{T}
\sqrt{
  \frac{1}{N}\sum_{n=1}^{N}
  \sum_{i=1}^{W}
  \sum_{j=1}^{H}
    (y_{t,n,i,j} - \hat{y}_{t,n,i,j})^{2} \cdot \mathbb{I}(y_{t,n,i,j} \neq -2e6)
}
$$

где

- `T` — количество спрогнозированных кадров;
- `N` — количество примеров в тесте;
- `W` — ширина кадра;
- `H` — высота кадра.

Функция `evaluate_on_val` в файле `main.py` содержит код для расчёта метрики.

## Ссылки

[1] https://ru.wikipedia.org/wiki/Радиолокационная_отражаемость 

[2] https://arxiv.org/abs/1506.04214 

[3] https://en.wikipedia.org/wiki/Root-mean-square_deviation 



Структура проекта
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
