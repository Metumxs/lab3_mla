# --- Розділ 1: Імпорти ---
# Імпортуємо бібліотеки, які ми встановили
import pandas as pd  # Для роботи з даними (таблицями/часовими рядами)
import matplotlib.pyplot as plt  # Для графіків
import numpy as np  # Для математичних операцій (потрібен для .reshape та create_sequences)
from sklearn.preprocessing import MinMaxScaler  # Для нормалізації

print("--- Скрипт запущено. Початок Частини 1 ---")

# --- Розділ 2: Завантаження та підготовка даних ---
# (Відповідає п. 1.1 "Оберіть часовий ряд")
file_path = 'weatherHistory.csv'  # Назва вашого файлу
try:
    # 1. Просто завантажуємо CSV
    data = pd.read_csv(file_path)

    # 2. Конвертуємо 'Formatted Date' в datetime.
    data['Formatted Date'] = pd.to_datetime(data['Formatted Date'], utc=True)

    # 3. Встановлюємо дату як індекс (це зручно для resample)
    data.set_index('Formatted Date', inplace=True)

    # 4. Обираємо наш цільовий стовпець
    ts_hourly = data['Temperature (C)']
except Exception as e:
    print(f"Помилка при завантаженні або обробці файлу: {e}")
    exit()

# 5. ВАЖЛИВО: Дані погодинні. Перетворимо їх на ЩОДЕННІ МАКСИМАЛЬНІ.
# .resample('D') - групує дані по Днях (Daily)
# .max() - беремо максимальне значення за кожен день
ts_daily_max = ts_hourly.resample('D').max()

# 6. Видаляємо пропуски (NaN), якщо вони з'явились (напр. день без даних)
ts_daily_max = ts_daily_max.dropna()

print(f"Дані завантажено та перетворено на щоденні (всього {len(ts_daily_max)} записів).\n")
# Збережемо "сирі" значення для п. 1.5
raw_data_values = ts_daily_max.values

# --- Розділ 3: Статистичний аналіз (Завдання 1.1) ---
print("--- Завдання 1.1: Статистичні показники ряду ---")

# .describe() рахує більшість потрібних нам показників
stats = ts_daily_max.describe()
print(stats)

# Додатково рахуємо асиметрію та куртозис
skewness = ts_daily_max.skew()
print(f"Асиметрія (Skewness): {skewness:.3f}")  #

kurtosis = ts_daily_max.kurt()
print(f"Куртозис (Kurtosis): {kurtosis:.3f}")  #
print("-" * 30 + "\n")

# --- Розділ 4: Візуалізація даних (Завдання 1.2) ---
print("--- Завдання 1.2: Побудова графіків ---")

# Графік 1: Динаміка змін в часі [cite: 16]
plt.figure(figsize=(14, 7))  # Встановлюємо розмір
ts_daily_max.plot()  # Будуємо лінійний графік
plt.title('Графік 1: Динаміка щоденної макс. температури (2006-2016)')
plt.xlabel('Дата')
plt.ylabel('Макс. Температура (°C)')
plt.grid(True)
plt.savefig('grafik_1_timeseries.png')  # Зберігаємо у файл
print("Збережено 'grafik_1_timeseries.png'")

# Графік 2: Гістограма розподілу
plt.figure(figsize=(10, 6))
ts_daily_max.hist(bins=50)  # bins=50 - кількість стовпчиків
plt.title('Графік 2: Гістограма розподілу щоденних макс. температур')
plt.xlabel('Макс. Температура (°C)')
plt.ylabel('Частота (кількість днів)')
plt.savefig('grafik_2_histogram.png')  # Зберігаємо у файл
print("Збережено 'grafik_2_histogram.png'")
print("-" * 30 + "\n")
# plt.show() # Можете розкоментувати цей рядок, якщо хочете, щоб графіки
#            # показувались на екрані одразу після запуску скрипта


# --- Розділ 5: Нормалізація (Завдання 1.3) ---
print("--- Завдання 1.3: Нормалізація даних ---")
# Обґрунтування:
# 1. Аналіз статистики (Розділ 3) показав велике стандартне відхилення (std).
# 2. Аналіз графіків (Розділ 4) показав не-нормальний, "двогорбий" розподіл.
# 3. Висновок: Використовуємо MinMaxScaler для приведення даних до діапазону [0, 1].

# 1. Scaler очікує 2D-масив. Перетворюємо [4019] -> [4019, 1]
# (raw_data_values ми отримали в Розділі 2)
raw_data_2d = raw_data_values.reshape(-1, 1)

# 2. Створюємо та "навчаємо" (fit_transform) scaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized_2d = scaler.fit_transform(raw_data_2d)

# 3. Повертаємо у зручний 1D-формат [4019] для функції "вікна"
data_normalized_1d = data_normalized_2d.flatten()

print(f"Дані нормалізовано. "
      f"Мін: {data_normalized_1d.min()}, Макс: {data_normalized_1d.max()}")
print("-" * 30 + "\n")

# --- Розділ 6: Функція "Ковзного вікна" (Завдання 1.4) ---
print("--- Завдання 1.4: Створення функції 'ковзного вікна' ---")


def create_sequences(data, n_steps):
    """
    Перетворює 1D-масив часового ряду у 2D-масив (X)
    послідовностей та 1D-масив (y) прогнозів.

    """
    X, y = [], []
    # Ідемо по масиву, зупиняючись за n_steps до кінця
    for i in range(len(data) - n_steps):
        # i - початок послідовності
        # end_ix - кінець послідовності (не включно)
        end_ix = i + n_steps

        # seq_x - це наші вхідні дані (n_steps штук) [34, 36, 35, 34, 31]
        # seq_y - це наш прогноз (наступне значення) [28]
        seq_x, seq_y = data[i:end_ix], data[end_ix]

        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


print("Функцію create_sequences() створено.\n")

# --- Розділ 7: Формування наборів даних (Завдання 1.5) ---
print("--- Завдання 1.5: Формування 5 наборів даних ---")

# Обираємо 5 розмірів вікна (n) згідно з завданням
n_steps_list = [7, 14, 30, 60, 90]  # напр: тиждень, 2 тижні, місяць, 2 міс, 3 міс

# Створюємо "контейнери" (словники Python) для зберігання наших 10-ти наборів
# (5 нормалізованих + 5 ненормалізованих)
normalized_datasets = {}
raw_datasets = {}

for n in n_steps_list:
    print(f"Створюємо набори для n = {n}...")

    # 1. Для нормалізованих даних (data_normalized_1d з Розділу 5)
    X_norm, y_norm = create_sequences(data_normalized_1d, n)
    normalized_datasets[n] = (X_norm, y_norm)

    # 2. Для ненормалізованих ("сирих") даних (raw_data_values з Розділу 2)
    X_raw, y_raw = create_sequences(raw_data_values, n)
    raw_datasets[n] = (X_raw, y_raw)

print("\n--- Перевірка створених наборів ---")
# Давайте подивимось, що вийшло, на прикладі n=30
n_example = 30
X_check, y_check = normalized_datasets[n_example]

print(f"Для n={n_example} (нормалізовані):")
print(f"  Форма (shape) X: {X_check.shape}")  # Має бути (кількість, 30)
print(f"  Форма (shape) y: {y_check.shape}")  # Має бути (кількість,)
print(f"\n  Перша послідовність (X[0]): \n{X_check[0]}")
print(f"  Перший прогноз (y[0]): \n{y_check[0]}")

# Перевірка, що все логічно
# y[0] має бути тим же значенням, що й data_normalized_1d[30]
print(f"\n  Перевірочне значення (має бути = y[0]): \n{data_normalized_1d[n_example]}")

print("\n--- ✅ ЧАСТИНА 1 ПОВНІСТЮ ЗАВЕРШЕНА ---")
print("Усі дані підготовлено. Можна переходити до Частини 2.")

# ######################################################################
# ######################################################################
# ПОЧАТОК КОДУ ДЛЯ ЧАСТИНИ 2
# (Додайте це в кінець вашого файлу main.py)
# ######################################################################
# ######################################################################

print("\n\n--- ✅ ПОЧАТОК ЧАСТИНИ 2: МОДЕЛЮВАННЯ ---")

# --- Розділ 8: Додаткові імпорти для Частини 2 ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import math  # Для розрахунку RMSE (корінь з MSE)
import time  # Для вимірювання часу навчання

# (Ми вже імпортували numpy as np, pandas as pd та MinMaxScaler раніше)


# --- Розділ 9: Розбиття даних на вибірки (Завдання 2.1) ---
print("--- Завдання 2.1: Підготовка до розбиття даних ---")

# Для часових рядів КАТЕГОРИЧНО НЕ МОЖНА перемішувати дані (shuffle=True).
# Ми повинні розбити їх послідовно.
# Обираємо пропорції: 70% - навчання, 15% - валідація, 15% - тест.
train_percent = 0.7
val_percent = 0.15
# test_percent = 0.15 (вийде автоматично)

print(f"Пропорції розбиття: {train_percent * 100}% train, {val_percent * 100}% validation, 15% test")


def split_sequences(X, y, train_pct, val_pct):
    """
    Розбиває X та y послідовно, без перемішування.
    """
    # Загальний розмір
    total_samples = len(X)

    # Визначаємо індекси для розбиття
    train_idx = int(total_samples * train_pct)
    val_idx = int(total_samples * (train_pct + val_pct))

    # Розбиваємо дані
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    return X_train, y_train, X_val, y_val, X_test, y_test


print("Функцію split_sequences() створено.\n")

# --- Розділ 10: Архітектура моделі LSTM (Завдання 2.2) ---
print("--- Завдання 2.2: Створення функції для побудови моделі ---")


def build_model(n_steps_in, n_features_in=1):
    """
    Створює та компілює модель LSTM "Багато-до-Одного".
    """
    model = Sequential()

    # n_steps_in - це наш розмір вікна n (напр., 30)
    # n_features_in - це 1, оскільки у нас лише 1 показник (температура)
    input_shape = (n_steps_in, n_features_in)

    # Додаємо шар LSTM. 50 - це кількість нейронів (популярне значення)
    # "return_sequences=False" (за замовчуванням) означає,
    # що шар видасть лише останній вихід, що нам і потрібно.
    model.add(LSTM(50, activation='relu', input_shape=input_shape))

    # Додаємо вихідний шар. 1 - тому що ми прогнозуємо 1 значення.
    model.add(Dense(1))

    # Компілюємо модель. 'adam' - стандартний оптимізатор.
    # 'mean_squared_error' (MSE) - найкраща функція втрат для регресії.
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


print("Функцію build_model() створено.\n")

# --- Розділ 11: Цикл експериментів (Завдання 2.3) ---
print("--- Завдання 2.3: Запуск циклу експериментів ---")

# Тут ми будемо зберігати результати
results_list = []

# Використовуємо ті ж 'n', що й в Частині 1
# n_steps_list = [7, 14, 30, 60, 90] (вже визначено в Частині 1)

# ВАЖЛИВО: Оскільки навчання нейромережі займає час,
# для тестування можна взяти менший список:
# n_steps_list_test = [7, 30]
# ...або зменшити кількість епох
EPOCHS = 20  # Кількість епох навчання. 20 - для швидкого тесту.
# Для гарних результатів можна поставити 50-100.
BATCH_SIZE = 32  # Розмір "пачки" даних за один крок навчання.

# Проходимо по двох словниках, які ми створили в Частині 1
# (raw_datasets, normalized_datasets)
all_experiments = {
    "Normalized": normalized_datasets,
    "Raw (Unnormalized)": raw_datasets
}

for data_type, datasets in all_experiments.items():
    print(f"\n--- ОБРОБКА ТИПУ ДАНИХ: {data_type} ---")

    # Проходимо по 5 розмірах вікна (n)
    for n_steps in n_steps_list:
        start_time = time.time()
        print(f"  > Початок експерименту: n = {n_steps}, Тип: {data_type}")

        # 1. Отримуємо дані
        X, y = datasets[n_steps]

        # 2. Розбиваємо дані
        X_train, y_train, X_val, y_val, X_test, y_test = \
            split_sequences(X, y, train_percent, val_percent)

        # 3. ВАЖЛИВИЙ КРОК: Зміна форми X для LSTM
        # LSTM очікує 3D-вхід: (кількість_прикладів, кроки_часу, кількість_ознак)
        # Наші X_train зараз (N, n_steps). Треба (N, n_steps, 1)
        # 1 - це 1 ознака (тільки температура)
        n_features = 1
        X_train_3d = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
        X_val_3d = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))
        X_test_3d = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

        # 4. Будуємо модель
        model = build_model(n_steps_in=n_steps, n_features_in=n_features)

        # 5. Навчаємо модель
        print(f"    ...Навчання ({EPOCHS} епох)...")
        history = model.fit(
            X_train_3d,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val_3d, y_val),
            verbose=0  # 0=тихий режим, 1=показувати прогрес
        )
        print("    ...Навчання завершено.")

        # 6. Оцінка на тестових даних
        # .predict() повертає прогнози
        y_pred = model.predict(X_test_3d)

        # 7. Розрахунок помилки (RMSE)
        # Помилку треба рахувати в ОДНАКОВОМУ МАСШТАБІ (градусах Цельсія)

        if data_type == "Normalized":
            # 7a. Якщо дані нормалізовані, ми ПОВЕРТАЄМО їх до
            # початкового масштабу перед розрахунком помилки
            # .inverse_transform() очікує 2D-масив
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_inv = scaler.inverse_transform(y_pred)

            # Рахуємо RMSE на "справжніх", де-нормалізованих даних
            rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

        else:  # data_type == "Raw (Unnormalized)"
            # 7b. Якщо дані "сирі", просто рахуємо RMSE
            rmse = math.sqrt(mean_squared_error(y_test, y_pred))

        end_time = time.time()
        duration = end_time - start_time

        # 8. Зберігаємо результати
        results_list.append({
            "Data Type": data_type,
            "N (Window Size)": n_steps,
            "RMSE (Test)": rmse,
            "Time (sec)": duration,
            "Epochs": EPOCHS
        })
        print(f"    > Готово! RMSE: {rmse:.3f} °C (Час: {duration:.1f} сек)")

# --- Розділ 12: Таблиця результатів та аналіз (Завдання 2.4) ---
print("\n\n--- ✅ ЧАСТИНА 2 ЗАВЕРШЕНА ---")
print("--- Завдання 2.4: Узагальнення результатів ---")

# Створюємо Pandas DataFrame для гарної таблиці
results_df = pd.DataFrame(results_list)

# Сортуємо для зручності: спочатку тип даних, потім розмір вікна
results_df.sort_values(by=["Data Type", "N (Window Size)"], inplace=True)

print(results_df)

# Зберігаємо у CSV для звіту
results_df.to_csv('lab3_part2_results.csv', index=False)
print("\nРезультати збережено у 'lab3_part2_results.csv'")

print("\n--- ✅ Кінець Частини 2 ---")