# --- Розділ 1: Імпорт необхідних бібліотек ---
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

print("--- Скрипт запущено. Початок Частини 1 ---")

# --- Розділ 2: Завантаження та підготовка даних ---
file_path = 'weatherHistory.csv'  # Назва файлу
try:
    # 1. Завантаження CSV
    data = pd.read_csv(file_path)
    # 2. Конвертація 'Formatted Date' в datetime
    data['Formatted Date'] = pd.to_datetime(data['Formatted Date'], utc=True)
    # 3. Встановлення дати як індекс
    data.set_index('Formatted Date', inplace=True)
    # 4. Вибір цільового стовпця
    ts_hourly = data['Temperature (C)']
except Exception as e:
    print(f"Помилка при завантаженні або обробці файлу: {e}")
    exit()

# 5. Дані погодинні. Агрегуємо їх у щоденні максимальні.
# .resample('D') - групує дані по Днях (Daily)
# .max() - беремо максимальне значення за кожен день
ts_daily_max = ts_hourly.resample('D').max()

# 6. Видаляємо пропуски (NaN)
ts_daily_max = ts_daily_max.dropna()

print(f"Дані завантажено та перетворено на щоденні (всього {len(ts_daily_max)} записів).\n")
# Зберігаємо "сирі" значення для подальшої роботи
raw_data_values = ts_daily_max.values

# --- Розділ 3: Статистичний аналіз (Завдання 1.1) ---
print("--- Завдання 1.1: Статистичні показники ряду ---")

# Розрахунок основних статистичних показників
stats = ts_daily_max.describe()
print(stats)

# Додатково рахуємо асиметрію та куртозис
skewness = ts_daily_max.skew()
print(f"Асиметрія (Skewness): {skewness:.3f}")
kurtosis = ts_daily_max.kurt()
print(f"Куртозис (Kurtosis): {kurtosis:.3f}")
print("-" * 30 + "\n")

# --- Розділ 4: Візуалізація даних (Завдання 1.2) ---
print("--- Завдання 1.2: Побудова графіків ---")

# Графік 1: Динаміка змін в часі
plt.figure(figsize=(14, 7))
ts_daily_max.plot()
plt.title('Графік 1: Динаміка щоденної макс. температури (2006-2016)')
plt.xlabel('Дата')
plt.ylabel('Макс. Температура (°C)')
plt.grid(True)
plt.savefig('grafik_1_timeseries.png')  # Зберігаємо у файл
print("Збережено 'grafik_1_timeseries.png'")

# Графік 2: Гістограма розподілу
plt.figure(figsize=(10, 6))
ts_daily_max.hist(bins=50)
plt.title('Графік 2: Гістограма розподілу щоденних макс. температур')
plt.xlabel('Макс. Температура (°C)')
plt.ylabel('Частота (кількість днів)')
plt.savefig('grafik_2_histogram.png')  # Зберігаємо у файл
print("Збережено 'grafik_2_histogram.png'")
print("-" * 30 + "\n")
# plt.show() # Розкоментувати, щоб показати графіки під час виконання

# --- Розділ 5: Нормалізація (Завдання 1.3) ---
print("--- Завдання 1.3: Нормалізація даних ---")

# 1. Reshape даних для Scaler (потрібен 2D-масив)
raw_data_2d = raw_data_values.reshape(-1, 1)

# 2. Створення та навчання MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized_2d = scaler.fit_transform(raw_data_2d)

# 3. Повернення до 1D-масиву для подальшої обробки
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
    # Формуємо послідовності X та y
    for i in range(len(data) - n_steps):
        end_ix = i + n_steps
        # seq_x - вхідна послідовність, seq_y - цільове значення
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


print("Функцію create_sequences() створено.\n")

# --- Розділ 7: Формування наборів даних (Завдання 1.5) ---
print("--- Завдання 1.5: Формування 5 наборів даних ---")

# Обираємо 5 розмірів вікна (n)
n_steps_list = [7, 14, 30, 60, 90]  # напр: тиждень, 2 тижні, місяць, 2 міс, 3 міс

# Словники для зберігання наборів даних
normalized_datasets = {}
raw_datasets = {}

for n in n_steps_list:
    print(f"Створюємо набори для n = {n}...")
    # 1. Для нормалізованих даних
    X_norm, y_norm = create_sequences(data_normalized_1d, n)
    normalized_datasets[n] = (X_norm, y_norm)
    # 2. Для ненормалізованих ("сирих") даних
    X_raw, y_raw = create_sequences(raw_data_values, n)
    raw_datasets[n] = (X_raw, y_raw)

print("\n--- Перевірка створених наборів ---")
# Перевірка розмірності створених наборів (на прикладі n=30)
n_example = 30
X_check, y_check = normalized_datasets[n_example]
print(f"Для n={n_example} (нормалізовані):")
print(f"  Форма (shape) X: {X_check.shape}")
print(f"  Форма (shape) y: {y_check.shape}")

print("\n--- Частина 1 Завершена ---")

# --- Початок Частини 2: Моделювання ---
print("\n\n--- Початок Частини 2: Моделювання ---")

# --- Розділ 8: Розбиття даних на вибірки (Завдання 2.1) ---
print("--- Завдання 2.1: Підготовка до розбиття даних ---")

# Для часових рядів дані не можна перемішувати.
# Пропорції: 70% - навчання, 15% - валідація, 15% - тест.
train_percent = 0.7
val_percent = 0.15
print(f"Пропорції розбиття: {train_percent * 100}% train, {val_percent * 100}% validation, 15% test")


def split_sequences(X, y, train_pct, val_pct):
    """
    Розбиває X та y послідовно, без перемішування.
    """
    total_samples = len(X)
    train_idx = int(total_samples * train_pct)
    val_idx = int(total_samples * (train_pct + val_pct))

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    return X_train, y_train, X_val, y_val, X_test, y_test


print("Функцію split_sequences() створено.\n")

# --- Розділ 9: Архітектура моделі LSTM (Завдання 2.2) ---
print("--- Завдання 2.2: Створення функції для побудови моделі ---")


def build_model(n_steps_in, n_features_in=1):
    """
    Створює та компілює модель LSTM "Багато-до-Одного".
    """
    model = Sequential()
    # n_steps_in - розмір вікна, n_features_in - кількість ознак
    input_shape = (n_steps_in, n_features_in)

    # Шар LSTM з 50 нейронами.
    # 'return_sequences=False' (за замовчуванням) означає,
    # що шар видасть лише останній вихід.
    model.add(LSTM(50, activation='relu', input_shape=input_shape))

    # Вихідний шар (1 нейрон для прогнозу 1 значення)
    model.add(Dense(1))

    # Компіляція моделі
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


print("Функцію build_model() створено.\n")

# --- Розділ 10: Цикл експериментів (Завдання 2.3) ---
print("--- Завдання 2.3: Запуск циклу експериментів ---")

# Список для зберігання результатів експериментів
results_list = []

EPOCHS = 20  # Кількість епох навчання
BATCH_SIZE = 32  # Розмір "пачки" даних

all_experiments = {
    "Normalized": normalized_datasets,
    "Raw (Unnormalized)": raw_datasets
}

for data_type, datasets in all_experiments.items():
    print(f"\n--- ОБРОБКА ТИПУ ДАНИХ: {data_type} ---")
    for n_steps in n_steps_list:
        start_time = time.time()
        print(f"  > Початок експерименту: n = {n_steps}, Тип: {data_type}")

        # 1. Отримання даних
        X, y = datasets[n_steps]

        # 2. Розбиття даних
        X_train, y_train, X_val, y_val, X_test, y_test = \
            split_sequences(X, y, train_percent, val_percent)

        # 3. Зміна форми X для LSTM (N, steps, features)
        n_features = 1
        X_train_3d = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
        X_val_3d = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))
        X_test_3d = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

        # 4. Побудова моделі
        model = build_model(n_steps_in=n_steps, n_features_in=n_features)

        # 5. Навчання моделі
        print(f"    ...Навчання ({EPOCHS} епох)...")
        history = model.fit(
            X_train_3d, y_train,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_data=(X_val_3d, y_val),
            verbose=0  # 0=тихий режим
        )
        print("    ...Навчання завершено.")

        # 6. Оцінка на тестових даних
        y_pred = model.predict(X_test_3d)

        # 7. Розрахунок помилки (RMSE)
        # Помилка розраховується в однакових одиницях (градусах Цельсія)
        if data_type == "Normalized":
            # 7a. Для нормалізованих даних, повертаємо прогноз до початкового масштабу
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_inv = scaler.inverse_transform(y_pred)
            rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        else:
            # 7b. Для "сирих" даних, просто рахуємо RMSE
            rmse = math.sqrt(mean_squared_error(y_test, y_pred))

        end_time = time.time()
        duration = end_time - start_time

        # 8. Збереження результатів
        results_list.append({
            "Data Type": data_type,
            "N (Window Size)": n_steps,
            "RMSE (Test)": rmse,
            "Time (sec)": duration,
            "Epochs": EPOCHS
        })
        print(f"    > Готово! RMSE: {rmse:.3f} °C (Час: {duration:.1f} сек)")

print("\n--- Частина 2 Завершена ---")

# --- Початок Частини 3: Врахування трендів та сезонності ---
print("\n\n--- Початок Частини 3: Врахування трендів та сезонності ---")

# --- Розділ 11: Feature Engineering (Завдання 3.1) ---
print("--- Завдання 3.1: Створення додаткових ознак (сезонність) ---")

# Створюємо DataFrame з датами для додавання ознак
features_df = ts_daily_max.to_frame()
dates_index = features_df.index
day_of_year = dates_index.day_of_year
month = dates_index.month

# Набір 1: Синус/Косинус перетворення ДНЯ РОКУ (річний цикл)
features_df['day_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
features_df['day_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

# Набір 2: Синус/Косинус перетворення МІСЯЦЯ (місячний цикл)
features_df['month_sin'] = np.sin(2 * np.pi * month / 12)
features_df['month_cos'] = np.cos(2 * np.pi * month / 12)

# Додаємо нормалізовану температуру до DataFrame
features_df['temp_scaled'] = data_normalized_1d

print("Створено 2 набори додаткових ознак (день року, місяць).")


# --- Розділ 12: Нова функція "ковзного вікна" для >1 ознаки ---
def create_sequences_multivariate(data_df, n_steps, target_col_name):
    """
    Створює послідовності для багатовимірних даних.
    X буде (N, n_steps, n_features)
    y буде (N,) і братиметься з target_col_name
    """
    X, y = [], []
    data_values = data_df.values
    # Отримуємо індекс цільової колонки (яку прогнозуємо)
    target_col_idx = data_df.columns.get_loc(target_col_name)

    for i in range(len(data_values) - n_steps):
        end_ix = i + n_steps
        # X - це послідовність ВСІХ ознак
        seq_x = data_values[i:end_ix, :]
        # y - це ТІЛЬКИ температура в наступний момент часу
        seq_y = data_values[end_ix, target_col_idx]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


print("Функцію create_sequences_multivariate() створено.\n")

# --- Розділ 13: Моделювання з дод. ознаками (Завдання 3.2) ---
print("--- Завдання 3.2: Моделювання з дод. ознаками ---")

# Використовуємо n=90 як один з найкращих варіантів з Частини 2
n_steps = 90

# [ (назва, DataFrame з ознаками, кількість ознак) ]
experiments_part3 = [
    (
        "Normalized + DayOfYear (n=90)",
        features_df[['temp_scaled', 'day_sin', 'day_cos']],
        3
    ),
    (
        "Normalized + Month (n=90)",
        features_df[['temp_scaled', 'month_sin', 'month_cos']],
        3
    )
]

for exp_name, exp_df, n_features in experiments_part3:
    start_time = time.time()
    print(f"  > Початок експерименту: {exp_name}")

    # 1. Створення послідовностей
    X, y = create_sequences_multivariate(exp_df, n_steps, 'temp_scaled')

    # 2. Розбиття даних
    X_train, y_train, X_val, y_val, X_test, y_test = \
        split_sequences(X, y, train_percent, val_percent)

    # 3. Побудова моделі (n_features_in тепер 3)
    model = build_model(n_steps_in=n_steps, n_features_in=n_features)

    # 4. Навчання моделі
    print(f"    ...Навчання ({EPOCHS} епох)...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val), verbose=0
    )
    print("    ...Навчання завершено.")

    # 5. Оцінка
    y_pred = model.predict(X_test)

    # 6. Розрахунок RMSE (з поверненням до початкового масштабу)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)
    rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

    end_time = time.time()
    duration = end_time - start_time

    # 7. Збереження результатів
    results_list.append({
        "Data Type": exp_name,
        "N (Window Size)": n_steps,
        "RMSE (Test)": rmse,
        "Time (sec)": duration,
        "Epochs": EPOCHS
    })
    print(f"    > Готово! RMSE: {rmse:.3f} °C (Час: {duration:.1f} сек)")

# --- Розділ 14: Модифікація архітектури (Завдання 3.3) ---
print("\n--- Завдання 3.3: Моделювання зі Stacked LSTM ---")


def build_stacked_model(n_steps_in, n_features_in):
    """
    Створює "глибоку" (stacked) модель з двома шарами LSTM.
    """
    model = Sequential()
    input_shape = (n_steps_in, n_features_in)

    # Шар 1: return_sequences=True для передачі повної послідовності наступному шару
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape))

    # Шар 2:
    model.add(LSTM(50, activation='relu'))

    # Вихідний шар
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


print("Функцію build_stacked_model() створено.")

# Тестуємо stacked-модель на найкращому наборі ознак (DayOfYear)
start_time = time.time()
exp_name = "Stacked LSTM + DayOfYear (n=90)"
n_steps = 90
n_features = 3
print(f"  > Початок експерименту: {exp_name}")

# 1. Готуємо дані (аналогічно до Розділу 13)
exp_df = features_df[['temp_scaled', 'day_sin', 'day_cos']]
X, y = create_sequences_multivariate(exp_df, n_steps, 'temp_scaled')
X_train, y_train, X_val, y_val, X_test, y_test = \
    split_sequences(X, y, train_percent, val_percent)

# 2. Будуємо "глибоку" модель
stacked_model = build_stacked_model(n_steps_in=n_steps, n_features_in=n_features)

# 3. Навчаємо
print(f"    ...Навчання ({EPOCHS} епох)...")
history = stacked_model.fit(
    X_train, y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val), verbose=0
)
print("    ...Навчання завершено.")

# 4. Оцінка
y_pred = stacked_model.predict(X_test)

# 5. Розрахунок RMSE
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)
rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

end_time = time.time()
duration = end_time - start_time

# 6. Збереження результату
results_list.append({
    "Data Type": exp_name,
    "N (Window Size)": n_steps,
    "RMSE (Test)": rmse,
    "Time (sec)": duration,
    "Epochs": EPOCHS
})
print(f"    > Готово! RMSE: {rmse:.3f} °C (Час: {duration:.1f} сек)")

# --- Фінальні результати ---
print("\n\n--- Всі експерименти завершено ---")
print("--- Загальна таблиця результатів (Частини 2 та 3) ---")

# Створення фінального DataFrame з результатами
results_df = pd.DataFrame(results_list)

# Сортування за RMSE (від найкращого до найгіршого)
results_df.sort_values(by="RMSE (Test)", inplace=True)

print(results_df)

# Збереження у CSV для звіту
results_df.to_csv('lab3_FINAL_results.csv', index=False)
print("\nРезультати збережено у 'lab3_FINAL_results.csv'")

print("\n--- Кінець роботи ---")