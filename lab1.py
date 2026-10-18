import shutil
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# ------------------------------------------------------------
# 1. Настройка путей и очистка папки результатов
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "lab1_results"

if RESULT_DIR.exists():
    shutil.rmtree(RESULT_DIR)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def save_result(df, name):
    """Сохраняет DataFrame в csv, txt, xlsx."""
    df.to_csv(RESULT_DIR / f"{name}.csv", index=False, encoding='utf-8-sig')
    df.to_csv(RESULT_DIR / f"{name}.txt", index=False, sep='\t', encoding='utf-8')
    try:
        df.to_excel(RESULT_DIR / f"{name}.xlsx", index=False, engine='openpyxl')
    except ImportError:
        print("Модуль openpyxl не установлен. Установите: pip install openpyxl")
    except Exception as e:
        print(f"Ошибка при сохранении {name}.xlsx: {e}")


# ------------------------------------------------------------
# 2. Автоматический поиск и загрузка датасета
# ------------------------------------------------------------
datasets_dir = BASE_DIR / "datasets"

# Создаем папку datasets, если её нет
if not datasets_dir.exists():
    datasets_dir.mkdir(parents=True, exist_ok=True)

# Ищем все CSV файлы в папке datasets
csv_files = list(datasets_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"В папке '{datasets_dir}' не найдено ни одного .csv файла.\n"
        f"Пожалуйста, поместите файл с датасетом в эту папку и запустите снова."
    )

# Берем первый найденный файл
dataset_path = csv_files[0]
print(f"Автоматически выбран датасет: {dataset_path.name}")

df = pd.read_csv(dataset_path)

# Исключаем Student_ID, если он есть (errors='ignore' защищает от ошибки, если колонки нет)
df = df.drop(columns=['Student_ID'], errors='ignore')

# ВНИМАНИЕ: Если вы загрузите совершенно другой датасет, 
# вам нужно будет изменить 'Backlogs' на название вашей целевой колонки
target_col = 'Backlogs'

if target_col not in df.columns:
    raise ValueError(f"В датасете нет целевой колонки '{target_col}'. Пожалуйста, проверьте данные.")

features = df.drop(columns=[target_col])
target = df[target_col]

numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = features.select_dtypes(exclude=[np.number]).columns.tolist()

for col in numeric_cols:
    features[col] = features[col].fillna(features[col].median())

for col in categorical_cols:
    mode_val = features[col].mode()
    features[col] = features[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')

target = target.fillna(target.median() if not target.mode().empty else 0)


# ------------------------------------------------------------
# 3. Масштабирование числовых признаков и ВИЗУАЛИЗАЦИЯ
# ------------------------------------------------------------
numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()

scalers = {
    'standard_scaler': StandardScaler(),
    'robust_scaler': RobustScaler(),
    'minmax_scaler': MinMaxScaler(),
    'normalizer': Normalizer(norm='l2')
}

# Сохраняем таблицы с результатами
for name, scaler in scalers.items():
    scaled = scaler.fit_transform(features[numeric_features])
    scaled_df = pd.DataFrame(scaled, columns=[f"{col}_scaled" for col in numeric_features])
    save_result(scaled_df, f"2_{name}")

# --- Построение графиков масштабирования ---
print("Построение графиков масштабирования...")

# Защита: берем первые два числовых признака для визуализации (чтобы не зависеть от жестких имен)
if len(numeric_features) >= 2:
    f1, f2 = numeric_features[0], numeric_features[1]
else:
    raise ValueError("Для построения графика нужно хотя бы 2 числовых признака.")

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('Различные способы масштабирования и предварительной обработки данных', fontsize=14)

# Исходные данные
axes[0].scatter(features[f1], features[f2], c=target, cmap='bwr', alpha=0.7, edgecolors='k')
axes[0].set_title('Original Data')
axes[0].set_xlabel(f1)
axes[0].set_ylabel(f2)

# Масштабированные данные
for i, (name, scaler) in enumerate(scalers.items(), start=1):
    X_subset = features[[f1, f2]]
    scaled_subset = scaler.fit_transform(X_subset)
    axes[i].scatter(scaled_subset[:, 0], scaled_subset[:, 1], c=target, cmap='bwr', alpha=0.7, edgecolors='k')
    axes[i].set_title(name.replace('_', ' ').title())
    axes[i].set_xlabel('Feature 1')
    axes[i].set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig(RESULT_DIR / "scaling_visualization.png", dpi=150)
plt.close()


# ------------------------------------------------------------
# 4. Прямое кодирование категориальных признаков
# ------------------------------------------------------------
features_enc = features.copy()

# Кодируем ВСЕ категориальные колонки, которые есть в датасете (делаем код универсальным)
encoded = pd.get_dummies(features_enc, columns=categorical_cols, dummy_na=True, drop_first=False)
save_result(encoded, "3_one_hot_encoded")

# ------------------------------------------------------------
# 5. Отбор признаков с разными alpha
# ------------------------------------------------------------
# Удаляем константные признаки
selector_var = VarianceThreshold(threshold=0.0)
X_var = selector_var.fit_transform(encoded)
retained_cols = encoded.columns[selector_var.get_support()]
X = pd.DataFrame(X_var, columns=retained_cols)

y = target.astype(int)

# Считаем F-статистику для всех признаков
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)

feat_scores = pd.DataFrame({
    'feature': X.columns,
    'f_score': selector.scores_,
    'p_value': selector.pvalues_
}).dropna()

save_result(feat_scores, "4_all_feature_scores")

# Проверяем разные пороги alpha (p-value)
alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
print("\nВлияние параметра alpha на количество отобранных признаков:")
for alpha in alphas:
    selected = feat_scores[feat_scores['p_value'] < alpha].sort_values('p_value')
    print(f"  - alpha = {alpha}: отобрано {len(selected)} признаков")
    save_result(selected, f"4_selected_features_alpha_{str(alpha).replace('.', '_')}")

# ------------------------------------------------------------
# 6. Обработка текстовых данных (Общий корпус)
# ------------------------------------------------------------
texts_dir = BASE_DIR / "texts"

if not texts_dir.exists():
    print(f"\nВнимание: Папка '{texts_dir}' не найдена. Создайте её и поместите туда .txt файлы.")
else:
    text_files = list(texts_dir.glob("*.txt"))
    print(f"\nНайдено текстовых файлов: {len(text_files)}")

    if len(text_files) > 0:
        corpus = []
        filenames = []
        
        for path in text_files:
            text = path.read_text(encoding='utf-8')
            text = re.sub(r'<br\s*/?>', ' ', text)
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            corpus.append(text)
            filenames.append(path.stem)

        # --- Мешок слов ---
        vectorizer = CountVectorizer(stop_words='english')
        X_counts = vectorizer.fit_transform(corpus)
        
        bow_df = pd.DataFrame(X_counts.toarray(), index=filenames, columns=vectorizer.get_feature_names_out()).T
        save_result(bow_df, "5_bag_of_words_corpus")

        # --- TF-IDF ---
        transformer = TfidfTransformer(smooth_idf=True, norm=None)
        X_tfidf = transformer.fit_transform(X_counts)
        
        tfidf_df = pd.DataFrame(X_tfidf.toarray(), index=filenames, columns=vectorizer.get_feature_names_out()).T
        save_result(tfidf_df, "5_tfidf_corpus")

# ------------------------------------------------------------
# 7. Формирование отчёта
# ------------------------------------------------------------
report_lines = [
    "Лабораторная работа №1: Методы предварительной обработки данных",
    "=" * 80,
    "Постановка задачи:",
    f"1. Загрузить датасет {dataset_path.name}.",
    "2. Реализовать масштабирование: StandardScaler, RobustScaler, MinMaxScaler, Normalizer.",
    "3. Кодирование категориальных признаков методом прямого кодирования (one-hot).",
    "4. Отбор признаков с помощью одномерных статистик (f_classif, разные alpha).",
    "5. Обработка текстов: мешок слов и tf-idf для всего корпуса текстов.",
    "",
    "Ход работы:",
    f"Датасет {dataset_path.name} загружен. Исключён Student_ID (если он был).",
    f"Целевая переменная: {target_col}. Признаки: все остальные.",
    "Пропуски в числовых признаках заполнены медианой, в категориальных — модой.",
    f"Масштабирование применено ко всем числовым признакам. Построен график сравнения (признаки: {f1} и {f2}).",
    "Все категориальные признаки закодированы one-hot.",
    "Перед отбором признаков удалены константные столбцы (дисперсия = 0).",
    "Отбор признаков выполнен с помощью f_classif. Проверены пороги alpha: 0.01, 0.05, 0.1, 0.2, 0.5.",
    "Тексты объединены в единый корпус для корректного расчета IDF.",
    "Построены общие модели мешка слов (CountVectorizer) и tf-idf (TfidfTransformer).",
    "",
    "Результаты:",
    f"Файлы результатов сохранены в: {RESULT_DIR}",
    "Список файлов:"
]

for f in sorted(RESULT_DIR.iterdir()):
    report_lines.append(f"  - {f.name}")

report_lines.extend([
    "",
    "Выводы:",
    "В ходе работы освоены методы предварительной обработки данных: масштабирование, кодирование категориальных признаков, отбор признаков, обработка текстов. Изменение параметра alpha позволяет гибко настраивать количество оставляемых признаков.",
    ""
])

report_text = "\n".join(report_lines)
(RESULT_DIR / "report.txt").write_text(report_text, encoding='utf-8')
print(report_text)
