import shutil
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.feature_selection import SelectKBest, f_classif
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
# 2. Загрузка датасета и предварительная обработка
# ------------------------------------------------------------
dataset_path = BASE_DIR / "student_burnout_dropout_dataset_2-selected-columns.csv"
if not dataset_path.exists():
    raise FileNotFoundError(f"Датасет не найден: {dataset_path}")

df = pd.read_csv(dataset_path)
df = df.drop(columns=['Student_ID', 'Study_Hours_Per_Day'], errors='ignore')

target_col = 'Backlogs'
features = df.drop(columns=[target_col])
target = df[target_col]

numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = features.select_dtypes(exclude=[np.number]).columns.tolist()

for col in numeric_cols:
    features[col] = features[col].fillna(features[col].median())

for col in categorical_cols:
    mode_val = features[col].mode()
    features[col] = features[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')

target = target.fillna(target.median())

# ------------------------------------------------------------
# 3. Масштабирование числовых признаков
# ------------------------------------------------------------
numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()

scalers = {
    'standard_scaler': StandardScaler(),
    'robust_scaler': RobustScaler(),
    'minmax_scaler': MinMaxScaler(),
    'normalizer': Normalizer(norm='l2')
}

for name, scaler in scalers.items():
    scaled = scaler.fit_transform(features[numeric_features])
    scaled_df = pd.DataFrame(scaled, columns=[f"{col}_scaled" for col in numeric_features])
    save_result(scaled_df, f"2_{name}")

# ------------------------------------------------------------
# 4. Прямое кодирование категориальных признаков
# ------------------------------------------------------------
features_enc = features.copy()
cat_cols_to_encode = ['Gender', 'Year_of_Study', 'Department', 'Residence_Type']

for col in cat_cols_to_encode:
    if col in features_enc.columns:
        features_enc[col] = features_enc[col].astype(str)

encoded = pd.get_dummies(features_enc, columns=cat_cols_to_encode, dummy_na=True, drop_first=False)
save_result(encoded, "3_one_hot_encoded")

# ------------------------------------------------------------
# 5. Отбор признаков с помощью одномерной статистики
# ------------------------------------------------------------
X = encoded
y = target.astype(int)

selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)

feat_scores = pd.DataFrame({
    'feature': X.columns,
    'f_score': selector.scores_,
    'p_value': selector.pvalues_
})

selected_features = feat_scores[feat_scores['p_value'] < 0.05].sort_values('p_value')
save_result(feat_scores, "4_all_feature_scores")
save_result(selected_features, "4_selected_features_p005")

# ------------------------------------------------------------
# 6. Обработка текстовых данных (мешок слов и tf-idf)
# ------------------------------------------------------------
text_files = ['559_4.txt', '3_4.txt', '555_4.txt', '556_1.txt']

for fname in text_files:
    path = BASE_DIR / fname
    if not path.exists():
        print(f"Предупреждение: файл {path} не найден, пропуск.")
        continue

    text = path.read_text(encoding='utf-8')
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    vectorizer = CountVectorizer(stop_words='english')
    X_counts = vectorizer.fit_transform([text])
    bow_df = pd.DataFrame(X_counts.toarray(), columns=vectorizer.get_feature_names_out())
    bow_df = bow_df.T.reset_index()
    bow_df.columns = ['term', 'count']
    bow_df = bow_df.sort_values('count', ascending=False)
    base_name = Path(fname).stem
    save_result(bow_df, f"5_bag_of_words_{base_name}")

    transformer = TfidfTransformer(smooth_idf=True, norm=None)
    X_tfidf = transformer.fit_transform(X_counts)
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df = tfidf_df.T.reset_index()
    tfidf_df.columns = ['term', 'tfidf']
    tfidf_df = tfidf_df.sort_values('tfidf', ascending=False)
    save_result(tfidf_df, f"5_tfidf_{base_name}")

# ------------------------------------------------------------
# 7. Формирование отчёта
# ------------------------------------------------------------
report_lines = [
    "Лабораторная работа №1: Методы предварительной обработки данных",
    "=" * 80,
    "Постановка задачи:",
    "1. Загрузить датасет student_burnout_dropout_dataset_2-selected-columns.csv.",
    "2. Реализовать масштабирование: StandardScaler, RobustScaler, MinMaxScaler, Normalizer.",
    "3. Кодирование категориальных признаков методом прямого кодирования (one-hot).",
    "4. Отбор признаков с помощью одномерных статистик (f_classif, порог p < 0.05).",
    "5. Обработка текстов: мешок слов и tf-idf для каждого файла отдельно.",
    "",
    "Ход работы:",
    "Датасет загружен. Исключены Student_ID и Study_Hours_Per_Day.",
    f"Целевая переменная: {target_col}. Признаки: все остальные.",
    "Пропуски в числовых признаках заполнены медианой, в категориальных — модой.",
    "Масштабирование применено к числовым признакам.",
    "Категориальные признаки (Gender, Year_of_Study, Department, Residence_Type) закодированы one-hot.",
    "Отбор признаков выполнен с помощью f_classif и SelectKBest, выбраны признаки с p < 0.05.",
    "Тексты обработаны: удалены HTML-теги, приведены к нижнему регистру, удалена пунктуация.",
    "Построены модели мешка слов (CountVectorizer) и tf-idf (TfidfTransformer) для каждого файла.",
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
    "В ходе работы освоены методы предварительной обработки данных: масштабирование, кодирование категориальных признаков, отбор признаков, обработка текстов.",
    ""
])

report_text = "\n".join(report_lines)
(RESULT_DIR / "report.txt").write_text(report_text, encoding='utf-8')
print(report_text)