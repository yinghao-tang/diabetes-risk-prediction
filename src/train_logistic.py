from pathlib import Path

from data_processing import load_data, preprocess_data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1.加载数据
ROOT_DIR = Path(__file__).resolve().parent.parent

data_path = ROOT_DIR / "data" / "raw" / "diabetes_risk_prediction_dataset.csv"
df = load_data(data_path)

# 2.数据预处理

X, y, preprocesser = preprocess_data(df)

# 3.划分数据集和训练集

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=12,
    stratify=y #按照y的类型比例分类
)

# 4.创造模型

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# 5.训练

model.fit(
    X_train,
    y_train
)

# 6.测试

y_pred = model.predict(
    X_test
)

# 7.评价

print(
    classification_report(
        y_test,
        y_pred
    )
)