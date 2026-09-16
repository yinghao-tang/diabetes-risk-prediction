# Diabetes Risk Prediction

## 1.项目介绍

项目作为一次小练习

本项目基于患者临床指标、生活方式和人口统计信息，
构建机器学习模型预测糖尿病风险等级。

目标是通过患者的健康相关特征：

- 年龄
- 身体指标
- 血糖指标
- 血脂指标
- 生活方式
- 家族病史

预测：

- Low
- Moderate
- High

三个糖尿病风险等级。

该项目主要用于学习完整的机器学习项目流程，包括：

- 数据探索（EDA）
- 数据清洗
- 特征工程
- 模型训练
- 模型评估
- 模型解释


---

# 2. Dataset

数据集：

Diabetes Risk Prediction Dataset

数据规模：

- 样本数量：50,000
- 特征数量：41

包含：

- 人口统计信息
- 身体测量指标
- 血糖相关指标
- 心血管指标
- 生活方式信息
- 家族病史

目标变量：

Diabetes_Risk

类别：

High
Moderate
Low


---

# 3. Project Structure

diabetes-risk-prediction/
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_EDA.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│
├── results/
│
├── README.md
└── requirements.txt


---

# 4. Exploratory Data Analysis

## 4.1 Target Distribution

目标变量分布：

| Risk | Count |
|---|---:|
| High | 36593 |
| Moderate | 12937 |
| Low | 470 |


发现：

数据存在严重类别不平衡问题。

High 类别占比约 73%，
Low 类别不足 1%。

因此后续模型评估不能只依赖 Accuracy，
需要关注：

- Precision 模型预测为 Low 的人里面，真正是 Low 的比例
- Recall 所有真正是 Low 的人里面，模型成功找出来的比例
- F1-score F1 = 2 * (Precision * Recall) / (Precision + Recall)
- Confusion Matrix


---

# 5. Data Leakage Analysis

在 EDA 阶段发现部分特征存在目标泄漏风险。


## 5.1 Diabetes_Risk_Score

分析发现：

| Risk | Score Range |
|-|-|
| Low | 13-34 |
| Moderate | 35-64 |
| High | 65-100 |


该特征与目标变量存在近乎完全分离关系。

推测：

Diabetes_Risk_Score
可能由 Diabetes_Risk 生成。

因此：

Diabetes_Risk_Score将在模型训练阶段删除。

## 5.2 AI_Health_Recommendation

发现：

不同 AI 建议几乎完全对应不同风险类别。

例如：

Consult Endocrinologist Immediately
        ↓
High Risk

因此该变量属于预测结果之后生成的信息。

为了避免数据泄漏：

AI_Health_Recommendation不参与模型训练

## 5.3 Doctor_Consultation_Needed

分析发现：

该变量与风险等级高度相关。

由于医生咨询需求通常是在风险评估之后产生，

因此不作为预测输入。


---

# 6. Current Progress

目前已经完成：

- [x] 数据读取
- [x] 数据基本信息分析
- [x] 缺失值检查
- [x] 目标变量分布分析
- [x] 数据泄漏分析


---

# 7. Next Steps

下一步计划：

## Data Processing

- 删除无效字段
- 处理缺失值
- 类别变量编码
- 特征标准化


## Modeling

建立 Baseline：

- Logistic Regression

进一步比较：

- Decision Tree
- Random Forest
- XGBoost
- CatBoost


## Evaluation

评价指标：

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
