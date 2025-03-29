# -*- coding: utf-8 -*-
"""
COMP 247 Lab Assignment: Ensemble Learning (By: tian)
Dataset: pima-indians-diabetes.csv
"""

# 1. 引入所需库 Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

# 2. 加载数据 Load the dataset into DataFrame
file_path = "/Users/troy/Downloads/pima-indians-diabetes.csv"
df_tian = pd.read_csv(file_path, header=None)
df_tian.columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# 3. 特征和标签拆分 Feature / Label split
X = df_tian.drop("Outcome", axis=1)
y = df_tian["Outcome"]

# 4. 标准化转换器定义 StandardScaler transformer
tf_tian = StandardScaler()

# 5. 拆分训练与测试集 Split into training and test sets
X_train_tian, X_test_tian, y_train_tian, y_test_tian = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 6. 对训练和测试特征进行标准化 Fit and transform
X_train_tian = tf_tian.fit_transform(X_train_tian)
X_test_tian = tf_tian.transform(X_test_tian)

# 7. 定义五个基础分类器 Define 5 base classifiers
clf_L = LogisticRegression(max_iter=1400)
clf_R = RandomForestClassifier()
clf_S = SVC(probability=True)  # Soft voting 需要 probability=True
clf_T = DecisionTreeClassifier(criterion="entropy", max_depth=42)
clf_E = ExtraTreesClassifier()

# 8. 硬投票 Hard voting classifier
hard_voting = VotingClassifier(estimators=[
    ('lr', clf_L), ('rf', clf_R), ('svc', clf_S), ('dt', clf_T), ('et', clf_E)
], voting='hard')

hard_voting.fit(X_train_tian, y_train_tian)

print("\n硬投票预测前3个结果 Hard Voting Predictions:", hard_voting.predict(X_test_tian[:3]))
print("实际值 True Labels:", y_test_tian[:3].values)

# 9. 输出每个模型对前3项的预测 Print each classifier's prediction
models = [('LR', clf_L), ('RF', clf_R), ('SVM', clf_S), ('DT', clf_T), ('ET', clf_E)]
print("\n各模型预测前3个结果 Each model's predictions:")
for name, model in models:
    model.fit(X_train_tian, y_train_tian)
    preds = model.predict(X_test_tian[:3])
    print(f"分类器 {name} 的预测: {preds}")

# 10. 软投票 Soft voting classifier
soft_voting = VotingClassifier(estimators=[
    ('lr', clf_L), ('rf', clf_R), ('svc', clf_S), ('dt', clf_T), ('et', clf_E)
], voting='soft')
soft_voting.fit(X_train_tian, y_train_tian)
print("\n软投票预测前3个结果 Soft Voting Predictions:", soft_voting.predict(X_test_tian[:3]))

# 11. 构建两个Pipeline模型（ExtraTrees 和 DecisionTree）
pipeline1_tian = Pipeline([
    ('scaler', tf_tian),
    ('et', clf_E)
])

pipeline2_tian = Pipeline([
    ('scaler', tf_tian),
    ('dt', clf_T)
])

# 12. 使用原始数据进行交叉验证 Cross-validation
scores1 = cross_val_score(pipeline1_tian, X, y, cv=10)
scores2 = cross_val_score(pipeline2_tian, X, y, cv=10)

print("\nPipeline1 (ExtraTrees) 平均准确率:", scores1.mean())
print("Pipeline2 (DecisionTree) 平均准确率:", scores2.mean())

# 13. 在测试集上进行预测并输出评估指标 Evaluate on test set
print("\n模型评估 Metrics:")
for name, pipe in zip(['ET', 'DT'], [pipeline1_tian, pipeline2_tian]):
    pipe.fit(X_train_tian, y_train_tian)
    y_pred = pipe.predict(X_test_tian)
    print(f"\n{name} 模型：")
    print("混淆矩阵 Confusion Matrix:\n", confusion_matrix(y_test_tian, y_pred))
    print("准确率 Accuracy:", accuracy_score(y_test_tian, y_pred))
    print("精确率 Precision:", precision_score(y_test_tian, y_pred))
    print("召回率 Recall:", recall_score(y_test_tian, y_pred))

# 14. 随机参数搜索 Randomized Grid Search
param_dist = {
    'et__n_estimators': list(range(10, 3000, 20)),
    'et__max_depth': list(range(1, 1000, 2))
}

grid_tian = RandomizedSearchCV(
    estimator=pipeline1_tian,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    random_state=42,
    verbose=1
)
grid_tian.fit(X_train_tian, y_train_tian)

print("\n最佳参数 Best Parameters:", grid_tian.best_params_)
print("最佳准确率 Best Accuracy:", grid_tian.best_score_)

# 15. 使用最佳模型预测并评估 Test best estimator
y_pred_best = grid_tian.best_estimator_.predict(X_test_tian)
print("\nFine-Tuned Model on Test:")
print("准确率 Accuracy:", accuracy_score(y_test_tian, y_pred_best))
print("精确率 Precision:", precision_score(y_test_tian, y_pred_best))
print("召回率 Recall:", recall_score(y_test_tian, y_pred_best))
