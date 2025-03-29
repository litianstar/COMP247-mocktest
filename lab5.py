# 导入所需的库
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold

# 第1部分：数据加载与初步分析
# Part 1: Load and explore the data

# 指定列名（从官方元数据中获取）
# Define column names according to dataset description
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# 加载数据集（假设从 CSV 文件中加载）
# Load the dataset
# 此处需替换为你的数据路径或变量
# Replace this with the actual data loading in practice
data_path = "/Users/troy/Downloads/pima-indians-diabetes.csv"
df_tian = pd.read_csv(data_path, names=columns, header=0)

# 打印数据结构和基本信息
print("\n列名和类型 Columns and Data Types:")
print(df_tian.dtypes)

# 检查缺失值
print("\n缺失值检查 Missing Values:")
print(df_tian.isnull().sum())

# 数值字段的描述性统计信息
print("\n统计信息 Statistics:")
print(df_tian.describe())

# 类别分布检查（Outcome字段）
print("\n类别数量 Class Distribution:")
print(df_tian["Outcome"].value_counts())

# 第2部分：数据预处理与拆分
# Part 2: Preprocess and prepare the data for ML

# 初始化标准化器
# Initialize standard scaler
transformer_tian = StandardScaler()

# 拆分特征与标签
# Split features and labels
X = df_tian.drop("Outcome", axis=1)
y = df_tian["Outcome"]

# 拆分训练集与测试集
# Split into train and test sets
X_train_tian, X_test_tian, y_train_tian, y_test_tian = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 对特征进行标准化处理
# Fit and transform the scaler
X_train_tian = transformer_tian.fit_transform(X_train_tian)
X_test_tian = transformer_tian.transform(X_test_tian)

# 第3部分：硬投票分类器 Hard Voting Classifier
# Part 3: Hard Voting Classifier

# 定义五个基础分类器
lr_T = LogisticRegression(max_iter=1400)
rf_T = RandomForestClassifier()
svm_T = SVC(probability=True)
dt_T = DecisionTreeClassifier(criterion="entropy", max_depth=42)
et_T = ExtraTreesClassifier()

# 构建硬投票分类器
voting_clf_hard = VotingClassifier(
    estimators=[
        ('LR', lr_T),
        ('RF', rf_T),
        ('SVM', svm_T),
        ('DT', dt_T),
        ('ET', et_T)
    ],
    voting='hard'
)

# 拟合硬投票模型并进行预测
voting_clf_hard.fit(X_train_tian, y_train_tian)
preds_hard = voting_clf_hard.predict(X_test_tian[:3])
print("\n硬投票预测前3个结果 Hard Voting Predictions:", preds_hard)
print("实际值 True Labels:", y_test_tian[:3].values)

# 对所有分类器进行预测并输出结果
for name, clf in voting_clf_hard.named_estimators_.items():
    preds = clf.predict(X_test_tian[:3])
    print(f"\n分类器 {name} 的预测 Predictions by {name}: {preds}")

# 第4部分：软投票分类器 Soft Voting Classifier
# Part 4: Soft Voting Classifier
voting_clf_soft = VotingClassifier(
    estimators=[
        ('LR', lr_T),
        ('RF', rf_T),
        ('SVM', svm_T),
        ('DT', dt_T),
        ('ET', et_T)
    ],
    voting='soft'
)

voting_clf_soft.fit(X_train_tian, y_train_tian)
preds_soft = voting_clf_soft.predict(X_test_tian[:3])
print("\n软投票预测前3个结果 Soft Voting Predictions:", preds_soft)

# 第5部分：Pipeline 和交叉验证
# Part 5: Pipelines and Cross Validation
pipeline1_tian = Pipeline([
    ('scaler', transformer_tian),
    ('et', et_T)
])
pipeline2_tian = Pipeline([
    ('scaler', transformer_tian),
    ('dt', dt_T)
])

# 原始数据未分割进行训练和交叉验证
cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)

scores1 = cross_val_score(pipeline1_tian, X, y, cv=cv_strategy)
scores2 = cross_val_score(pipeline2_tian, X, y, cv=cv_strategy)

print("\nPipeline1 平均准确率 Mean CV Accuracy (ET):", scores1.mean())
print("Pipeline2 平均准确率 Mean CV Accuracy (DT):", scores2.mean())

# 在测试集上评估
for i, (name, pipe) in enumerate([("ExtraTrees", pipeline1_tian), ("DecisionTree", pipeline2_tian)]):
    pipe.fit(X_train_tian, y_train_tian)
    y_pred = pipe.predict(X_test_tian)
    print(f"\n{name} 模型评估 Metrics for {name}:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_tian, y_pred))
    print("Precision:", precision_score(y_test_tian, y_pred))
    print("Recall:", recall_score(y_test_tian, y_pred))
    print("Accuracy:", accuracy_score(y_test_tian, y_pred))

# 第6部分：随机搜索优化 ExtraTreesClassifier
# Part 6: Randomized Search on ExtraTreesClassifier
param_dist = {
    'et__n_estimators': list(range(10, 3001, 20)),
    'et__max_depth': list(range(1, 1001, 2))
}

random_search = RandomizedSearchCV(
    pipeline1_tian,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_tian, y_train_tian)
print("\n最佳参数 Best Parameters:", random_search.best_params_)
print("最佳准确率 Best Accuracy:", random_search.best_score_)

# 在测试集上评估调参后的模型
best_model = random_search.best_estimator_
y_pred_best = best_model.predict(X_test_tian)
print("\n调参后模型在测试集上的表现 Test Performance after Tuning:")
print("Precision:", precision_score(y_test_tian, y_pred_best))
print("Recall:", recall_score(y_test_tian, y_pred_best))
print("Accuracy:", accuracy_score(y_test_tian, y_pred_best))
