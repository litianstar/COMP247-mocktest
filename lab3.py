import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import joblib

# 1. 加载数据集 / Load Dataset
data_tian = pd.read_csv('/Users/troy/Downloads/student-por.csv', sep=';')
print("数据集前5行: \n", data_tian.head())

# 2. 数据探索 / Data Exploration
print("列名及数据类型: \n", data_tian.dtypes)
print("缺失值统计: \n", data_tian.isnull().sum())
print("数值字段统计: \n", data_tian.describe())
print("类别字段统计: \n", data_tian.select_dtypes(include=['object']).nunique())

# 3. 创建目标变量 / Create Target Variable
data_tian['pass_tian'] = np.where(data_tian[['G1', 'G2', 'G3']].sum(axis=1) >= 35, 1, 0)
data_tian.drop(columns=['G1', 'G2', 'G3'], inplace=True)

# 4. 拆分特征和目标 / Split Features and Target
features_tian = data_tian.drop(columns=['pass_tian'])
target_variable_tian = data_tian['pass_tian']

# 5. 统计类别分布 / Print Class Distribution
print("类别分布: \n", target_variable_tian.value_counts())

# 6. 识别数值和类别特征 / Identify Numeric and Categorical Features
numeric_features_tian = features_tian.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features_tian = features_tian.select_dtypes(include=['object']).columns.tolist()

# 7. 数据转换器（类别特征One-Hot编码）/ Column Transformer
transformer_tian = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features_tian)],
    remainder='passthrough')

# 8. 决策树模型 / Decision Tree Model
clf_tian = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)

# 9. 构建数据流水线 / Build Pipeline
pipeline_tian = Pipeline([
    ('transformer', transformer_tian),
    ('classifier', clf_tian)
])

# 10. 划分训练集和测试集 / Split Train-Test Data
X_train_tian, X_test_tian, y_train_tian, y_test_tian = train_test_split(
    features_tian, target_variable_tian, test_size=0.2, random_state=42)

# 11. 训练模型 / Train Model
pipeline_tian.fit(X_train_tian, y_train_tian)

# 12. 交叉验证 / Cross Validation
cv_scores_tian = cross_val_score(pipeline_tian, X_train_tian, y_train_tian, cv=10)
print("10次交叉验证准确率: ", cv_scores_tian)
print("交叉验证平均准确率: ", cv_scores_tian.mean())

# 13. 决策树可视化 / Visualize Decision Tree
# 获取 One-Hot 编码后的特征名称
feature_names_tian = pipeline_tian.named_steps['transformer'].get_feature_names_out()

# 画出决策树
plt.figure(figsize=(20,10))
plot_tree(clf_tian, filled=True, feature_names=feature_names_tian, class_names=['Fail', 'Pass'])
plt.savefig("decision_tree_tian.png")
plt.show()


# 14. 计算模型评估指标 / Model Evaluation
train_acc_tian = accuracy_score(y_train_tian, pipeline_tian.predict(X_train_tian))
test_acc_tian = accuracy_score(y_test_tian, pipeline_tian.predict(X_test_tian))
precision_tian = precision_score(y_test_tian, pipeline_tian.predict(X_test_tian))
recall_tian = recall_score(y_test_tian, pipeline_tian.predict(X_test_tian))
conf_matrix_tian = confusion_matrix(y_test_tian, pipeline_tian.predict(X_test_tian))

print("训练集准确率: ", train_acc_tian)
print("测试集准确率: ", test_acc_tian)
print("精确率: ", precision_tian)
print("召回率: ", recall_tian)
print("混淆矩阵: \n", conf_matrix_tian)

# 15. 超参数优化 / Hyperparameter Tuning
param_grid_tian = {
    'classifier__min_samples_split': range(10, 300, 20),
    'classifier__max_depth': range(1, 30, 2),
    'classifier__min_samples_leaf': range(1, 15, 3)
}

random_search_tian = RandomizedSearchCV(
    pipeline_tian, param_distributions=param_grid_tian, n_iter=7, cv=5, scoring='accuracy', random_state=42, verbose=3
)

random_search_tian.fit(X_train_tian, y_train_tian)
print("最佳参数: ", random_search_tian.best_params_)
print("优化后模型得分: ", random_search_tian.best_score_)

# 16. 使用最佳模型评估测试集 / Evaluate Test Data with Best Model
best_model_tian = random_search_tian.best_estimator_
final_acc_tian = accuracy_score(y_test_tian, best_model_tian.predict(X_test_tian))
final_precision_tian = precision_score(y_test_tian, best_model_tian.predict(X_test_tian))
final_recall_tian = recall_score(y_test_tian, best_model_tian.predict(X_test_tian))
final_conf_matrix_tian = confusion_matrix(y_test_tian, best_model_tian.predict(X_test_tian))

print("优化后测试集准确率: ", final_acc_tian)
print("优化后精确率: ", final_precision_tian)
print("优化后召回率: ", final_recall_tian)
print("优化后混淆矩阵: \n", final_conf_matrix_tian)

# 17. 保存模型 / Save Model
joblib.dump(best_model_tian, "best_decision_tree_tian.pkl")
joblib.dump(pipeline_tian, "pipeline_tian.pkl")
