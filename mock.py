import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# 📌 1. 读取数据
file_path = "//Users/troy/Downloads/mockdiabetes.csv"
df = pd.read_csv(file_path)

# 查看数据的前几行
print("数据预览：")
print(df.head())

# 📌 2. 数据探索（绘制散点图和热力图）

# 散点图矩阵
sns.pairplot(df, diag_kind="kde")
plt.title("散点图矩阵")
plt.show()

# 计算相关性矩阵
corr_matrix = df.corr()

# 热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("特征相关性热力图")
plt.show()

# 📌 3. 计算特征相关性（找到相关性最高的特征）
print("特征相关性矩阵：")
print(corr_matrix)

# 📌 4. 添加新特征（创建类别变量）
# 例如：如果 BMI > 30 设为 1（肥胖），否则设为 0（正常）
df['Obese'] = np.where(df['BMI'] > 30, 1, 0)

# 查看新增特征
print("添加的新特征预览：")
print(df[['BMI', 'Obese']].head())

# 📌 5. 数据预处理（缺失值检查 & 处理）
print("缺失值统计：")
print(df.isnull().sum())

# 填充缺失值（如果有）
df.fillna(df.median(), inplace=True)

# 📌 6. 划分训练集和测试集
X = df.drop(columns=['Outcome'])  # 预测变量
y = df['Outcome']  # 目标变量（是否患有糖尿病）

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 📌 7. 进行标准化，并创建 SVM Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 标准化
    ('classifier', SVC(kernel='rbf'))  # SVM 分类器
])

# 📌 8. 定义 Grid Search 进行超参数优化
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],  # 正则化参数
    'classifier__gamma': [0.01, 0.1, 1, 10],  # 核函数参数
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 📌 9. 获取最佳参数
print(f"最佳参数：{grid_search.best_params_}")

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 📌 10. 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM 模型的最佳准确率：{accuracy:.4f}")

# 📌 11. 生成混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 📌 12. 绘制混淆矩阵
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("SVM 混淆矩阵")
plt.show()