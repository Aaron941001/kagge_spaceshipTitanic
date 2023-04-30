import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 讀取數據
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# EDA
print(train_df.head())
print(train_df.info())
print(train_df.describe())

# 檢查缺失值
print(train_df.isnull().sum())

# 繪製年齡分布
plt.figure()
sns.histplot(train_df["Age"], kde=True)
plt.title("Age Distribution")
plt.show()

# 預處理
# 合併訓練集和測試集以便統一處理
all_data = pd.concat([train_df.drop("Transported", axis=1), test_df], ignore_index=True)

# 填充缺失值
all_data["Age"].fillna(all_data["Age"].median(), inplace=True)

# 類別特徵編碼
all_data = pd.get_dummies(all_data, columns=["HomePlanet", "Cabin", "Destination"])

scaler = StandardScaler()
numerical_features = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
all_data[numerical_features] = scaler.fit_transform(all_data[numerical_features])

# 分離訓練集和測試集
train_df_processed = all_data[:len(train_df)]
test_df_processed = all_data[len(train_df):]

# 檢查處理後的數據
print(train_df_processed.head())


# Age 和 Transported 之間的關係
plt.figure()
sns.histplot(train_df[train_df["Transported"] == True]["Age"], kde=True, color="blue", label="Transported")
sns.histplot(train_df[train_df["Transported"] == False]["Age"], kde=True, color="red", label="Not Transported")
plt.title("Age Distribution by Transported")
plt.legend()
plt.show()

# CryoSleep 和 Transported 之間的關係
plt.figure()
sns.countplot(data=train_df, x="CryoSleep", hue="Transported")
plt.title("CryoSleep vs Transported")
plt.show()

# VIP 和 Transported 之間的關係
plt.figure()
sns.countplot(data=train_df, x="VIP", hue="Transported")
plt.title("VIP vs Transported")
plt.show()

# HomePlanet 和 Transported 之間的關係
plt.figure(figsize=(10, 5))
sns.countplot(data=train_df, x="HomePlanet", hue="Transported")
plt.title("HomePlanet vs Transported")
plt.xticks(rotation=45)
plt.show()

# Destination 和 Transported 之間的關係
plt.figure(figsize=(10, 5))
sns.countplot(data=train_df, x="Destination", hue="Transported")
plt.title("Destination vs Transported")
plt.xticks(rotation=45)
plt.show()

# RoomService, FoodCourt, ShoppingMall, Spa, VRDeck 與 Transported 之間的關係
luxury_amenities = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

for amenity in luxury_amenities:
    plt.figure()
    sns.kdeplot(data=train_df[train_df["Transported"] == True], x=amenity, color="blue", label="Transported", common_norm=False)
    sns.kdeplot(data=train_df[train_df["Transported"] == False], x=amenity, color="red", label="Not Transported", common_norm=False)
    plt.title(f"{amenity} Distribution by Transported")
    plt.legend()
    plt.show()

# 相關矩陣熱力圖
plt.figure(figsize=(10, 8))
corr_matrix = train_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")
plt.show()

# 年齡和VIP的關係
plt.figure()
sns.boxplot(data=train_df, x="VIP", y="Age")
plt.title("Age Distribution by VIP")
plt.show()

# 目的地和各項消費的關係
for amenity in luxury_amenities:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=train_df, x="Destination", y=amenity)
    plt.title(f"{amenity} Distribution by Destination")
    plt.xticks(rotation=45)
    plt.show()

# 年齡與 CryoSleep 之間的關係
plt.figure()
sns.boxplot(data=train_df, x="CryoSleep", y="Age")
plt.title("Age Distribution by CryoSleep")
plt.show()

# 年齡與 HomePlanet 之間的關係
plt.figure(figsize=(10, 5))
sns.boxplot(data=train_df, x="HomePlanet", y="Age")
plt.title("Age Distribution by HomePlanet")
plt.xticks(rotation=45)
plt.show()