import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


print("⚡ Veri yükleniyor...")
df = pd.read_csv("college_students_habits_1M.csv")


print("🧹 Veri temizleniyor...")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)


print("📊 Veri optimize ediliyor (Örneklem alınıyor)...")
df = df.sample(
    200000,
    random_state=42
)


print("📐 Balanced GPA oluşturuluyor...")
academic_score = (
    df["final_score"] * 0.25 +
    df["midterm_score"] * 0.20 +
    df["assignment_completion"] * 0.15 +
    df["attendance"] * 0.10 +
    df["project_score"] * 0.10 +
    df["class_participation"] * 0.08 +
    df["weekly_study_sessions"] * 1.5 +
    df["group_study_hours"] * 1.0
)

cognitive_score = (
    df["motivation"] * 1.5 +
    df["concentration"] * 1.5 +
    df["time_management"] * 1.2 +
    df["self_discipline"] * 1.2 +
    df["previous_gpa"] * 4
)

lifestyle_score = (
    df["sleep_hours"] * 2 +
    df["physical_activity"] * 2 +
    df["library_visits"] * 1 +
    df["online_courses_completed"] * 1
)

negative_score = (
    df["stress"] * 2 +
    df["anxiety"] * 1.5 +
    df["depression"] * 1.5 +
    df["procrastination_score"] * 2 +
    df["backlogs"] * 10 +
    df["social_media_hours"] * 4 +
    df["gaming_hours"] * 4 +
    df["netflix_hours"] * 3 +
    df["screen_time"] * 2 +
    df["late_night_frequency"] * 5 +
    df["junk_food_frequency"] * 2 +
    df["financial_stress"] * 2
)

bonus_score = (
    df["internet_quality"] * 1 +
    df["parental_education_level"] * 1 +
    df["peer_study_group"] * 2
)

raw_score = (
    academic_score +
    cognitive_score +
    lifestyle_score +
    bonus_score -
    negative_score
)

# Min-Max normalize ile gerçek 0-4 GPA dağılımı
df["custom_gpa"] = (
    (raw_score - raw_score.min()) /
    (raw_score.max() - raw_score.min())
) * 4

df["custom_gpa"] = df["custom_gpa"].clip(0, 4)

print("\n--- Custom GPA Dağılım Özeti ---")
print(df["custom_gpa"].describe())
print("---------------------------------\n")


# Sütunları ayır ve ALFABETİK olarak sırala (Girdi karmaşasını çözer)
X = df.drop(
    ["gpa", "custom_gpa", "performance_level"],
    axis=1
)
X = X.reindex(sorted(X.columns), axis=1) # Sütun isimleri alfabetik sıraya dizildi

y = df["custom_gpa"]

# Gelecekte app.py'ın tam olarak bu sırayı kullanması için listeyi kaydediyoruz
feature_names = X.columns.tolist()

X = X.astype(np.float32)
y = y.astype(np.float32)


print("✂️ Train/Test ayrılıyor...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


print("⚖️ Scaling uygulanıyor...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Ortak bağımlılıkları disket gibi pkl dosyalarına kaydetme
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")


print("🤖 Modeller güçlü hiperparametrelerle kuruluyor...")
# Modellerin öğrenme kapasiteleri (Ağaç sayıları ve derinlikleri) artırıldı.
models = {

    "LinearRegression": LinearRegression(),

    "RandomForest": RandomForestRegressor(
        n_estimators=100,      # 20'den 100'e çıkarıldı (Daha kararlı ağaç yapısı)
        max_depth=15,          # 10'dan 15'e çıkarıldı (Daha detaylı örüntü yakalama)
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,      # 100'den 200'e çıkarıldı
        learning_rate=0.08,    # Adım boyutu optimize edildi
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    ),

    "CatBoost": CatBoostRegressor(
        iterations=300,        # 120'den 300'e çıkarıldı (Erken durmayı ve eksik öğrenmeyi engeller)
        learning_rate=0.08,
        depth=7,
        loss_function='RMSE',
        random_state=42,
        verbose=50             # Her 50 adımda bir ekrana çıktı verir
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=200,      # 100'den 200'e çıkarıldı
        learning_rate=0.08,
        max_depth=7,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
}

results = {}
model_metrics = {}

print("\n🚀 MODEL EĞİTİMİ BAŞLIYOR...\n")

for name, model in models.items():
    print(f"-> {name} eğitiliyor...")
    
    if name == "LinearRegression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        # Ağaç modelleri ham (ama alfabetik sıralanmış) X_train ile eğitiliyor
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    # Tahminleri mantıklı aralığa (0-4) sıkıştır
    preds = np.clip(preds, 0, 4)

    # Metrikleri Hesaplama
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results[name] = {
        "preds": preds,
        "errors": y_test - preds,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "model": model
    }

    model_metrics[name] = {
        "Accuracy": round(r2 * 100, 2),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }

    # Model dosyasını kaydetme (Küçük harfe çevirerek app.py ile eşleştiriyoruz)
    filename = f"model_{name.lower()}.pkl"
    joblib.dump(model, filename)
    print(f"   Sonuç: {model_metrics[name]}")

# Karşılaştırma metriklerini kaydetme
joblib.dump(model_metrics, "model_metrics.pkl")

comparison_df = pd.DataFrame(model_metrics).T
comparison_df.to_csv("model_comparison.csv")

print("\n📊 MODEL KARŞILAŞTIRMA TABLOSU")
print(comparison_df)


print("\n🎨 Grafikler oluşturuluyor...")

# 1. Feature Importance Grafiği
plt.figure(figsize=(16, 28))
top_n = 10
for i, (name, res) in enumerate(results.items()):
    plt.subplot(5, 1, i + 1)
    model = res["model"]
    
    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    elif hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        continue

    indices = np.argsort(imp)[-top_n:]
    plt.barh(range(top_n), imp[indices], color='teal')
    plt.yticks(range(top_n), [feature_names[j] for j in indices])
    plt.title(f"{name} Feature Importance")

plt.tight_layout()
plt.savefig("1_feature_importance.png", dpi=250)
plt.close()

# 2. Model Accuracy Grafiği
plt.figure(figsize=(10, 5))
sns.barplot(x=comparison_df.index, y=comparison_df["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.savefig("2_model_accuracy.png", dpi=250)
plt.close()

# 3. Scatter Plot Grafiği
plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'orange', 'purple', 'red']
for i, (name, res) in enumerate(results.items()):
    plt.scatter(
        y_test,
        res["preds"],
        alpha=0.08,
        s=2,
        color=colors[i],
        label=f"{name} ({res['r2']:.3f})"
    )
plt.plot([0, 4], [0, 4], 'k--')
plt.xlabel("Gerçek GPA")
plt.ylabel("Tahmin GPA")
plt.title("Gerçek vs Tahmin GPA")
plt.legend()
plt.savefig("3_scatter_models.png", dpi=250)
plt.close()

print("\n✅ TÜM EĞİTİM BAŞARIYLA TAMAMLANDI!")
print("📁 Modeller ve 'feature_names.pkl' dizine kaydedildi.")
print("🖼️ Grafik dosyaları (1, 2, 3) oluşturuldu.")