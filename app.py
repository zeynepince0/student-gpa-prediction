import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# 1. Sayfa Ayarları
st.set_page_config(
    page_title="GPA Prediction AI",
    layout="wide"
)

# Koyu Tema ve Stil Özelleştirmeleri
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3, h4, h5, h6, p, label {
    color: white;
}
.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 15px;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# 2. Model ve Dosyaları Yükleme Fonksiyonu
@st.cache_resource
def load_files():
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_names.pkl")
    metrics = joblib.load("model_metrics.pkl")
    
    models = {
        "LinearRegression": joblib.load("model_linearregression.pkl"),
        "RandomForest": joblib.load("model_randomforest.pkl"),
        "XGBoost": joblib.load("model_xgboost.pkl"),
        "CatBoost": joblib.load("model_catboost.pkl"),
        "LightGBM": joblib.load("model_lightgbm.pkl")
    }
    return scaler, features, metrics, models

# Dosyaları yükle
scaler, features, metrics, models = load_files()

# 3. Slider Sınırları ve Varsayılan Değerler (Feature Config)
feature_config = {
    "study_hours": (0, 12, 5),
    "attendance": (0, 100, 75),
    "assignment_completion": (0, 100, 70),
    "midterm_score": (0, 100, 60),
    "final_score": (0, 100, 65),
    "project_score": (0, 100, 65),
    "backlogs": (0, 10, 2),
    "sleep_hours": (0, 12, 7),
    "stress": (0, 10, 4),
    "anxiety": (0, 100, 50),
    "depression": (0, 100, 45),
    "motivation": (0, 10, 6),
    "concentration": (0, 10, 6),
    "time_management": (0, 10, 6),
    "self_discipline": (0, 10, 6),
    "social_media_hours": (0, 10, 3),
    "gaming_hours": (0, 10, 2),
    "netflix_hours": (0, 10, 2),
    "screen_time": (0, 15, 6),
    "physical_activity": (0, 10, 4),
    "junk_food_frequency": (0, 10, 3),
    "caffeine_mg": (0, 400, 180),
    "late_night_frequency": (0, 10, 2),
    "procrastination_score": (0, 10, 5),
    "family_income": (0, 70000, 25000),
    "parental_education_level": (0, 7, 3),
    "internet_quality": (0, 10, 5),
    "library_visits": (0, 5, 2),
    "online_courses_completed": (0, 5, 1),
    "part_time_hours": (0, 10, 3),
    "peer_study_group": (0, 1, 1),
    "relationship_status": (0, 1, 0),
    "hostel_student": (0, 1, 0),
    "extracurricular_hours": (0, 10, 3),
    "phone_unlocks_per_day": (0, 120, 50),
    "previous_gpa": (0.0, 4.0, 2.5),
    "class_participation": (0, 10, 6),
    "weekly_study_sessions": (0, 10, 5),
    "group_study_hours": (0, 10, 3),
    "financial_stress": (0, 10, 5)
}

# 4. Arayüz Başlığı ve Metrikler
st.title("GPA Prediction AI System")
st.subheader("📊 Model R2 Skorları (Doğruluk)")

metric_cols = st.columns(len(metrics))
for i, (name, vals) in enumerate(metrics.items()):
    metric_cols[i].metric(
        label=name,
        value=f"%{vals['Accuracy']}"
    )

st.divider()

# Ana Tahmini Yapacak Model Seçimi
selected_model = st.selectbox(
    "🤖 Tahmin Yapacak Modeli Seçin",
    list(models.keys())
)

st.subheader("📝 Öğrenci Bilgilerini Giriniz")

# 5. Kullanıcı Girdilerinin Dinamik Oluşturulması
user_input = {}
cols = st.columns(4)

for i, feature in enumerate(features):
    with cols[i % 4]:
        display_name = feature.replace("_", " ").title()
        min_val, max_val, default_val = feature_config.get(
            feature,
            (0, 100, 50)
        )

        if feature in ["peer_study_group", "relationship_status", "hostel_student"]:
            val = st.selectbox(
                display_name,
                ["No", "Yes"],
                key=f"select_{feature}"
            )
            user_input[feature] = 1 if val == "Yes" else 0
            
        elif feature == "previous_gpa":
            user_input[feature] = st.slider(
                display_name,
                float(min_val),
                float(max_val),
                float(default_val),
                0.1,
                key=f"slider_{feature}"
            )
        else:
            user_input[feature] = st.slider(
                display_name,
                float(min_val),
                float(max_val),
                float(default_val),
                key=f"slider_{feature}"
            )

st.divider()

# 6. Hesaplama ve Tahmin Bölümü
if st.button("GPA HESAPLA", use_container_width=True):
    
    input_df = pd.DataFrame([user_input])
    input_df = input_df[features]

    all_predictions = {}
    
    for name, mdl in models.items():
        if name == "LinearRegression":
            scaled_input = scaler.transform(input_df)
            pred = mdl.predict(scaled_input)[0]
        else:
            pred = mdl.predict(input_df)[0]
            
        pred = np.clip(pred, 0, 4)
        all_predictions[name] = round(float(pred), 2)

    prediction = all_predictions[selected_model]
    model = models[selected_model]
  
    # Görsel Sonuç Panelleri (Sol: Özet metin, Sağ: Feature Importance)
    left, right = st.columns([1, 2])

    with left:
        st.success(f"### Tahmini GPA: {prediction:.2f} / 4.00")
        st.write(f"""
        - Kullanılan Model: **{selected_model}**
        - Model R² Skoru: **%{metrics[selected_model]['Accuracy']}**
        - Hedef Ölçek: **4.00 Maks**
        """)

    with right:
        st.subheader("🎯 En Etkili Faktörler (Feature Importance)")
        
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_)
        else:
            importance = np.zeros(len(features))

        feat_imp = pd.Series(importance, index=features).sort_values().tail(10)

        fig, ax = plt.subplots(figsize=(9, 5))
        feat_imp.plot(kind="barh", ax=ax, color="#1f77b4")
        ax.set_xlabel("Etki Katsayısı")
        ax.set_ylabel("Öznitelikler (Features)")
        st.pyplot(fig)

    st.divider()
    
    # 7. Model Tahmin Karşılaştırma Tablosu ve Bar Grafiği
    st.subheader("📈 Model Tahmin Karşılaştırması")

    comparison_df = pd.DataFrame({
        "Model": list(all_predictions.keys()),
        "Tahmini GPA": list(all_predictions.values())
    })

    st.dataframe(comparison_df, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    bars = ax2.bar(
        comparison_df["Model"], 
        comparison_df["Tahmini GPA"], 
        color=['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
    )
    ax2.set_ylim(0, 4.3)
    ax2.set_ylabel("GPA")
    ax2.set_title("Farklı Algoritmaların GPA Öngörüleri")

    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    st.pyplot(fig2)