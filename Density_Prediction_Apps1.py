import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Page Configuration
st.set_page_config(
    page_title="Density Prediction App",
    page_icon="🧪",
    layout="wide"
)

# 2. App Title and Description
st.title("🧪 Prediksi Nilai Density Non-Linier")
st.markdown("""
Aplikasi ini memprediksi nilai `Density` menggunakan metode **Random Forest Regression**.
Model akan mencari persamaan non-linier (polinomial) untuk setiap domain (`LIM`, `SAP`, `BRK`).
Pastikan file yang diunggah memiliki kolom: `Ni`, `Fe`, `SiO2`, `Al2O3`, `MgO`, `LOI`, `Domain`, dan `Density`.
""")

st.markdown("---")

# 3. File Uploader
uploaded_file = st.file_uploader(
    "Unggah file Excel atau CSV",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("File berhasil diunggah!")
        st.dataframe(df.head())

        # Button to run analysis
        if st.button("Dapatkan Persamaan Density"):
            with st.spinner("Memproses data dan melatih model..."):
                # Define features and target
                numerical_features = ['Ni', 'Fe', 'SiO2', 'Al2O3', 'MgO', 'LOI']
                categorical_features = ['Domain']
                target = 'Density'

                # Validate required columns
                required_cols = numerical_features + categorical_features + [target]
                if not all(col in df.columns for col in required_cols):
                    st.error(f"File tidak memiliki kolom yang dibutuhkan. Pastikan ada kolom: {required_cols}")
                    st.stop()
                
                # Handle missing values
                df.dropna(subset=required_cols, inplace=True)
                if df.empty:
                    st.error("Data setelah dibersihkan tidak memiliki baris yang cukup. Periksa kembali data Anda.")
                    st.stop()
                
                # Set up preprocessor for categorical features
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', OneHotEncoder(), categorical_features)],
                    remainder='passthrough'
                )

                # Looping 10 times to find the best Random Forest model
                best_model = None
                best_r2_score = -np.inf

                for i in range(10):
                    X = df[numerical_features + categorical_features]
                    y = df[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
                    
                    pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
                    ])
                    
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    
                    if r2 > best_r2_score:
                        best_r2_score = r2
                        best_model = pipeline

                st.markdown("### Hasil Analisis")
                
                # Generate polynomial regression equations for each domain
                domain_results = []
                for domain in df['Domain'].unique():
                    domain_df = df[df['Domain'] == domain].copy()
                    domain_df.dropna(subset=numerical_features + [target], inplace=True)
                    
                    if len(domain_df) > 1:
                        # Polynomial Regression
                        poly_features = PolynomialFeatures(degree=2, include_bias=False)
                        X_poly_domain = poly_features.fit_transform(domain_df[numerical_features])
                        y_poly_domain = domain_df[target]
                        
                        poly_reg_model = LinearRegression()
                        poly_reg_model.fit(X_poly_domain, y_poly_domain)
                        
                        coefficients = poly_reg_model.coef_
                        intercept = poly_reg_model.intercept_
                        
                        # Generate the equation string
                        feature_names = poly_features.get_feature_names_out(numerical_features)
                        equation = "Density = "
                        for coeff, feature in zip(coefficients, feature_names):
                            equation += f"({coeff:.4f} * {feature}) + "
                        equation += f"({intercept:.4f})"

                        # Predict and evaluate
                        y_pred_poly = poly_reg_model.predict(X_poly_domain)
                        r2_domain = r2_score(y_poly_domain, y_pred_poly)
                        rmse_domain = np.sqrt(mean_squared_error(y_poly_domain, y_pred_poly))
                        
                        domain_results.append({
                            'Domain': domain,
                            'Persamaan': equation,
                            'R2 Score': f"{r2_domain:.4f}",
                            'RMSE': f"{rmse_domain:.4f}"
                        })
                    else:
                        st.warning(f"Tidak ada cukup data untuk domain '{domain}' untuk membuat persamaan.")
                
                # Display results in a table
                if domain_results:
                    results_df = pd.DataFrame(domain_results)
                    st.table(results_df)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")