# model_monitoring_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Monitoreo de Data Drift",
    page_icon="📊",
    layout="wide"
)

# Título de la aplicación
st.title("📊 Sistema de Monitoreo de Data Drift")
st.markdown("---")

# Configuración inicial
try:
    from carga_datos import cargarDatosLimpios
    from ft_engineering import ft_engineering, X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, preprocessor
    df = cargarDatosLimpios()
    
    with st.sidebar:
        st.success("✅ Datos cargados exitosamente")
        st.info(f"Dataset: {df.shape[0]} filas × {df.shape[1]} columnas")
except Exception as e:
    st.error(f"❌ Error al cargar los datos: {str(e)}")
    # Crear datos de ejemplo si falla la carga
    df = None
    # Inicializar variables necesarias
    X_train = None
    X_test = None
    y_train = None
    y_test = None

# Definir la función determine_severity al principio del archivo
def determine_severity(metrics):
    """Determina la severidad del drift para una feature"""
    psi = metrics.get('psi', 0)
    diff_pct = metrics.get('mean_diff_pct', 0)
    ks_stat = metrics.get('ks_statistic', 0)
    
    if psi > 0.25 or diff_pct > 30 or ks_stat > 0.3:
        return "🔴 Crítico"
    elif psi > 0.2 or diff_pct > 20 or ks_stat > 0.2:
        return "🟡 Alto"
    elif psi > 0.1 or diff_pct > 10 or ks_stat > 0.1:
        return "🟠 Moderado"
    else:
        return "🟢 Bajo"

# Sidebar con controles
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Controles de muestreo
    sample_size = st.slider(
        "Tamaño de muestra",
        min_value=100,
        max_value=1000,
        value=300,
        step=50,
        help="Número de registros para análisis"
    )
    
    drift_factor = st.slider(
        "Factor de drift simulado",
        min_value=0.0,
        max_value=0.5,
        value=0.15,
        step=0.05,
        help="Simula cambios en los datos (0 = sin cambios)"
    )
    
    # Métricas a calcular
    st.subheader("📈 Métricas a calcular")
    calc_psi = st.checkbox("PSI (Population Stability Index)", value=True)
    calc_ks = st.checkbox("KS Test (Kolmogorov-Smirnov)", value=True)
    calc_js = st.checkbox("JS Divergence", value=True)
    calc_chi2 = st.checkbox("Chi-cuadrado", value=True)
    
    # Umbrales
    st.subheader("🚨 Umbrales de alerta")
    psi_threshold = st.number_input("Umbral PSI", value=0.2, step=0.05, min_value=0.0)
    ks_threshold = st.number_input("Umbral KS", value=0.3, step=0.05, min_value=0.0)
    diff_threshold = st.number_input("Umbral diferencia %", value=20.0, step=5.0, min_value=0.0)
    
    # Botón de ejecución
    if st.button("🚀 Ejecutar Monitoreo", type="primary", use_container_width=True):
        st.session_state.run_monitoring = True
    else:
        if 'run_monitoring' not in st.session_state:
            st.session_state.run_monitoring = False

# Funciones para métricas de Data Drift
@st.cache_data
def calculate_metrics(ref_data, new_data, feature_name):
    """Calcula todas las métricas para una feature"""
    metrics = {}
    
    try:
        # Métricas básicas
        ref_mean = ref_data.mean()
        new_mean = new_data.mean()
        mean_diff_pct = abs((new_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0
        
        metrics['ref_mean'] = ref_mean
        metrics['new_mean'] = new_mean
        metrics['mean_diff_pct'] = mean_diff_pct
        
        # PSI
        if calc_psi:
            psi = calculate_psi(ref_data, new_data)
            metrics['psi'] = psi['psi']
            metrics['psi_interpretation'] = psi['interpretation']
        
        # KS Test
        if calc_ks:
            ks = calculate_ks_statistic(ref_data, new_data)
            metrics['ks_statistic'] = ks['ks_statistic']
            metrics['ks_p_value'] = ks['p_value']
            metrics['ks_significant'] = ks['significant']
        
        # JS Divergence
        if calc_js:
            js = calculate_js_divergence(ref_data, new_data)
            metrics['js_divergence'] = js['js_divergence']
        
        # Chi-cuadrado
        if calc_chi2:
            chi2 = calculate_chi_square(ref_data, new_data)
            metrics['chi_square'] = chi2['chi_square']
            metrics['chi_p_value'] = chi2['p_value']
            metrics['chi_significant'] = chi2['significant']
        
        # Determinar si hay drift
        has_drift = (
            (calc_psi and metrics.get('psi', 0) > psi_threshold) or
            (calc_ks and metrics.get('ks_significant', False)) or
            metrics['mean_diff_pct'] > diff_threshold
        )
        
        metrics['has_drift'] = has_drift
        
    except Exception as e:
        st.warning(f"Error calculando métricas para {feature_name}: {e}")
        metrics = {'error': str(e), 'has_drift': False}
    
    return metrics

def calculate_ks_statistic(ref_data, new_data):
    """Calcula el estadístico Kolmogorov-Smirnov"""
    try:
        ref_clean = ref_data.dropna()
        new_clean = new_data.dropna()
        
        if len(ref_clean) < 2 or len(new_clean) < 2:
            return {'ks_statistic': 0, 'p_value': 1, 'significant': False}
        
        ks_stat, p_value = stats.ks_2samp(ref_clean, new_clean)
        return {'ks_statistic': ks_stat, 'p_value': p_value, 'significant': p_value < 0.05}
    except Exception as e:
        return {'ks_statistic': 0, 'p_value': 1, 'significant': False}

def calculate_psi(ref_data, new_data, bins=10):
    """Calcula el Population Stability Index"""
    try:
        ref_clean = ref_data.dropna()
        new_clean = new_data.dropna()
        
        if len(ref_clean) < 2 or len(new_clean) < 2:
            return {'psi': 0, 'interpretation': "Datos insuficientes"}
        
        min_val = min(ref_clean.min(), new_clean.min())
        max_val = max(ref_clean.max(), new_clean.max())
        
        if min_val == max_val:
            return {'psi': 0, 'interpretation': "Sin variación"}
            
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        ref_dist, _ = np.histogram(ref_clean, bins=bin_edges, density=True)
        new_dist, _ = np.histogram(new_clean, bins=bin_edges, density=True)
        
        ref_dist = ref_dist + 1e-10
        new_dist = new_dist + 1e-10
        
        psi_values = (new_dist - ref_dist) * np.log(new_dist / ref_dist)
        psi_total = np.sum(psi_values)
        
        if psi_total < 0.1:
            interpretation = "Sin cambio significativo"
        elif psi_total < 0.2:
            interpretation = "Cambio moderado"
        else:
            interpretation = "Cambio significativo"
            
        return {'psi': psi_total, 'interpretation': interpretation}
    except Exception as e:
        return {'psi': 0, 'interpretation': "Error en cálculo"}

def calculate_js_divergence(ref_data, new_data, bins=20):
    """Calcula la Jensen-Shannon Divergence"""
    try:
        ref_clean = ref_data.dropna()
        new_clean = new_data.dropna()
        
        if len(ref_clean) < 2 or len(new_clean) < 2:
            return {'js_divergence': 0}
        
        min_val = min(ref_clean.min(), new_clean.min())
        max_val = max(ref_clean.max(), new_clean.max())
        
        if min_val == max_val:
            return {'js_divergence': 0}
            
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        ref_dist, _ = np.histogram(ref_clean, bins=bin_edges, density=True)
        new_dist, _ = np.histogram(new_clean, bins=bin_edges, density=True)
        
        js_distance = jensenshannon(ref_dist, new_dist)
        
        return {'js_divergence': js_distance}
    except Exception as e:
        return {'js_divergence': 0}

def calculate_chi_square(ref_data, new_data, bins=10):
    """Calcula Chi-cuadrado para variables categóricas/numéricas discretizadas"""
    try:
        ref_clean = ref_data.dropna()
        new_clean = new_data.dropna()
        
        if len(ref_clean) < 2 or len(new_clean) < 2:
            return {'chi_square': 0, 'p_value': 1, 'significant': False}
        
        if pd.api.types.is_numeric_dtype(ref_data):
            min_val = min(ref_clean.min(), new_clean.min())
            max_val = max(ref_clean.max(), new_clean.max())
            
            if min_val == max_val:
                return {'chi_square': 0, 'p_value': 1, 'significant': False}
                
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
            new_counts, _ = np.histogram(new_clean, bins=bin_edges)
        else:
            all_categories = pd.unique(np.concatenate([ref_clean.unique(), new_clean.unique()]))
            ref_counts = ref_clean.value_counts().reindex(all_categories).fillna(0)
            new_counts = new_clean.value_counts().reindex(all_categories).fillna(0)
        
        # Asegurar que las matrices tengan la misma forma
        min_len = min(len(ref_counts), len(new_counts))
        ref_counts = ref_counts[:min_len]
        new_counts = new_counts[:min_len]
        
        if len(ref_counts) < 2 or len(new_counts) < 2:
            return {'chi_square': 0, 'p_value': 1, 'significant': False}
            
        contingency_table = np.array([ref_counts, new_counts])
        
        # Verificar que la tabla no sea todo ceros
        if np.sum(contingency_table) == 0:
            return {'chi_square': 0, 'p_value': 1, 'significant': False}
        
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        
        return {'chi_square': chi2, 'p_value': p_value, 'significant': p_value < 0.05}
    except Exception as e:
        return {'chi_square': 0, 'p_value': 1, 'significant': False}

def simulate_production_data(df, n_samples=200, drift_factor=0.0):
    """Simula datos de producción"""
    if df is None:
        return None, None
    
    base_data = df.copy()
    n_samples = min(n_samples, len(base_data))
    
    simulated_data = base_data.sample(n=n_samples, replace=False, random_state=int(time.time())).copy()
    
    if drift_factor > 0:
        numeric_cols = simulated_data.select_dtypes(include=[np.number]).columns
        # Excluir la columna target si existe
        numeric_cols = [col for col in numeric_cols if col != 'Pago_atiempo']
        
        if len(numeric_cols) > 0:
            n_features_to_drift = max(1, int(len(numeric_cols) * 0.3))
            cols_to_drift = np.random.choice(numeric_cols, n_features_to_drift, replace=False)
            
            for col in cols_to_drift:
                current_std = simulated_data[col].std()
                if current_std > 0:
                    noise = np.random.normal(0, drift_factor * current_std, len(simulated_data))
                    simulated_data[col] = simulated_data[col] * (1 + drift_factor * 0.3) + noise
    
    return simulated_data.drop('Pago_atiempo', axis=1), simulated_data['Pago_atiempo']

# Función para crear datos de ejemplo si no hay datos reales
def create_sample_data():
    """Crea datos de ejemplo para demostración"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'tipo_credito': np.random.choice([4, 7, 9], n_samples),
        'capital_prestado': np.random.lognormal(14, 1, n_samples),
        'plazo_meses': np.random.randint(2, 90, n_samples),
        'edad_cliente': np.random.randint(19, 69, n_samples),
        'salario_cliente': np.random.lognormal(14, 1, n_samples),
        'total_otros_prestamos': np.random.lognormal(12, 1, n_samples),
        'puntaje_datacredito': np.random.normal(700, 100, n_samples),
        'Pago_atiempo': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    }
    
    return pd.DataFrame(data)

# Contenido principal
if df is None:
    st.warning("⚠️ No se pudieron cargar los datos reales. Usando datos de demostración.")
    df = create_sample_data()
    
    # Crear datos de entrenamiento y prueba de ejemplo
    from sklearn.model_selection import train_test_split
    X = df.drop('Pago_atiempo', axis=1)
    y = df['Pago_atiempo']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

if st.session_state.get('run_monitoring', False):
    with st.spinner("📊 Generando datos y calculando métricas..."):
        # Generar datos
        new_X, new_y = simulate_production_data(df, n_samples=sample_size, drift_factor=drift_factor)
        
        # Mostrar información de los datos
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Datos de referencia", f"{len(X_train):,}")
        with col2:
            st.metric("Nuevos datos", f"{len(new_X):,}")
        with col3:
            st.metric("Factor de drift", f"{drift_factor:.2f}")
        
        st.markdown("---")
        
        # Tabs para organizar la información
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Análisis por Feature", 
            "📊 Métricas Globales", 
            "📉 Visualizaciones",
            "🚨 Recomendaciones"
        ])
        
        with tab1:
            st.subheader("Análisis detallado por Feature")
            
            # Seleccionar features para analizar
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            selected_features = st.multiselect(
                "Seleccionar features para analizar",
                numeric_cols,
                default=numeric_cols[:min(10, len(numeric_cols))] if len(numeric_cols) > 0 else []
            )
            
            if selected_features:
                # Crear DataFrame con resultados
                results = []
                
                progress_bar = st.progress(0)
                for i, feature in enumerate(selected_features):
                    if feature in X_train.columns and feature in new_X.columns:
                        metrics = calculate_metrics(
                            X_train[feature], 
                            new_X[feature], 
                            feature
                        )
                        
                        if 'error' not in metrics:
                            results.append({
                                'Feature': feature,
                                'Tipo': 'Numérica',
                                'Diff %': round(metrics.get('mean_diff_pct', 0), 2),
                                'PSI': round(metrics.get('psi', 0), 3),
                                'KS Stat': round(metrics.get('ks_statistic', 0), 3),
                                'JS Div': round(metrics.get('js_divergence', 0), 3),
                                'Drift': '✅ Sí' if metrics.get('has_drift', False) else '❌ No',
                                'Severidad': determine_severity(metrics)
                            })
                        else:
                            results.append({
                                'Feature': feature,
                                'Tipo': 'Numérica',
                                'Diff %': 'Error',
                                'PSI': 'Error',
                                'KS Stat': 'Error',
                                'JS Div': 'Error',
                                'Drift': '❌ Error',
                                'Severidad': '❌ Error'
                            })
                    
                    progress_bar.progress((i + 1) / len(selected_features))
                
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # Mostrar tabla
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Exportar resultados
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar resultados",
                        data=csv,
                        file_name=f"drift_analysis_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No se pudieron calcular métricas para las features seleccionadas")
            else:
                st.info("Selecciona al menos una feature para analizar")
        
        with tab2:
            st.subheader("Métricas Globales de Drift")
            
            if 'results_df' in locals() and len(results_df) > 0:
                # Calcular métricas globales
                total_features = len(results_df)
                # Filtrar filas sin error
                valid_results = results_df[~results_df['Diff %'].astype(str).str.contains('Error')]
                
                if len(valid_results) > 0:
                    features_with_drift = len(valid_results[valid_results['Drift'] == '✅ Sí'])
                    drift_percentage = (features_with_drift / len(valid_results) * 100)
                    
                    # Mostrar KPIs
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Features analizadas", total_features)
                    with col2:
                        st.metric("Features con drift", features_with_drift)
                    with col3:
                        st.metric("% con drift", f"{drift_percentage:.1f}%")
                    with col4:
                        # Indicador de severidad
                        if drift_percentage > 30:
                            severity = "🔴 Crítico"
                        elif drift_percentage > 20:
                            severity = "🟡 Alto"
                        elif drift_percentage > 10:
                            severity = "🟠 Moderado"
                        else:
                            severity = "🟢 Bajo"
                        st.metric("Severidad", severity)
                    
                    # Distribución de métricas
                    st.subheader("Distribución de Métricas")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Convertir a numérico para el histograma
                        try:
                            diff_values = pd.to_numeric(valid_results['Diff %'], errors='coerce')
                            fig1 = px.histogram(
                                valid_results, 
                                x=diff_values,
                                title='Distribución de Diferencias %',
                                nbins=20,
                                labels={'x': 'Diferencia %'}
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        except:
                            st.info("No se pudo crear el histograma de diferencias")
                    
                    with col2:
                        try:
                            psi_values = pd.to_numeric(valid_results['PSI'], errors='coerce')
                            fig2 = px.histogram(
                                valid_results, 
                                x=psi_values,
                                title='Distribución de PSI',
                                nbins=20,
                                labels={'x': 'PSI'}
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        except:
                            st.info("No se pudo crear el histograma de PSI")
                else:
                    st.warning("No hay resultados válidos para calcular métricas globales")
            else:
                st.info("Ejecuta el análisis en la pestaña anterior primero")
        
        with tab3:
            st.subheader("Visualizaciones de Drift")
            
            if 'results_df' in locals() and len(results_df) > 0:
                # Filtrar resultados válidos
                valid_results = results_df[~results_df['Diff %'].astype(str).str.contains('Error')]
                
                if len(valid_results) > 0:
                    # Heatmap de correlación de métricas
                    st.subheader("Heatmap de Métricas de Drift")
                    
                    # Seleccionar solo columnas numéricas
                    numeric_metrics = ['Diff %', 'PSI', 'KS Stat', 'JS Div']
                    metrics_data = valid_results[numeric_metrics].apply(pd.to_numeric, errors='coerce')
                    
                    if not metrics_data.isnull().all().all():
                        fig = px.imshow(
                            metrics_data.T,
                            labels=dict(x="Features", y="Métricas", color="Valor"),
                            x=valid_results['Feature'],
                            title="Heatmap de Métricas por Feature",
                            color_continuous_scale="RdYlGn_r",
                            aspect="auto"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No hay datos numéricos suficientes para el heatmap")
                    
                    # Scatter plot PSI vs Diff %
                    st.subheader("Relación PSI vs Diferencia %")
                    
                    try:
                        psi_numeric = pd.to_numeric(valid_results['PSI'], errors='coerce')
                        diff_numeric = pd.to_numeric(valid_results['Diff %'], errors='coerce')
                        
                        fig_scatter = px.scatter(
                            valid_results,
                            x=diff_numeric,
                            y=psi_numeric,
                            color=valid_results['Severidad'],
                            hover_name='Feature',
                            size=pd.to_numeric(valid_results['KS Stat'], errors='coerce'),
                            title="PSI vs Diferencia % por Feature"
                        )
                        
                        # Añadir líneas de umbral
                        fig_scatter.add_hline(y=psi_threshold, line_dash="dash", line_color="red")
                        fig_scatter.add_vline(x=diff_threshold, line_dash="dash", line_color="red")
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    except:
                        st.info("No se pudo crear el gráfico de dispersión")
                    
                    # Gráfico de barras de severidad
                    st.subheader("Distribución de Severidad")
                    
                    severity_counts = valid_results['Severidad'].value_counts()
                    fig_severity = px.bar(
                        x=severity_counts.index,
                        y=severity_counts.values,
                        title="Cantidad de Features por Nivel de Severidad",
                        labels={'x': 'Severidad', 'y': 'Número de Features'},
                        color=severity_counts.index
                    )
                    st.plotly_chart(fig_severity, use_container_width=True)
                else:
                    st.info("No hay resultados válidos para visualizar")
            else:
                st.info("Ejecuta el análisis en la primera pestaña primero")
        
        with tab4:
            st.subheader("Recomendaciones y Alertas")
            
            if 'results_df' in locals() and len(results_df) > 0:
                # Filtrar resultados válidos
                valid_results = results_df[~results_df['Diff %'].astype(str).str.contains('Error')]
                
                if len(valid_results) > 0:
                    drift_percentage = len(valid_results[valid_results['Drift'] == '✅ Sí']) / len(valid_results) * 100
                    
                    # Recomendaciones basadas en severidad
                    if drift_percentage > 30:
                        st.error("""
                        ## 🚨 ALERTA CRÍTICA
                        
                        **Nivel de drift: CRÍTICO** (>30% de features con drift)
                        
                        ### Acciones inmediatas requeridas:
                        1. **Detener el modelo en producción** - El desempeño está comprometido
                        2. **Retrenar inmediatamente** - Usar datos de los últimos 3 meses
                        3. **Investigación urgente** - Identificar causas del cambio
                        4. **Comunicar a stakeholders** - Notificar impacto en negocio
                        
                        **Tiempo estimado para acción: < 24 horas**
                        """)
                    
                    elif drift_percentage > 20:
                        st.warning("""
                        ## ⚠️ ALERTA ALTA
                        
                        **Nivel de drift: ALTO** (20-30% de features con drift)
                        
                        ### Acciones recomendadas (próximas 48h):
                        1. **Programar retraining** - Para las próximas 48 horas
                        2. **Monitoreo intensivo** - Revisar performance cada 6 horas
                        3. **Análisis de causas** - Investigar features con mayor drift
                        4. **Preparar rollback** - Tener versión estable disponible
                        
                        **Tiempo estimado para acción: 48 horas**
                        """)
                    
                    elif drift_percentage > 10:
                        st.info("""
                        ## 📢 ADVERTENCIA MODERADA
                        
                        **Nivel de drift: MODERADO** (10-20% de features con drift)
                        
                        ### Acciones recomendadas:
                        1. **Monitoreo diario** - Seguimiento cercano del drift
                        2. **Preparar datos** - Recopilar datos para posible retraining
                        3. **Análisis de tendencias** - Evaluar evolución del drift
                        4. **Documentar cambios** - Registrar observaciones
                        
                        **Tiempo estimado para acción: 1 semana**
                        """)
                    
                    else:
                        st.success("""
                        ## ✅ ESTADO ESTABLE
                        
                        **Nivel de drift: BAJO** (<10% de features con drift)
                        
                        ### Acciones de mantenimiento:
                        1. **Monitoreo semanal** - Continuar con frecuencia normal
                        2. **Revision periódica** - Revisar tendencias mensuales
                        3. **Actualizar baseline** - Si hay cambios menores
                        4. **Mantener documentación** - Actualizar registros
                        
                        **Próxima revisión programada: 1 mes**
                        """)
                    
                    # Features con mayor drift
                    high_drift_features = valid_results[valid_results['Drift'] == '✅ Sí']
                    if len(high_drift_features) > 0:
                        st.subheader("Features que requieren atención")
                        
                        # Ordenar por severidad
                        high_drift_features = high_drift_features.sort_values('Diff %', ascending=False)
                        
                        for _, row in high_drift_features.head(5).iterrows():
                            with st.expander(f"🔍 {row['Feature']} - {row['Severidad']}"):
                                st.write(f"**Métricas:**")
                                st.write(f"- Diferencia: {row['Diff %']:.1f}%")
                                st.write(f"- PSI: {row['PSI']:.3f}")
                                st.write(f"- KS Statistic: {row['KS Stat']:.3f}")
                                st.write(f"- JS Divergence: {row['JS Div']:.3f}")
                                
                                st.write("**Acciones recomendadas para esta feature:**")
                                try:
                                    diff_pct = float(row['Diff %'])
                                    if diff_pct > 30:
                                        st.write("1. Investigar cambios en recolección de datos")
                                        st.write("2. Validar transformaciones aplicadas")
                                        st.write("3. Considerar exclusión temporal")
                                    elif diff_pct > 20:
                                        st.write("1. Revisar procesos de limpieza")
                                        st.write("2. Verificar outliers")
                                        st.write("3. Monitorear evolución")
                                    else:
                                        st.write("1. Observar tendencia")
                                        st.write("2. Documentar comportamiento")
                                except:
                                    st.write("1. Verificar calidad de datos")
                                    st.write("2. Revisar transformaciones")
                else:
                    st.warning("No hay resultados válidos para generar recomendaciones")
            else:
                st.info("Ejecuta el análisis en la primera pestaña para ver recomendaciones")

else:
    # Pantalla inicial
    st.markdown("""
    ## 🎯 Sistema de Monitoreo de Data Drift
    
    Este sistema permite detectar cambios en la distribución de los datos que puedan afectar el desempeño de modelos de machine learning.
    
    ### 📋 Funcionalidades principales:
    
    1. **Detección de Data Drift** - Usando múltiples métricas estadísticas
    2. **Métricas implementadas:**
       - 📊 Population Stability Index (PSI)
       - 📈 Kolmogorov-Smirnov Test
       - 📉 Jensen-Shannon Divergence
       - 🧮 Chi-cuadrado
       - 📐 Diferencia porcentual en medias
    
    3. **Visualización interactiva** - Gráficos y tablas en tiempo real
    4. **Sistema de alertas** - Clasificación por severidad
    5. **Recomendaciones automáticas** - Acciones sugeridas según nivel de drift
    
    ### 🚀 Para comenzar:
    1. Configura los parámetros en la barra lateral
    2. Haz clic en **"Ejecutar Monitoreo"**
    3. Revisa los resultados en las diferentes pestañas
    """)
    
    # Mostrar información del dataset si está disponible
    if df is not None:
        with st.expander("📁 Información del dataset cargado"):
            st.write(f"**Dimensiones:** {df.shape[0]} filas × {df.shape[1]} columnas")
            
            # Información del target
            if 'Pago_atiempo' in df.columns:
                target_dist = df['Pago_atiempo'].value_counts()
                st.write(f"**Distribución del target (Pago_atiempo):**")
                st.write(f"- Paga a tiempo (1): {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(df)*100:.1f}%)")
                st.write(f"- No paga a tiempo (0): {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(df)*100:.1f}%)")
            
            # Features numéricas disponibles
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.write(f"**Features numéricas disponibles:** {len(numeric_cols)}")
            
            # Mostrar nombres de algunas features
            if len(numeric_cols) > 0:
                st.write("**Algunas features disponibles:**")
                cols_display = st.columns(3)
                for i, col in enumerate(numeric_cols[:9]):
                    with cols_display[i % 3]:
                        st.code(col)
            
            # Preview del dataset
            if st.checkbox("Mostrar preview del dataset"):
                st.dataframe(df.head(10), use_container_width=True)

# Footer
st.markdown("---")
st.caption("Sistema de Monitoreo de Data Drift v1.0 | Última actualización: " + time.strftime("%Y-%m-%d %H:%M:%S"))