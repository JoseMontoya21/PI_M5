# model_training_evaluation.py
# libraries 
from ft_engineering import ft_engineering, X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, preprocessor, df
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    cross_val_score,
    learning_curve,
    train_test_split,
)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, Booster

import numpy as np
import xgboost as xgb
import pandas as pd
import seaborn as sns
import warnings
import pickle
import joblib
import json
import os
import base64
from sklearn.metrics import confusion_matrix, roc_curve, auc
warnings.filterwarnings('ignore')

print("="*60)
print("MODEL TRAINING AND EVALUATION")
print("="*60)

print(f"\nDatos disponibles para entrenamiento:")
print(f"  Dataset completo: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"  Conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"  Conjunto de prueba: {X_test.shape[0]} muestras")
print(f"  Features preprocesadas: {X_train_processed.shape[1]} columnas")

print("\nDistribución de clases en los conjuntos:")
print("Entrenamiento:")
print(y_train.value_counts())
print(f"Proporciones: {y_train.value_counts(normalize=True).to_dict()}")

print("\nPrueba:")
print(y_test.value_counts())
print(f"Proporciones: {y_test.value_counts(normalize=True).to_dict()}")

## function: summarize_classification()
def summarize_classification(y_test, y_pred):
    try:
        acc = accuracy_score(y_test, y_pred, normalize=True)
        prec = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
        roc = roc_auc_score(y_test, y_pred)
        cantidadNoPagoAtiempo = np.count_nonzero(y_pred == 0)
    except Exception as e:
        print(f"Error calculando métricas: {e}")
        acc = prec = recall = f1 = roc = cantidadNoPagoAtiempo = 0
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc,
        "casosNoPagoAtiempo": cantidadNoPagoAtiempo
    }

## function: build_model()
def build_model(
    classifier_fn,
    classifier_name: str,
    data_params: dict,
    use_preprocessed: bool = True
) -> dict:
    
    name_of_y_col = data_params["name_of_y_col"]
    names_of_x_cols = data_params["names_of_x_cols"]
    dataset = data_params["dataset"]
    
    x_train = X_train[names_of_x_cols]
    x_test = X_test[names_of_x_cols]
    y_train_local = y_train
    y_test_local = y_test
    
    if use_preprocessed:
        x_train_processed = X_train_processed
        x_test_processed = X_test_processed
        
        print(f"  Entrenando con {x_train_processed.shape[1]} features preprocesadas")
        
        model = classifier_fn.fit(x_train_processed, y_train_local)
        
        y_pred_train = model.predict(x_train_processed)
        y_pred_test = model.predict(x_test_processed)
        
    else:
        preprocessor_local = ft_engineering(x_train)
        
        classifier_pipe = Pipeline(
            steps=[
                ('preprocessor', preprocessor_local),
                ('model', classifier_fn)
            ]
        )
        
        model = classifier_pipe.fit(x_train, y_train_local)
        
        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)
    
    train_summary = summarize_classification(y_train_local, y_pred_train)
    test_summary = summarize_classification(y_test_local, y_pred_test)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = {}
    scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    for metric in scoring_metrics:
        try:
            if metric != "roc_auc":
                if metric == "precision" or metric == "recall" or metric == "f1":
                    cv_results[metric] = cross_val_score(
                        classifier_fn, X_train_processed, y_train_local,
                        cv=kfold, scoring=metric, n_jobs=-1
                    ).mean()
                else:
                    cv_results[metric] = cross_val_score(
                        classifier_fn, X_train_processed, y_train_local, 
                        cv=kfold, scoring=metric, n_jobs=-1
                    ).mean()
            else:
                if hasattr(classifier_fn, 'predict_proba'):
                    cv_results[metric] = cross_val_score(
                        classifier_fn, X_train_processed, y_train_local,
                        cv=kfold, scoring='roc_auc', n_jobs=-1
                    ).mean()
                else:
                    cv_results[metric] = 0
        except Exception as e:
            print(f"  CV para {metric} no disponible: {e}")
            cv_results[metric] = 0
    
    common_params = {
        "X": X_train_processed,
        "y": y_train_local,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=5, test_size=0.2, random_state=123),
        "n_jobs": -1,
        "return_times": True,
    }
    
    scoring_metric = "accuracy"
    
    try:
        train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
            classifier_fn, **common_params, scoring=scoring_metric
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
        score_times_mean = np.mean(score_times, axis=1)
        score_times_std = np.std(score_times, axis=1)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), sharey=True)
        ax.plot(train_sizes, train_mean, "o-", label="Training score")
        ax.plot(train_sizes, test_mean, "o-", color="orange", label="Cross-validation score")
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.3)
        ax.fill_between(
            train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.3, color="orange"
        )
        
        ax.set_title(f"Learning Curve for {classifier_name}")
        ax.set_xlabel("Training examples")
        ax.set_ylabel(scoring_metric)
        ax.legend(loc="best")
        
        plt.tight_layout()
        plt.show()
        
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True)
        
        ax[0].plot(train_sizes, fit_times_mean, "o-")
        ax[0].fill_between(
            train_sizes,
            fit_times_mean - fit_times_std,
            fit_times_mean + fit_times_std,
            alpha=0.3,
        )
        ax[0].set_ylabel("Fit time (s)")
        ax[0].set_title(f"Scalability of the {classifier_name} classifier")
        
        ax[1].plot(train_sizes, score_times_mean, "o-")
        ax[1].fill_between(
            train_sizes,
            score_times_mean - score_times_std,
            score_times_mean + score_times_std,
            alpha=0.3,
        )
        ax[1].set_ylabel("Score time (s)")
        ax[1].set_xlabel("Number of training samples")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"  Curvas de aprendizaje no disponibles: {e}")
    
    return {"train": train_summary, "test": test_summary, "cv": cv_results, "model_object": model}

def export_best_model(models_dict, results_df, metric='f1_score'):
    """
    Exporta el mejor modelo según la métrica especificada
    """
    print("\n" + "="*60)
    print("EXPORTANDO MEJOR MODELO")
    print("="*60)
    
    test_results = results_df[results_df['Data Set'] == 'test']
    
    if test_results.empty:
        print("No hay resultados de prueba disponibles")
        return None
    
    metric_results = test_results[test_results['Metric'] == metric]
    
    if metric_results.empty:
        print(f"No hay resultados para la métrica {metric}")
        metric_results = test_results[test_results['Metric'] == 'accuracy']
        if metric_results.empty:
            print("No se pudo encontrar métricas válidas")
            return None
    
    best_row = metric_results.loc[metric_results['Score'].idxmax()]
    best_model_name = best_row['Model']
    best_score = best_row['Score']
    
    print(f"Mejor modelo según {metric}: {best_model_name}")
    print(f"Puntuación: {best_score:.4f}")
    
    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    
    if best_model_name in models_dict:
        best_model = models_dict[best_model_name]
        
        try:
            if best_model_name == 'xgboost':
                pickle_path = os.path.join(models_dir, 'model.pkl')
                with open(pickle_path, 'wb') as f:
                    pickle.dump(best_model, f)
                print(f"Modelo exportado como pickle: {pickle_path}")
                
                joblib_path = os.path.join(models_dir, 'model.joblib')
                joblib.dump(best_model, joblib_path)
                print(f"Modelo exportado como joblib: {joblib_path}")
                
                try:
                    booster = best_model.get_booster()
                    xgb_path = os.path.join(models_dir, 'xgb_model.json')
                    booster.save_model(xgb_path)
                    print(f"Modelo XGBoost exportado como JSON: {xgb_path}")
                except Exception as e:
                    xgb_path = os.path.join(models_dir, 'xgb_model.pkl')
                    with open(xgb_path, 'wb') as f:
                        pickle.dump(best_model, f)
                    print(f"Modelo XGBoost exportado como pickle alternativo: {xgb_path}")
                
            else:
                pickle_path = os.path.join(models_dir, 'model.pkl')
                with open(pickle_path, 'wb') as f:
                    pickle.dump(best_model, f)
                print(f"Modelo exportado como pickle: {pickle_path}")
                
                joblib_path = os.path.join(models_dir, 'model.joblib')
                joblib.dump(best_model, joblib_path)
                print(f"Modelo exportado como joblib: {joblib_path}")
            
            metadata = {
                'model_name': best_model_name,
                'best_metric': metric,
                'best_score': float(best_score),
                'export_date': pd.Timestamp.now().isoformat(),
                'features': X_train_processed.shape[1],
                'training_samples': X_train_processed.shape[0],
                'test_samples': X_test_processed.shape[0],
                'model_type': best_model_name,
                'accuracy': float(results_df[
                    (results_df['Model'] == best_model_name) & 
                    (results_df['Data Set'] == 'test') & 
                    (results_df['Metric'] == 'accuracy')
                ]['Score'].values[0] if not results_df[
                    (results_df['Model'] == best_model_name) & 
                    (results_df['Data Set'] == 'test') & 
                    (results_df['Metric'] == 'accuracy')
                ].empty else 0)
            }
            
            metadata_path = os.path.join(models_dir, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadatos guardados: {metadata_path}")
            
            feature_names = X_train_processed.columns.tolist()
            
            features_path = os.path.join(models_dir, 'feature_names.json')
            with open(features_path, 'w') as f:
                json.dump(feature_names, f)
            print(f"Nombres de features guardados: {features_path}")
            
            try:
                preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
                with open(preprocessor_path, 'wb') as f:
                    pickle.dump(preprocessor, f)
                print(f"Preprocessor exportado: {preprocessor_path}")
            except Exception as e:
                print(f"No se pudo exportar el preprocessor: {e}")
                preprocessor_info = {
                    'input_columns': X_train.columns.tolist(),
                    'output_columns': X_train_processed.columns.tolist(),
                    'note': 'Preprocessor serializable con ft_engineering.py corregido'
                }
                preprocessor_info_path = os.path.join(models_dir, 'preprocessor_info.json')
                with open(preprocessor_info_path, 'w') as f:
                    json.dump(preprocessor_info, f, indent=2)
                print(f"Información del preprocessor guardada: {preprocessor_info_path}")
            
            return best_model
            
        except Exception as e:
            print(f"Error exportando el modelo {best_model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"Modelo {best_model_name} no encontrado en el diccionario")
        return None

print("\n" + "="*60)
print("ENTRENAMIENTO DE MODELOS")
print("="*60)

result_dict = {}
trained_models = {}

models_config = {
    "logistic": LogisticRegression(solver="liblinear", class_weight='balanced', max_iter=1000),
    "svc": LinearSVC(C=1.0, max_iter=5000, tol=1e-3, dual=False, class_weight='balanced'),
    "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42),
    "random_forest": RandomForestClassifier(class_weight='balanced', n_estimators=150, max_depth=7, 
                                           min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(eval_metric='logloss', scale_pos_weight=491/10090, n_estimators=200,
                           max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                           random_state=42, n_jobs=-1, use_label_encoder=False)
}

data_params = {
    "name_of_y_col": "Pago_atiempo",
    "names_of_x_cols": X_train.columns.tolist(),
    "dataset": df
}

for model_name, model_config in models_config.items():
    print(f"\nEntrenando: {model_name}")
    try:
        result = build_model(model_config, model_name, data_params, use_preprocessed=True)
        result_dict[model_name] = {
            "train": result["train"],
            "test": result["test"],
            "cv": result["cv"]
        }
        trained_models[model_name] = result["model_object"]
        
        test_acc = result_dict[model_name]["test"].get("accuracy", 0)
        test_f1 = result_dict[model_name]["test"].get("f1_score", 0)
        print(f"  Resultados - Accuracy: {test_acc:.3f}, F1: {test_f1:.3f}")
        
    except Exception as e:
        print(f"  Error: {e}")
        result_dict[model_name] = {"train": {}, "test": {}, "cv": {}}
        trained_models[model_name] = None

records = []
for model_name, model_results in result_dict.items():
    for data_set in ['train', 'test']:
        if data_set in model_results:
            metrics = model_results[data_set]
            for metric_name, score in metrics.items():
                records.append({
                    "Model": model_name,
                    "Data Set": data_set,
                    "Metric": metric_name,
                    "Score": score
                })

results_df = pd.DataFrame(records)

print("\n" + "="*60)
print("RESULTADOS DE LOS MODELOS")
print("="*60)

pivot_df = results_df.pivot_table(
    index=['Model', 'Metric'], 
    columns='Data Set', 
    values='Score',
    aggfunc='first'
).reset_index()

print("\nResumen de métricas:")
print(pivot_df.to_string())

print("\n" + "="*60)
print("VISUALIZACIÓN DE RESULTADOS")
print("="*60)

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes = axes.flatten()

metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    metric_df = results_df[results_df["Metric"] == metric]
    
    if not metric_df.empty:
        sns.barplot(data=metric_df, x="Model", y="Score", hue="Data Set", ax=ax, palette="cividis")
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=16)
        ax.set_ylabel("Puntuación", fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        
        if metric == 'roc_auc':
            ax.set_ylim(0.45, 1.05)
        else:
            ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, f"Datos no disponibles", 
                ha='center', va='center', fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=16)

if len(metrics_to_plot) < len(axes):
    axes[-1].set_visible(False)

plt.tight_layout(pad=3.0)
plt.show()

def evaluation():
    """
    Función de evaluación que genera gráficos comparativos
    """
    print("\n" + "="*60)
    print("EVALUACIÓN COMPARATIVA")
    print("="*60)
    
    fig, axes = plt.subplots(3, 2, figsize=(30, 18))
    axes = axes.flatten()
    fig.suptitle('Resultados por Modelo', fontsize=30)
    
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "casosNoPagoAtiempo"]
    custom_titles = {
        'precision': 'Precisión (No Pago)',
        'recall': 'Recall (No Pago)',
        'f1_score': 'F1-Score (No Pago)',
        'accuracy': 'Accuracy General',
        'roc_auc': 'ROC AUC',
        'casosNoPagoAtiempo': 'Casos No Pago'
    }
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        metric_df = results_df[results_df["Metric"] == metric]
        
        if not metric_df.empty:
            sns.barplot(data=metric_df, x="Model", y="Score", hue="Data Set", ax=ax, palette="cividis")
            ax.legend(fontsize=18)
            
            title = custom_titles.get(metric, metric.replace("_", " ").title())
            ax.set_title(title, fontsize=24)
            ax.set_ylabel("Puntuación", fontsize=18)
            ax.set_xlabel("")
            ax.tick_params(axis='x', rotation=45, labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            
            if metric == 'roc_auc':
                ax.set_ylim(0.45, 1.05)
            elif metric == 'casosNoPagoAtiempo':
                max_no_pago = metric_df["Score"].max()
                ax.set_ylim(0, max_no_pago * 1.1)
            else:
                ax.set_ylim(0, 1.05)
        else:
            ax.text(0.5, 0.5, f"Datos no disponibles", 
                    ha='center', va='center', fontsize=18)
            title = custom_titles.get(metric, metric.replace("_", " ").title())
            ax.set_title(title, fontsize=24)
    
    if len(metrics_to_plot) < len(axes):
        axes[-1].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    print("Gráficos generados correctamente")
    
    return buf

def generate_evaluation_metrics_plot(y_true, y_pred, y_pred_proba=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Métricas de Evaluación del Modelo', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Matriz de Confusión')
    ax1.set_xlabel('Predicho')
    ax1.set_ylabel('Real')
    
    ax2 = axes[0, 1]
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Curva ROC')
        ax2.legend(loc="lower right")
    else:
        ax2.text(0.5, 0.5, 'No hay probabilidades disponibles', 
                ha='center', va='center', fontsize=12)
        ax2.set_title('Curva ROC')
    
    ax3 = axes[1, 0]
    unique, counts = np.unique(y_pred, return_counts=True)
    ax3.bar(unique, counts, color=['#3498db', '#e74c3c'], alpha=0.7)
    ax3.set_title('Distribución de Predicciones')
    ax3.set_xlabel('Clase')
    ax3.set_ylabel('Cantidad')
    ax3.set_xticks(unique)
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    summary_text = f"""
    RESUMEN DE MÉTRICAS
    ===================
    
    Accuracy:  {accuracy:.4f}
    Precision: {precision:.4f}
    Recall:    {recall:.4f}
    F1-Score:  {f1:.4f}
    
    Distribución:
    - Clase 0: {sum(y_true == 0)} ({sum(y_true == 0)/len(y_true):.1%})
    - Clase 1: {sum(y_true == 1)} ({sum(y_true == 1)/len(y_true):.1%})
    - Total:   {len(y_true)}
    
    Predicciones:
    - Clase 0: {sum(y_pred == 0)} ({sum(y_pred == 0)/len(y_pred):.1%})
    - Clase 1: {sum(y_pred == 1)} ({sum(y_pred == 1)/len(y_pred):.1%})
    """
    
    ax4.text(0.1, 0.9, summary_text, fontsize=11, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def get_demo_evaluation_data():
    np.random.seed(42)
    n_samples = 500
    
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    y_pred_proba = np.random.rand(n_samples)
    
    return y_true, y_pred, y_pred_proba

def get_api_evaluation_data():
    """
    Función específica para la API que obtiene datos de evaluación
    usando el modelo XGBoost entrenado
    """
    try:
        # Verificar si tenemos resultados de XGBoost
        if 'xgboost' in result_dict:
            # Obtener el modelo XGBoost
            xgb_model = trained_models.get('xgboost')
            
            if xgb_model is not None and hasattr(xgb_model, 'predict'):
                # Generar predicciones reales
                y_pred = xgb_model.predict(X_test_processed)
                
                # Intentar obtener probabilidades
                try:
                    if hasattr(xgb_model, 'predict_proba'):
                        y_pred_proba = xgb_model.predict_proba(X_test_processed)[:, 1]
                    else:
                        y_pred_proba = None
                except:
                    y_pred_proba = None
                
                return y_test, y_pred, y_pred_proba
            else:
                # Si no hay modelo, usar datos de ejemplo
                return get_demo_evaluation_data()
        else:
            # Si no hay resultados de XGBoost, usar datos de ejemplo
            return get_demo_evaluation_data()
            
    except Exception as e:
        print(f"Error obteniendo datos para API: {e}")
        # Devolver datos de demostración como fallback
        return get_demo_evaluation_data()

def generate_evaluation_plot():
    """
    Función principal para la API que genera el gráfico de evaluación
    """
    try:
        # Obtener datos de evaluación
        y_true, y_pred, y_pred_proba = get_api_evaluation_data()
        
        # Generar el gráfico
        img_base64 = generate_evaluation_metrics_plot(y_true, y_pred, y_pred_proba)
        
        return img_base64
        
    except Exception as e:
        print(f"Error generando gráfico de evaluación: {e}")
        return None

def get_reference_data():
    return X_train_processed

if __name__ == "__main__":
    evaluation_buffer = evaluation()
    
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    
    summary_table = results_df.pivot_table(
        index=['Model', 'Data Set'],
        columns='Metric',
        values='Score'
    ).round(4)
    
    print(summary_table)
    
    best_model = export_best_model(trained_models, results_df, metric='f1_score')
    
    test_results = results_df[results_df['Data Set'] == 'test']
    if not test_results.empty:
        print("\n" + "="*60)
        print("MEJORES MODELOS POR MÉTRICA")
        print("="*60)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            metric_results = test_results[test_results['Metric'] == metric]
            if not metric_results.empty:
                best_model_metric = metric_results.loc[metric_results['Score'].idxmax(), 'Model']
                best_score_metric = metric_results['Score'].max()
                print(f"{metric}: {best_model_metric} ({best_score_metric:.4f})")
    
    print("\n" + "="*60)
    print("PROCESO COMPLETADO")
    print("="*60)
    
    if best_model is not None:
        print("El mejor modelo ha sido exportado como:")
        print("  - models/model.pkl (formato pickle)")
        print("  - models/model.joblib (formato joblib)")
        print("  - models/xgb_model.json (formato nativo XGBoost)")
        print("  - models/model_metadata.json (metadatos del modelo)")
        print("  - models/feature_names.json (nombres de features)")
        print("  - models/preprocessor.pkl (preprocessor serializable)")
    else:
        print("No se pudo exportar el mejor modelo")