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
from xgboost import XGBClassifier

import numpy as np
import xgboost as xgb
import pandas as pd
import seaborn as sns
import warnings
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
        
        model_name = classifier_fn.__class__.__name__
        ax.set_title(f"Learning Curve for {model_name}")
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
        ax[0].set_title(f"Scalability of the {model_name} classifier")
        
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
    
    return {"train": train_summary, "test": test_summary, "cv": cv_results}

print("\n" + "="*60)
print("ENTRENAMIENTO DE MODELOS")
print("="*60)

result_dict = {}
models = {
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

for model_name, model in models.items():
    print(f"\nEntrenando: {model_name}")
    try:
        result_dict[model_name] = build_model(model, data_params, use_preprocessed=True)
        
        test_acc = result_dict[model_name]["test"].get("accuracy", 0)
        test_f1 = result_dict[model_name]["test"].get("f1_score", 0)
        print(f"  Resultados - Accuracy: {test_acc:.3f}, F1: {test_f1:.3f}")
        
    except Exception as e:
        print(f"  Error: {e}")
        result_dict[model_name] = {"train": {}, "test": {}, "cv": {}}

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
    
    test_results = results_df[results_df['Data Set'] == 'test']
    if not test_results.empty:
        f1_scores = test_results[test_results['Metric'] == 'f1_score']
        if not f1_scores.empty:
            best_model = f1_scores.loc[f1_scores['Score'].idxmax(), 'Model']
            best_score = f1_scores['Score'].max()
            print(f"\nModelo con mejor F1-Score: {best_model}")
            print(f"Valor F1-Score: {best_score:.4f}")