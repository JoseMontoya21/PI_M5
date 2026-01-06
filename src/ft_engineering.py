# librerías
import pandas as pd
from carga_datos import cargarDatosLimpios
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


# carga de datos
df = cargarDatosLimpios()

# preview del data dataset
df.info()
print(df.head())
print(df.describe())

# Paso 1: features/target split
X = df.drop('Pago_atiempo', axis=1) # features
y = df['Pago_atiempo']             # target

# Paso 2: definir variables por tipo
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numeric features:")
print(num_features)
print("\nCategorical features:")
print(cat_features)

# Paso 3: Crear pipelines para cada ruta
## Ruta 1: numéricas
num_transformer =  Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

## Ruta 2: categóricas
cat_transformer = Pipeline(steps=[
    ('to_str', FunctionTransformer(lambda x: x.astype(str))),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Paso 4: Combinar las 2 rutas en ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# Paso 5: dividir el dataset en train/test (antes de preprocesar)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Paso 6: Aplicamos el preprocesamiento
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Paso 7: resultados del preprocesamiento 
print("\n" + "="*50)
print("RESULTADOS DEL PREPROCESAMIENTO")
print("="*50)
print(f"X_train preprocesado - Shape: {X_train_processed.shape}")
print(f"X_test preprocesado - Shape: {X_test_processed.shape}")
print(f"Número de features después del encoding: {X_train_processed.shape[1]}")

# Paso 8: construimos una función para "exportar": ft_engineering()
def ft_engineering(X_data=None):
    """
    Función para crear y retornar el preprocesador configurado.
    
    Args:
        X_data: DataFrame opcional para determinar las columnas.
                Si es None, usa X global.
    
    Returns:
        preprocessor: ColumnTransformer configurado
    """
    if X_data is None:
        X_data = X
    
    # Definir columnas por tipo
    num_cols = X_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Pipelines individuales
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('to_str', FunctionTransformer(lambda x: x.astype(str))),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    
    return preprocessor

# Función adicional para obtener nombres de features después del encoding
def get_feature_names(preprocessor, X_data=None):
    """Obtiene los nombres de las features después del preprocessing"""
    if X_data is None:
        X_data = X
    
    feature_names = []
    
    # Nombres de features numéricas
    num_cols = X_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    feature_names.extend(num_cols)
    
    # Nombres de features categóricas (después de one-hot encoding)
    cat_cols = X_data.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_transformer = preprocessor.named_transformers['cat']
    
    # Obtener nombres de one-hot encoding si ya está ajustado
    if hasattr(cat_transformer, 'named_steps'):
        onehot = cat_transformer.named_steps['onehot']
        if hasattr(onehot, 'get_feature_names_out'):
            cat_features = onehot.get_feature_names_out(cat_cols)
            feature_names.extend(cat_features.tolist())
    
    return feature_names

# Ejemplo de uso si se ejecuta directamente
if __name__ == "__main__":
    print("\n" + "="*50)
    print("EJECUCIÓN DIRECTA DE ft_engineering.py")
    print("="*50)
    
    # Probar la función ft_engineering
    preprocessor = ft_engineering()
    print(f"\nPreprocesador creado exitosamente")
    
    # Simplemente mostrar los nombres de los transformadores configurados
    print(f"Transformadores configurados: ['num', 'cat']")
    
    # Información adicional sobre el preprocesador
    print(f"Número de features numéricas: {len(num_features)}")
    print(f"Número de features categóricas: {len(cat_features)}")
    print(f"Total de features después de encoding: {len(num_features) + len(cat_features)}")





