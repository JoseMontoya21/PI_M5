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

# Función auxiliar serializable
def convert_to_str(x):
    """Convierte datos a string (serializable)"""
    return x.astype(str)

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

## Ruta 2: categóricas - SIN FUNCIONES LAMBDA
cat_transformer = Pipeline(steps=[
    ('to_str', FunctionTransformer(convert_to_str)),  # Cambiado: función definida
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

# Convertir a DataFrame con nombres de columnas
def get_feature_names_from_preprocessor(preprocessor, X_data):
    """Obtiene nombres de features del preprocessor"""
    feature_names = []
    
    # Features numéricas (mantienen sus nombres)
    feature_names.extend(num_features)
    
    # Features categóricas (one-hot encoding)
    # Usar named_transformers_ (con guión bajo)
    cat_transformer = preprocessor.named_transformers_['cat']
    if hasattr(cat_transformer, 'named_steps'):
        onehot = cat_transformer.named_steps['onehot']
        if hasattr(onehot, 'get_feature_names_out'):
            cat_features_out = onehot.get_feature_names_out(cat_features)
            feature_names.extend(cat_features_out.tolist())
    
    return feature_names

# Paso 6.1: Obtener nombres de features DESPUÉS de ajustar el preprocessor
preprocessor.fit(X_train)  # Ajustar primero para poder obtener feature names
feature_names = get_feature_names_from_preprocessor(preprocessor, X_train)

# Paso 6.2: Transformar los datos
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convertir a DataFrame
X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

# Paso 7: resultados del preprocesamiento 
print("\n" + "="*50)
print("RESULTADOS DEL PREPROCESAMIENTO")
print("="*50)
print(f"X_train preprocesado - Shape: {X_train_processed_df.shape}")
print(f"X_test preprocesado - Shape: {X_test_processed_df.shape}")
print(f"Número de features después del encoding: {X_train_processed_df.shape[1]}")
print(f"Primeras 10 columnas: {list(X_train_processed_df.columns[:10])}")

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
    
    # Pipelines individuales - SIN FUNCIONES LAMBDA
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('to_str', FunctionTransformer(convert_to_str)),  # Cambiado: función definida
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
    
    # Usar named_transformers_ (con guión bajo)
    if hasattr(preprocessor, 'named_transformers_'):
        cat_transformer = preprocessor.named_transformers_['cat']
        if hasattr(cat_transformer, 'named_steps'):
            onehot = cat_transformer.named_steps['onehot']
            if hasattr(onehot, 'get_feature_names_out'):
                cat_features = onehot.get_feature_names_out(cat_cols)
                feature_names.extend(cat_features.tolist())
    
    return feature_names

# Exportar las variables necesarias
__all__ = [
    'ft_engineering', 
    'get_feature_names',
    'X_train', 'X_test', 
    'y_train', 'y_test',
    'preprocessor', 
    'df',
    'X_train_processed_df', 
    'X_test_processed_df'
]

# Renombrar para compatibilidad con código existente
X_train_processed = X_train_processed_df
X_test_processed = X_test_processed_df

# Ejemplo de uso si se ejecuta directamente
if __name__ == "__main__":
    print("\n" + "="*50)
    print("EJECUCIÓN DIRECTA DE ft_engineering.py")
    print("="*50)
    
    # Probar la función ft_engineering
    preprocessor_test = ft_engineering()
    print(f"\nPreprocesador creado exitosamente")
    print(f"✓ Preprocesador ahora es serializable (sin funciones lambda)")
    
    # Simplemente mostrar los nombres de los transformadores configurados
    print(f"Transformadores configurados: ['num', 'cat']")
    
    # Información adicional sobre el preprocesador
    print(f"Número de features numéricas: {len(num_features)}")
    print(f"Número de features categóricas: {len(cat_features)}")
    print(f"Total de features después de encoding: {X_train_processed_df.shape[1]}")
    
    # Mostrar algunos nombres de columnas
    print(f"\nPrimeras 5 columnas procesadas:")
    for i, col in enumerate(X_train_processed_df.columns[:5]):
        print(f"  {i+1}. {col}")
    
    # Probar serialización
    try:
        import pickle
        # Intentar serializar el preprocessor
        with open('test_preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"\n✓ Preprocesador serializable con pickle")
        
        # También probar con la función ft_engineering
        test_preprocessor = ft_engineering()
        with open('test_ft_preprocessor.pkl', 'wb') as f:
            pickle.dump(test_preprocessor, f)
        print(f"✓ Preprocesador de ft_engineering() también serializable")
        
    except Exception as e:
        print(f"\n✗ Error serializando preprocessor: {e}")