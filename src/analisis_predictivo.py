import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Cargar los datos
file_path = '../datos/data (1).xlsx'
data = pd.read_excel(file_path)

# Seleccionar y renombrar las columnas de interés
data = data[['month_year', 'SAP', 'Consumo Total']]
data = data.rename(columns={'Consumo Total': 'Consumo_Total'})

# Convertir 'month_year' a formato de fecha
data['month_year'] = pd.to_datetime(data['month_year'], format='%b-%y')

# Filtrar por el producto SAP específico
sap_code = 'A18130005688'
data_sap = data[data['SAP'] == sap_code].copy()

# Filtrar por el rango de fechas especificado
start_date_filter = '2022-01-01'
end_date_filter = '2025-10-31'
data_sap = data_sap[(data_sap['month_year'] >= start_date_filter) & (data_sap['month_year'] <= end_date_filter)]


# Agrupar por mes y sumar el consumo para manejar duplicados
data_sap = data_sap.groupby('month_year')['Consumo_Total'].sum().reset_index()
data_sap.set_index('month_year', inplace=True)

# Crear una serie de tiempo continua rellenando los meses faltantes
if not data_sap.empty:
    start_date = data_sap.index.min()
    end_date = data_sap.index.max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    data_sap = data_sap.reindex(date_range, fill_value=0)
else:
    print(f"No se encontraron datos para el SAP {sap_code} en el rango de fechas especificado.")
    exit()

# Crear variables de estacionalidad
data_sap['mes'] = data_sap.index.month
data_sap['temporada_pesca'] = data_sap['mes'].apply(lambda x: 1 if x in [4, 5, 6, 11, 12] else 0)
data_sap['pretemporada'] = data_sap['mes'].apply(lambda x: 1 if x in [2, 3, 9, 10] else 0)

# Crear variables de lags (de 1 a 12 meses)
for i in range(1, 13):
    data_sap[f'lag_{i}'] = data_sap['Consumo_Total'].shift(i)

# Crear variables de rolling means (promedio móvil de 3 y 6 meses)
data_sap['rolling_mean_3'] = data_sap['Consumo_Total'].shift(1).rolling(window=3).mean()
data_sap['rolling_mean_6'] = data_sap['Consumo_Total'].shift(1).rolling(window=6).mean()

# Eliminar filas con valores NaN generados por los lags y rolling means
data_sap.dropna(inplace=True)

# Dividir los datos en conjunto de entrenamiento y prueba
num_samples = len(data_sap)
train_size = int(num_samples * 0.8)
train_data = data_sap.iloc[:train_size]
test_data = data_sap.iloc[train_size:]

# Definir características y objetivo
features = ['mes', 'temporada_pesca', 'pretemporada'] + [f'lag_{i}' for i in range(1, 13)] + ['rolling_mean_3', 'rolling_mean_6']
target = 'Consumo_Total'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# --- Modelado ---
results = {}

# Modelo SARIMAX
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(y_train,
                          exog=X_train,
                          order=param,
                          seasonal_order=param_seasonal,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            results_sarimax = mod.fit(disp=False)
            if results_sarimax.aic < best_aic:
                best_aic = results_sarimax.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
        except:
            continue

best_mod = SARIMAX(y_train,
                   exog=X_train,
                   order=best_pdq,
                   seasonal_order=best_seasonal_pdq,
                   enforce_stationarity=False,
                   enforce_invertibility=False)
best_fit = best_mod.fit(disp=False)

# Gráficos de diagnóstico de SARIMAX (comentado por insuficiencia de datos)
# best_fit.plot_diagnostics(figsize=(15, 12))
# plt.savefig('../resultados/sarimax_diagnostics.png')

train_pred_sarimax = best_fit.predict(start=y_train.index[0], end=y_train.index[-1], exog=X_train)
test_pred_sarimax = best_fit.predict(start=y_test.index[0], end=y_test.index[-1], exog=X_test)

results['SARIMAX'] = {
    'train_pred': train_pred_sarimax,
    'test_pred': test_pred_sarimax,
    'metrics': {
        'train_mae': mean_absolute_error(y_train, train_pred_sarimax),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred_sarimax)),
        'test_mae': mean_absolute_error(y_test, test_pred_sarimax),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred_sarimax))
    }
}

# Modelo XGBoost
xgb = XGBRegressor(objective='reg:squarederror', n_estimators=1000, early_stopping_rounds=50)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Gráfico de importancia de características de XGBoost
plt.figure(figsize=(10, 8))
plt.barh(features, xgb.feature_importances_)
plt.xlabel("Importancia de la Característica")
plt.ylabel("Característica")
plt.title("Importancia de la Característica en XGBoost")
plt.savefig('../resultados/xgb_feature_importance.png')


train_pred_xgb = xgb.predict(X_train)
test_pred_xgb = xgb.predict(X_test)

results['XGBoost'] = {
    'train_pred': pd.Series(train_pred_xgb, index=y_train.index),
    'test_pred': pd.Series(test_pred_xgb, index=y_test.index),
    'metrics': {
        'train_mae': mean_absolute_error(y_train, train_pred_xgb),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred_xgb)),
        'test_mae': mean_absolute_error(y_test, test_pred_xgb),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred_xgb))
    }
}

# Modelo Prophet
prophet_train_df = pd.DataFrame({'ds': train_data.index, 'y': y_train, 'temporada_pesca': X_train['temporada_pesca'], 'pretemporada': X_train['pretemporada']})
m = Prophet()
m.add_regressor('temporada_pesca')
m.add_regressor('pretemporada')
m.fit(prophet_train_df)

# Gráfico de componentes de Prophet
fig = m.plot_components(m.predict(prophet_train_df))
fig.savefig('../resultados/prophet_components.png')

train_future = m.make_future_dataframe(periods=0, freq='MS')
train_future = pd.merge(train_future, prophet_train_df[['ds', 'temporada_pesca', 'pretemporada']], on='ds')
train_pred_prophet_df = m.predict(train_future)
prophet_test_df = pd.DataFrame({'ds': test_data.index, 'y': y_test, 'temporada_pesca': X_test['temporada_pesca'], 'pretemporada': X_test['pretemporada']})
test_future = prophet_test_df[['ds', 'temporada_pesca', 'pretemporada']]
test_pred_prophet_df = m.predict(test_future)
train_pred_prophet = pd.Series(train_pred_prophet_df['yhat'].values, index=y_train.index)
test_pred_prophet = pd.Series(test_pred_prophet_df['yhat'].values, index=y_test.index)
results['Prophet'] = {
    'train_pred': train_pred_prophet,
    'test_pred': test_pred_prophet,
    'metrics': {
        'train_mae': mean_absolute_error(y_train, train_pred_prophet),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred_prophet)),
        'test_mae': mean_absolute_error(y_test, test_pred_prophet),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred_prophet))
    }
}


# --- Predicción futura ---
future_dates = pd.date_range(start=data_sap.index.max() + pd.DateOffset(months=1), end='2026-12-31', freq='MS')
future_df = pd.DataFrame(index=future_dates)
future_df['mes'] = future_df.index.month
future_df['temporada_pesca'] = future_df['mes'].apply(lambda x: 1 if x in [4, 5, 6, 11, 12] else 0)
future_df['pretemporada'] = future_df['mes'].apply(lambda x: 1 if x in [2, 3, 9, 10] else 0)

# Crear lags y rolling means para el futuro de forma iterativa
full_history_sarimax = pd.concat([data_sap['Consumo_Total'], pd.Series(index=future_dates, dtype=float)])
future_features_iterative_sarimax = pd.DataFrame(index=future_dates, columns=features, dtype=float)
for date in future_dates:
    for i in range(1, 13):
        future_features_iterative_sarimax.loc[date, f'lag_{i}'] = full_history_sarimax.shift(i)[date]
    future_features_iterative_sarimax.loc[date, 'rolling_mean_3'] = full_history_sarimax.shift(1).rolling(window=3).mean()[date]
    future_features_iterative_sarimax.loc[date, 'rolling_mean_6'] = full_history_sarimax.shift(1).rolling(window=6).mean()[date]
    future_features_iterative_sarimax.loc[date, ['mes', 'temporada_pesca', 'pretemporada']] = future_df.loc[date, ['mes', 'temporada_pesca', 'pretemporada']]
    pred_sarimax = best_fit.get_forecast(steps=1, exog=future_features_iterative_sarimax.loc[[date]].astype(float)).predicted_mean[0]
    full_history_sarimax[date] = pred_sarimax
future_pred_sarimax = full_history_sarimax.loc[future_dates]
sarimax_forecast = best_fit.get_forecast(steps=len(future_dates), exog=future_features_iterative_sarimax[X_train.columns].astype(float))
future_pred_sarimax_ci = sarimax_forecast.conf_int()


full_history_xgb = pd.concat([data_sap['Consumo_Total'], pd.Series(index=future_dates, dtype=float)])
future_features_iterative_xgb = pd.DataFrame(index=future_dates, columns=features, dtype=float)
for date in future_dates:
    for i in range(1, 13):
        future_features_iterative_xgb.loc[date, f'lag_{i}'] = full_history_xgb.shift(i)[date]
    future_features_iterative_xgb.loc[date, 'rolling_mean_3'] = full_history_xgb.shift(1).rolling(window=3).mean()[date]
    future_features_iterative_xgb.loc[date, 'rolling_mean_6'] = full_history_xgb.shift(1).rolling(window=6).mean()[date]
    future_features_iterative_xgb.loc[date, ['mes', 'temporada_pesca', 'pretemporada']] = future_df.loc[date, ['mes', 'temporada_pesca', 'pretemporada']]
    pred_xgb = xgb.predict(future_features_iterative_xgb.loc[[date]].astype(float))[0]
    full_history_xgb[date] = pred_xgb
future_pred_xgb = full_history_xgb.loc[future_dates]

prophet_future = future_df.reset_index().rename(columns={'index': 'ds'})
future_pred_prophet_df = m.predict(prophet_future)
future_pred_prophet = pd.Series(future_pred_prophet_df['yhat'].values, index=future_dates)
future_pred_prophet_ci = future_pred_prophet_df[['yhat_lower', 'yhat_upper']]

# --- Visualización ---
plt.figure(figsize=(15, 8))
plt.plot(data_sap.index, data_sap['Consumo_Total'], label='Datos reales')
for model_name, result in results.items():
    plt.plot(result['test_pred'].index, result['test_pred'], label=f'Predicción de prueba ({model_name})')

plt.plot(future_pred_sarimax.index, future_pred_sarimax, label='Predicción futura (SARIMAX)', linestyle='--')
plt.fill_between(future_pred_sarimax.index, future_pred_sarimax_ci.iloc[:, 0], future_pred_sarimax_ci.iloc[:, 1], alpha=0.2, label='Intervalo de confianza (SARIMAX)')

plt.plot(future_pred_xgb.index, future_pred_xgb, label='Predicción futura (XGBoost)', linestyle='--')

plt.plot(future_pred_prophet.index, future_pred_prophet, label='Predicción futura (Prophet)', linestyle='--')
plt.fill_between(future_pred_prophet.index, future_pred_prophet_ci['yhat_lower'], future_pred_prophet_ci['yhat_upper'], alpha=0.2, label='Intervalo de confianza (Prophet)')


plt.title(f'Consumo mensual del producto {sap_code}')
plt.xlabel('Fecha')
plt.ylabel('Consumo Total')
plt.legend()
plt.grid(True)
plt.savefig('../resultados/predicciones.png')

# --- Generación de reporte en Excel ---
with pd.ExcelWriter('../resultados/reporte_predicciones.xlsx') as writer:
    # Pestaña de Métricas
    metrics_df = pd.DataFrame({
        'Modelo': ['SARIMAX', 'XGBoost', 'Prophet'],
        'MAE (Train)': [results['SARIMAX']['metrics']['train_mae'], results['XGBoost']['metrics']['train_mae'], results['Prophet']['metrics']['train_mae']],
        'RMSE (Train)': [results['SARIMAX']['metrics']['train_rmse'], results['XGBoost']['metrics']['train_rmse'], results['Prophet']['metrics']['train_rmse']],
        'MAE (Test)': [results['SARIMAX']['metrics']['test_mae'], results['XGBoost']['metrics']['test_mae'], results['Prophet']['metrics']['test_mae']],
        'RMSE (Test)': [results['SARIMAX']['metrics']['test_rmse'], results['XGBoost']['metrics']['test_rmse'], results['Prophet']['metrics']['test_rmse']]
    }).set_index('Modelo')
    metrics_df.to_excel(writer, sheet_name='Metricas_Modelos')

    # Pestaña de Predicción Final
    best_model_name = metrics_df['RMSE (Test)'].idxmin()
    if best_model_name == 'Prophet':
        prediction_df = pd.DataFrame({
            'fecha': future_pred_prophet.index,
            'prediccion': future_pred_prophet.values,
            'limite_inferior_ci': future_pred_prophet_ci['yhat_lower'].values,
            'limite_superior_ci': future_pred_prophet_ci['yhat_upper'].values
        })
    elif best_model_name == 'SARIMAX':
        prediction_df = pd.DataFrame({
            'fecha': future_pred_sarimax.index,
            'prediccion': future_pred_sarimax.values,
            'limite_inferior_ci': future_pred_sarimax_ci.iloc[:, 0].values,
            'limite_superior_ci': future_pred_sarimax_ci.iloc[:, 1].values
        })
    else: # XGBoost
        prediction_df = pd.DataFrame({
            'fecha': future_pred_xgb.index,
            'prediccion': future_pred_xgb.values,
            'limite_inferior_ci': np.nan,
            'limite_superior_ci': np.nan
        })
    prediction_df.to_excel(writer, sheet_name='Prediccion_Final_2026', index=False)

    # Pestaña de Validación en Test
    validation_df = pd.DataFrame({
        'fecha': y_test.index,
        'real': y_test.values,
        'pred_sarimax': test_pred_sarimax.values,
        'pred_xgboost': test_pred_xgb,
        'pred_prophet': test_pred_prophet.values
    })
    validation_df.to_excel(writer, sheet_name='Validacion_Test', index=False)

print("\nReporte de predicciones guardado como '../resultados/reporte_predicciones.xlsx'")
