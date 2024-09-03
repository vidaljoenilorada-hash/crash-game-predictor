import pandas as pd
import random
import os
import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.wrappers.scikit_learn import KerasRegressor

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Multiplier(Crash)'] = pd.to_numeric(data['Multiplier(Crash)'], errors='coerce')
    multiplier_data = data['Multiplier(Crash)'].dropna()
    return data, multiplier_data

def monte_carlo_simulation(multiplier_data, num_simulations=10000):
    mean_multiplier = multiplier_data.mean()
    std_multiplier = multiplier_data.std()
    simulated_values = [random.gauss(mean_multiplier, std_multiplier) for _ in range(num_simulations)]
    predicted_crash = sum(simulated_values) / num_simulations
    accuracy = min(max((1 / std_multiplier) * 100, 0), 100)
    return predicted_crash, accuracy

def linear_regression_prediction(multiplier_data):
    x = np.array(range(len(multiplier_data))).reshape(-1, 1)
    y = multiplier_data.values
    model = LinearRegression()
    model.fit(x, y)
    next_value = model.predict(np.array([[len(multiplier_data)]]))
    accuracy = model.score(x, y) * 100
    return next_value[0], accuracy

def random_forest_prediction(multiplier_data):
    x = np.array(range(len(multiplier_data))).reshape(-1, 1)
    y = multiplier_data.values
    model = RandomForestRegressor(n_estimators=100)
    model.fit(x, y)
    next_value = model.predict(np.array([[len(multiplier_data)]]))
    accuracy = model.score(x, y) * 100
    return next_value[0], accuracy

def svr_prediction(multiplier_data):
    x = np.array(range(len(multiplier_data))).reshape(-1, 1)
    y = multiplier_data.values
    model = SVR(kernel='rbf')
    model.fit(x, y)
    next_value = model.predict(np.array([[len(multiplier_data)]]))
    accuracy = model.score(x, y) * 100
    return next_value[0], accuracy

def simple_moving_average_prediction(multiplier_data, window_size=5):
    if len(multiplier_data) < window_size:
        return multiplier_data.mean(), 50.0
    sma = multiplier_data.rolling(window=window_size).mean().iloc[-1]
    accuracy = 100.0 if sma else 50.0
    return sma, accuracy

def exponential_moving_average_prediction(multiplier_data, span=5):
    if len(multiplier_data) < span:
        return multiplier_data.mean(), 50.0
    ema = multiplier_data.ewm(span=span).mean().iloc[-1]
    accuracy = 100.0 if ema else 50.0
    return ema, accuracy

def polynomial_regression_prediction(multiplier_data, degree=2):
    x = np.array(range(len(multiplier_data))).reshape(-1, 1)
    y = multiplier_data.values
    poly_features = np.vander(x.flatten(), degree + 1)
    model = LinearRegression()
    model.fit(poly_features, y)
    next_value = model.predict(np.vander([len(multiplier_data)], degree + 1))
    accuracy = model.score(poly_features, y) * 100
    return next_value[0], accuracy

def decision_tree_prediction(multiplier_data):
    x = np.array(range(len(multiplier_data))).reshape(-1, 1)
    y = multiplier_data.values
    model = DecisionTreeRegressor()
    model.fit(x, y)
    next_value = model.predict(np.array([[len(multiplier_data)]]))
    accuracy = model.score(x, y) * 100
    return next_value[0], accuracy

def gradient_boosting_prediction(multiplier_data):
    x = np.array(range(len(multiplier_data))).reshape(-1, 1)
    y = multiplier_data.values
    model = GradientBoostingRegressor()
    model.fit(x, y)
    next_value = model.predict(np.array([[len(multiplier_data)]]))
    accuracy = model.score(x, y) * 100
    return next_value[0], accuracy

def knn_prediction(multiplier_data, neighbors=5):
    x = np.array(range(len(multiplier_data))).reshape(-1, 1)
    y = multiplier_data.values
    model = KNeighborsRegressor(n_neighbors=neighbors)
    model.fit(x, y)
    next_value = model.predict(np.array([[len(multiplier_data)]]))
    accuracy = model.score(x, y) * 100
    return next_value[0], accuracy

def ridge_regression_prediction(multiplier_data):
    x = np.array(range(len(multiplier_data))).reshape(-1, 1)
    y = multiplier_data.values
    model = Ridge()
    model.fit(x, y)
    next_value = model.predict(np.array([[len(multiplier_data)]]))
    accuracy = model.score(x, y) * 100
    return next_value[0], accuracy

def lasso_regression_prediction(multiplier_data):
    x = np.array(range(len(multiplier_data))).reshape(-1, 1)
    y = multiplier_data.values
    model = Lasso()
    model.fit(x, y)
    next_value = model.predict(np.array([[len(multiplier_data)]]))
    accuracy = model.score(x, y) * 100
    return next_value[0], accuracy

def elastic_net_prediction(multiplier_data):
    x = np.array(range(len(multiplier_data))).reshape(-1, 1)
    y = multiplier_data.values
    model = ElasticNet()
    model.fit(x, y)
    next_value = model.predict(np.array([[len(multiplier_data)]]))
    accuracy = model.score(x, y) * 100
    return next_value[0], accuracy

def bayesian_ridge_prediction(multiplier_data):
    x = np.array(range(len(multiplier_data))).reshape(-1, 1)
    y = multiplier_data.values
    model = BayesianRidge()
    model.fit(x, y)
    next_value = model.predict(np.array([[len(multiplier_data)]]))
    accuracy = model.score(x, y) * 100
    return next_value[0], accuracy

def arima_prediction(multiplier_data):
    model = ARIMA(multiplier_data, order=(5, 1, 0))
    model_fit = model.fit()
    next_value = model_fit.forecast()[0]
    accuracy = 100  # Placeholder for ARIMA accuracy calculation
    return next_value, accuracy

def lstm_prediction(multiplier_data):
    x = np.array(range(len(multiplier_data))).reshape(-1, 1)
    y = multiplier_data.values

    model = Sequential()
    model.add(LSTM(50, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    x_lstm = x.reshape((x.shape[0], 1, 1))
    model.fit(x_lstm, y, epochs=10, batch_size=1, verbose=0)

    next_value = model.predict(np.array([[[len(multiplier_data)]]]))[0][0]
    accuracy = 100  # Placeholder for LSTM accuracy calculation
    return next_value, accuracy

def xgboost_prediction(multiplier_data):
    x = np.array(range(len(multiplier_data))).reshape(-1, 1)
    y = multiplier_data.values
    model = xgb.XGBRegressor()
    model.fit(x, y)
    next_value = model.predict(np.array([[len(multiplier_data)]]))
    accuracy = model.score(x, y) * 100
    return next_value[0], accuracy

def lightgbm_prediction(multiplier_data):
    x = np.array(range(len(multiplier_data))).reshape(-1, 1)
    y = multiplier_data.values
    model = lgb.LGBMRegressor()
    model.fit(x, y)
    next_value = model.predict(np.array([[len(multiplier_data)]]))
    accuracy = model.score(x, y) * 100
    return next_value[0], accuracy

def catboost_prediction(multiplier_data):
    x = np.array(range(len(multiplier_data))).reshape(-1, 1)
    y = multiplier_data.values
    model = cb.CatBoostRegressor(verbose=0)
    model.fit(x, y)
    next_value = model.predict(np.array([[len(multiplier_data)]]))
    accuracy = model.score(x, y) * 100
    return next_value[0], accuracy

def update_crash():
    try:
        _, multiplier_data = load_data(file_path)
        predictions = []

        # Check selected models and calculate predictions
        if monte_carlo_var.get():
            crash, acc = monte_carlo_simulation(multiplier_data)
            predictions.append(f"Monte Carlo: {crash:.2f}x ({acc:.1f}% accurate)")

        if linear_regression_var.get():
            crash, acc = linear_regression_prediction(multiplier_data)
            predictions.append(f"Linear Regression: {crash:.2f}x ({acc:.1f}% accurate)")

        if random_forest_var.get():
            crash, acc = random_forest_prediction(multiplier_data)
            predictions.append(f"Random Forest: {crash:.2f}x ({acc:.1f}% accurate)")

        if svr_var.get():
            crash, acc = svr_prediction(multiplier_data)
            predictions.append(f"Support Vector Regression (SVR): {crash:.2f}x ({acc:.1f}% accurate)")

        if sma_var.get():
            crash, acc = simple_moving_average_prediction(multiplier_data)
            predictions.append(f"Simple Moving Average: {crash:.2f}x ({acc:.1f}% accurate)")

        if ema_var.get():
            crash, acc = exponential_moving_average_prediction(multiplier_data)
            predictions.append(f"Exponential Moving Average: {crash:.2f}x ({acc:.1f}% accurate)")

        if polynomial_var.get():
            crash, acc = polynomial_regression_prediction(multiplier_data)
            predictions.append(f"Polynomial Regression: {crash:.2f}x ({acc:.1f}% accurate)")

        if decision_tree_var.get():
            crash, acc = decision_tree_prediction(multiplier_data)
            predictions.append(f"Decision Tree: {crash:.2f}x ({acc:.1f}% accurate)")

        if gradient_boosting_var.get():
            crash, acc = gradient_boosting_prediction(multiplier_data)
            predictions.append(f"Gradient Boosting: {crash:.2f}x ({acc:.1f}% accurate)")

        if knn_var.get():
            crash, acc = knn_prediction(multiplier_data)
            predictions.append(f"K-Nearest Neighbors (KNN): {crash:.2f}x ({acc:.1f}% accurate)")

        if ridge_var.get():
            crash, acc = ridge_regression_prediction(multiplier_data)
            predictions.append(f"Ridge Regression: {crash:.2f}x ({acc:.1f}% accurate)")

        if lasso_var.get():
            crash, acc = lasso_regression_prediction(multiplier_data)
            predictions.append(f"Lasso Regression: {crash:.2f}x ({acc:.1f}% accurate)")

        if elastic_net_var.get():
            crash, acc = elastic_net_prediction(multiplier_data)
            predictions.append(f"Elastic Net Regression: {crash:.2f}x ({acc:.1f}% accurate)")

        if bayesian_ridge_var.get():
            crash, acc = bayesian_ridge_prediction(multiplier_data)
            predictions.append(f"Bayesian Ridge Regression: {crash:.2f}x ({acc:.1f}% accurate)")

        if arima_var.get():
            crash, acc = arima_prediction(multiplier_data)
            predictions.append(f"ARIMA: {crash:.2f}x ({acc:.1f}% accurate)")

        if lstm_var.get():
            crash, acc = lstm_prediction(multiplier_data)
            predictions.append(f"LSTM Neural Network: {crash:.2f}x ({acc:.1f}% accurate)")

        if xgboost_var.get():
            crash, acc = xgboost_prediction(multiplier_data)
            predictions.append(f"XGBoost Regression: {crash:.2f}x ({acc:.1f}% accurate)")

        if lightgbm_var.get():
            crash, acc = lightgbm_prediction(multiplier_data)
            predictions.append(f"LightGBM Regression: {crash:.2f}x ({acc:.1f}% accurate)")

        if catboost_var.get():
            crash, acc = catboost_prediction(multiplier_data)
            predictions.append(f"CatBoost Regression: {crash:.2f}x ({acc:.1f}% accurate)")

        # Show results in a message box
        messagebox.showinfo("Prediction", "\n".join(predictions))

    except Exception as e:
        messagebox.showerror("Error", str(e))

def preview_csv():
    try:
        data, _ = load_data(file_path)
        preview_window = tk.Toplevel()
        preview_window.title("CSV Data Preview")

        frame = tk.Frame(preview_window)
        frame.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(frame)
        tree["columns"] = list(data.columns)
        tree["show"] = "headings"
        
        for col in data.columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center")

        for index, row in data.iterrows():
            tree.insert("", "end", values=list(row))

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill='y')
        tree.configure(yscrollcommand=scrollbar.set)
        
    except Exception as e:
        messagebox.showerror("Error", str(e))

def create_gui():
    window = tk.Tk()
    window.title("Crash Predictor")
    window.geometry("400x900")
    window.configure(bg="#1a1a2e")

    title_label = tk.Label(window, text="Crash Predictor", font=("Arial", 20, "bold"), fg="#e94560", bg="#1a1a2e")
    title_label.pack(pady=20)

    # Global variables for all model checkboxes
    global monte_carlo_var, linear_regression_var, random_forest_var, svr_var, sma_var
    global ema_var, polynomial_var, decision_tree_var, gradient_boosting_var, knn_var
    global ridge_var, lasso_var, elastic_net_var, bayesian_ridge_var, arima_var
    global lstm_var, xgboost_var, lightgbm_var, catboost_var

    # Initialize the checkbox variables
    monte_carlo_var = tk.BooleanVar(value=True)
    linear_regression_var = tk.BooleanVar(value=True)
    random_forest_var = tk.BooleanVar(value=True)
    svr_var = tk.BooleanVar(value=True)
    sma_var = tk.BooleanVar(value=True)
    ema_var = tk.BooleanVar(value=True)
    polynomial_var = tk.BooleanVar(value=True)
    decision_tree_var = tk.BooleanVar(value=True)
    gradient_boosting_var = tk.BooleanVar(value=True)
    knn_var = tk.BooleanVar(value=True)
    ridge_var = tk.BooleanVar(value=True)
    lasso_var = tk.BooleanVar(value=True)
    elastic_net_var = tk.BooleanVar(value=True)
    bayesian_ridge_var = tk.BooleanVar(value=True)
    arima_var = tk.BooleanVar(value=True)
    lstm_var = tk.BooleanVar(value=True)
    xgboost_var = tk.BooleanVar(value=True)
    lightgbm_var = tk.BooleanVar(value=True)
    catboost_var = tk.BooleanVar(value=True)

    # Create checkboxes for each model
    tk.Checkbutton(window, text="Monte Carlo", variable=monte_carlo_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="Linear Regression", variable=linear_regression_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="Random Forest", variable=random_forest_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="Support Vector Regression (SVR)", variable=svr_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="Simple Moving Average", variable=sma_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="Exponential Moving Average", variable=ema_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="Polynomial Regression", variable=polynomial_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="Decision Tree", variable=decision_tree_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="Gradient Boosting", variable=gradient_boosting_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="K-Nearest Neighbors (KNN)", variable=knn_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="Ridge Regression", variable=ridge_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="Lasso Regression", variable=lasso_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="Elastic Net Regression", variable=elastic_net_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="Bayesian Ridge Regression", variable=bayesian_ridge_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="ARIMA", variable=arima_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="LSTM Neural Network", variable=lstm_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="XGBoost Regression", variable=xgboost_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="LightGBM Regression", variable=lightgbm_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)
    tk.Checkbutton(window, text="CatBoost Regression", variable=catboost_var, bg="#1a1a2e", fg="white", font=("Arial", 10)).pack(anchor='w', padx=20)

    # Button to predict next crash value
    next_button = tk.Button(window, text="Predict Next Crash", command=update_crash, width=20, height=2, bg="#16213e", fg="#ffffff", font=("Arial", 12, "bold"))
    next_button.pack(pady=10)

    # Button to preview CSV data
    preview_button = tk.Button(window, text="Preview CSV", command=preview_csv, width=20, height=2, bg="#16213e", fg="#ffffff", font=("Arial", 12, "bold"))
    preview_button.pack(pady=10)

    # Button to exit the application
    exit_button = tk.Button(window, text="Exit", command=window.quit, width=20, height=2, bg="#16213e", fg="#ffffff", font=("Arial", 12, "bold"))
    exit_button.pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), 'data.csv')
    if not os.path.isfile(file_path):
        print(f"Error: {file_path} not found. Please ensure 'data.csv' is in the same directory as this script.")
    else:
        create_gui()


