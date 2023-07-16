import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set Streamlit app title and sidebar
st.title("Linear Regression")
# st.sidebar.header("User Input")

# Generate random data for demonstration
np.random.seed(50)
X = np.random.rand(100, 1) * 100
y = 2 * X + np.random.randn(100, 1) * 400

X.shape, y.shape

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Display evaluation metrics
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")

# Visualization 1: Scatter plot of the data
fig1 = px.scatter(x=X.flatten(),
                  y=y.flatten(), 
                  labels={"x": "X", 
                          "y": "y"}, 
                  title="Scatter Plot of Data")
st.plotly_chart(fig1)

# Visualization 2: Regression line and data points
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=X.flatten(),
                          y=y.flatten(),
                          mode='markers', 
                          name='Data'))
fig2.add_trace(go.Scatter(x=X.flatten(), 
                          y=lr.predict(X).
                          flatten(), 
                          mode='lines',
                          name='Regression Line'))
fig2.update_layout(xaxis_title="X",
                   yaxis_title="y",
                   title="Linear Regression")
st.plotly_chart(fig2)

# Visualization 3: Residual plot
residuals = y_test - y_pred
fig3 = px.scatter(x=y_pred.flatten(), y=residuals.flatten(), labels={"x": "Predicted y", "y": "Residuals"},
                  title="Residual Plot")
fig3.update_layout(showlegend=False)
fig3.add_hline(y=0, line_dash="dash", line_color="red")
st.plotly_chart(fig3)
