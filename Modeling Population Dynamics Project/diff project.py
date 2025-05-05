import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(20, activation='tanh')
        self.hidden2 = tf.keras.layers.Dense(20, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, t):
        x = self.hidden1(t)
        x = self.hidden2(x)
        return self.output_layer(x)

# Define the ODE residual (y' + 2xy = 0)
def residual(model, t):
    with tf.GradientTape() as tape:
        tape.watch(t)
        y = model(t)
        dy_dt = tape.gradient(y, t)
    # Residual of the differential equation
    return dy_dt + 2 * t * y

# Training data
t = np.linspace(0, 2, 100).reshape(-1, 1)  # Time points from 0 to 2
y0 = np.array([[1.0]])  # Initial condition for y(0)

# Convert numpy arrays to tensors
t = tf.convert_to_tensor(t, dtype=tf.float32)
y0 = tf.convert_to_tensor(y0, dtype=tf.float32)

# Define the model and optimizer
model = PINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
for epoch in range(10000):s
    with tf.GradientTape() as tape:
        # Compute the loss (residuals + initial conditions)
        loss_residual = tf.reduce_mean(tf.square(residual(model, t)))
        loss_initial = tf.reduce_mean(tf.square(model(tf.zeros((1, 1), dtype=tf.float32)) - y0))
        loss = loss_residual + loss_initial

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# Predict and plot the solution
t_test = np.linspace(0, 2, 100).reshape(-1, 1).astype(np.float32)
y_pred = model(t_test).numpy()

# Analytical solution for comparison (y = exp(-x^2))
y_analytical = np.exp(-t_test**2)

plt.plot(t_test, y_pred, label='Predicted')
plt.plot(t_test, y_analytical, label='Analytical', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('''PINN Solution of ODE y' + 2xy = 0''')
plt.legend()
plt.show()
