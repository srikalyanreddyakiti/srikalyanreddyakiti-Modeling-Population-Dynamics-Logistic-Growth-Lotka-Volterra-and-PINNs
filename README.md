📊 Modeling Population Dynamics using Differential Equations and PINNs
This project explores mathematical modeling techniques to simulate and analyze real-world ecological systems such as fish populations and predator-prey interactions, along with solving differential equations using modern deep learning methods like Physics-Informed Neural Networks (PINNs).

👨‍🎓 Author
Sri Kalyan Reddy Akiti
Department of Data Science
Texas A&M University – Corpus Christi, USA
📧 sakiti@islander.tamucc.edu

📘 Topics Covered
This project investigates three population modeling techniques:

Logistic Growth with Harvesting
Simulates fish population dynamics under natural resource limits and harvesting rates. Helps determine safe harvesting thresholds to prevent extinction.

Predator-Prey Dynamics (Lotka-Volterra Model)
Models cyclic population behavior between predators and prey (e.g., foxes and rabbits), showing their interdependence and ecological balance.

Solving ODEs using PINNs (Physics-Informed Neural Networks)
Applies deep learning to solve the differential equation y′ + 2xy = 0 using neural networks constrained by physical laws. PINNs are a modern alternative to traditional numerical solvers.

🧪 Experiments & Methodology
1. Logistic Growth with Harvesting
Equation Used:
dN/dt = r * N * (1 - N/K) - H

Parameters:
r = 0.5 (growth rate), K = 2×10⁶ (carrying capacity), H (harvesting rate)

Numerical Method: Forward Euler

Result: Visualized how population size decreases with increased harvesting. Critical point shows extinction under high exploitation.

2. Lotka-Volterra Predator-Prey Model
Equations Used:
du/dt = a * u - b * u * v
dv/dt = -c * v + d * b * u * v

Initial Conditions: Prey (u₀ = 10), Predators (v₀ = 5)

Visualization:

Population over time

Phase-space plots

Predator-to-prey ratio

Result: Demonstrated natural population oscillations, dependence of predator growth on prey abundance, and balance of ecosystems.

3. Solving ODE using PINNs
Equation: y′ + 2xy = 0

PINN Architecture:

2 hidden layers, 20 neurons each

Tanh activation

Loss = Residual Loss + Initial Condition Loss

Optimizer: Adam (0.001 learning rate, 10,000 epochs)

Result: Achieved a solution that matches the analytical solution y = e^(-x²) with high accuracy.

🛠 Tools & Libraries Used
Python

NumPy

Matplotlib

SciPy

TensorFlow (for PINNs)

🔍 Conclusion
This project shows how differential equations, both classical and modern, can help us:

Simulate real-world ecological behaviors

Make smarter decisions in conservation and resource harvesting

Use deep learning (PINNs) to solve complex mathematical problems in place of traditional solvers

By combining classic models with modern computational tools, we gain powerful insights into population dynamics and unlock new possibilities in ecological and mathematical modeling.

📂 Future Work
Extend PINN models to PDEs (partial differential equations)

Apply to real-world datasets like wildlife tracking or climate-affected systems

Integrate interactive visualizations for public-facing educational tools
