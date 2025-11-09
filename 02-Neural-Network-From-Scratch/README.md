# Neural Network Implementation From Scratch

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Only-orange.svg)
![DeepLearning](https://img.shields.io/badge/DeepLearning-Fundamentals-red.svg)

## 📋 Project Overview

A complete implementation of a **simple neural network from scratch** using only NumPy, without any deep learning frameworks (TensorFlow, PyTorch, etc.). This project demonstrates a deep understanding of the fundamental mathematics and algorithms behind neural networks.

## 🎯 Objectives

- Implement forward propagation step-by-step
- Code backward propagation with gradient descent
- Understand the mathematical foundations of neural networks
- Create a reusable, object-oriented neural network class

## 🧮 Mathematical Foundation

### Network Architecture

**Input Layer:** x ∈ ℝ^d  
**Hidden Layer:**
- Weights: A₁ ∈ ℝ^(h×d)
- Biases: b₁ ∈ ℝ^(h×1)
- Pre-activation: z₁ = A₁x^T + b₁
- Activation: h = tanh(z₁)

**Output Layer:**
- Weights: A₂ ∈ ℝ^(m×h)
- Biases: b₂ ∈ ℝ^(m×1)
- Pre-activation: z₂ = A₂h + b₂
- Output: ŷ = z₂^T

### Loss Function

Mean Squared Error (MSE):

```
L = (1/n) Σᵢ (ŷᵢ - yᵢ)²
```

### Backpropagation Gradients

1. **Output layer weights:**
   ```
   ∂L/∂A₂ = (∂L/∂z₂) · h₁^T
   ```

2. **Output layer biases:**
   ```
   ∂L/∂b₂ = Σᵢ (∂L/∂z₂)
   ```

3. **Hidden activation:**
   ```
   ∂L/∂h₁ = A₂^T · (∂L/∂z₂)
   ```

4. **Hidden pre-activation:**
   ```
   ∂L/∂z₁ = (∂L/∂h₁) · (1 - tanh²(z₁))
   ```

5. **Hidden layer weights & biases:**
   ```
   ∂L/∂A₁ = (∂L/∂z₁) · x
   ∂L/∂b₁ = Σᵢ (∂L/∂z₁)
   ```

## 🛠️ Technologies Used

- **Python 3.8+**
- **NumPy** (Mathematics & array operations)
- **Matplotlib** (Gradient visualization)

## 🔍 Implementation Details

### Step 1: Parameter Initialization
```python
A1 = np.random.randn(hidden_size, input_size)
b1 = np.zeros((hidden_size, 1))
A2 = np.random.randn(output_size, hidden_size)
b2 = np.zeros((output_size, 1))
```

### Step 2: Forward Propagation
```python
z1 = A1 @ X.T + b1
h1 = tanh(z1)
z2 = A2 @ h1 + b2
y_pred = z2.T
```

### Step 3: Backward Propagation
- Compute gradients for all parameters
- Update weights using gradient descent

### Step 4: Training Loop
- Mini-batch training with batch size flexibility
- Epoch-based training with loss monitoring
- Convergence tracking

## 📊 Features

1. **Modular Functions:** Separate functions for each step
2. **Object-Oriented Design:** Complete `SimpleNN` class
3. **Xavier Initialization:** For better convergence
4. **Mini-Batch Training:** Configurable batch size
5. **Visualization:** Gradient field and 3D function plots

## 📁 Project Structure

```
02-Neural-Network-From-Scratch/
├── Neural_Network_From_Scratch.ipynb  # Main notebook
├── README.md                           # This file
└── requirements.txt                    # Python dependencies
```

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Jupyter notebook:
   ```bash
   jupyter notebook Neural_Network_From_Scratch.ipynb
   ```

3. Follow the step-by-step implementation in the notebook

## 📈 Training Example

```python
# Create network
nn = SimpleNN(input_size=2, hidden_size=5, output_size=1)

# Train
nn.train(X, y, epochs=1000, lr=0.01, batch_size=16)

# Results:
# Epoch 0, Loss: 2.957363
# Epoch 100, Loss: 0.443000
# Epoch 200, Loss: 0.185020
# ...
# Epoch 900, Loss: 0.019042
```

**Excellent convergence!** Loss decreased from 2.96 to 0.019.

## 💡 Key Concepts Demonstrated

1. **Forward Propagation:** How data flows through the network
2. **Activation Functions:** tanh and its derivative
3. **Loss Functions:** MSE for regression tasks
4. **Backpropagation:** Computing gradients efficiently
5. **Gradient Descent:** Parameter optimization
6. **Mini-Batch Training:** Stochastic vs batch gradient descent
7. **Object-Oriented Programming:** Clean, reusable code

## 🎓 Educational Value

This project is perfect for:
- Understanding how neural networks work "under the hood"
- Learning the mathematics of deep learning
- Preparing for interviews on ML fundamentals
- Teaching others about neural network basics

## 🔮 Future Improvements

- Add more activation functions (ReLU, sigmoid, softmax)
- Implement different loss functions
- Add regularization (L1, L2, Dropout)
- Create visualization of decision boundaries
- Implement more complex architectures

## 👨‍💻 Author

**Gabriel Sultan**  
Engineering Student - Data Science & AI  
[GitHub](https://github.com/GabrielSultan) | [LinkedIn](https://www.linkedin.com/in/gabriel-sultan)

## 📝 License

This project is for educational and portfolio purposes.

---

⭐ If this helped you understand neural networks better, please star this project!

