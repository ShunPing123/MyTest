import numpy as np
import matplotlib.pyplot as plt

# 定义横坐标范围
x = np.linspace(-10, 10, 100)

# Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh函数
def tanh(x):
    return np.tanh(x)

# ReLU函数
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU函数
def leaky_relu(x):
    alpha = 0.1
    return np.maximum(alpha*x, x)

# 绘制图像
plt.figure(figsize=(12, 8))

# 绘制Sigmoid函数图像
plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sigmoid Activation')
plt.legend()

# 绘制Tanh函数图像
plt.subplot(2, 2, 2)
plt.plot(x, tanh(x), label='Tanh')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Tanh Activation')
plt.legend()

# 绘制ReLU函数图像
plt.subplot(2, 2, 3)
plt.plot(x, relu(x), label='ReLU')
plt.xlabel('x')
plt.ylabel('y')
plt.title('ReLU Activation')
plt.legend()

# 绘制Leaky ReLU函数图像
plt.subplot(2, 2, 4)
plt.plot(x, leaky_relu(x), label='Leaky ReLU')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Leaky ReLU Activation')
plt.legend()

# 显示图像
plt.tight_layout()
plt.show()
