#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
import numpy as np
import math


# In[14]:


x = torch.tensor(3.5,requires_grad = True)
y = x*x
z = 2*y + 3


# In[15]:


z.backward()


# In[16]:


print("Gradient at x = 3.5: ", x.grad)


# In[17]:


#Q1
a = torch.tensor(1.0,requires_grad=True)
b = torch.tensor(2.0,requires_grad=True)
x = 2 * a + 3 * b
y = 5 * a**2 + 3 * b**3
z = 2 * x + 3 * y
z.backward()
print("Gradient dz/da:", a.grad.item())

analytical_gradient = 4 + 30 * a.item()
print("Analytical Gradient dz/da:", analytical_gradient)


# In[18]:


#Q2
x = torch.tensor(1.0,requires_grad=True)
w = torch.tensor(2.0,requires_grad=True)
b = torch.tensor(3.0,requires_grad=True)
u = w*x
v = b + u
a = torch.maximum(torch.tensor(0.0), v) 
a.backward()
print("gradient w is ",w.grad.item())
analytical_gradient = x if v.item() > 0 else 0
print("Analytical Gradient dw:", analytical_gradient.item())


# In[19]:


#Q3
x = torch.tensor(1.0,requires_grad=True)
w = torch.tensor(2.0,requires_grad=True)
b = torch.tensor(3.0,requires_grad=True)
u = w*x
v = b + u
a = torch.sigmoid(v)
a.backward()

print("Gradient da/dw (PyTorch):", w.grad.item())

# Analytical gradient
sigmoid_v = torch.sigmoid(v)
analytical_gradient = sigmoid_v * (1 - sigmoid_v) * x
print("Analytical Gradient da/dw:", analytical_gradient.item())


# In[20]:


#Q4
x = torch.tensor(1.0, requires_grad=True)
f = torch.exp((-1 * x**2) - ( 2 *x ) - torch.sin(x))
f.backward()
print("Gradient ",x.grad.item())
analytical_grad = torch.exp(-x**2 - 2*x - torch.sin(x)) * (-2*x - 2 - torch.cos(x))
print("Analytical Gradient df/dx:", analytical_grad.item())


# In[21]:


#Q5
x = torch.tensor(2.0,requires_grad=True)
y = 8*x**4+3*x**3+7*x**2+6*x+3
y.backward()
print("gradient ", x.grad.item())
analytical_gradient = 32*x**3 + 9*x**2 + 14*x + 6
print("Analytical Gradient:", analytical_gradient.item())


# In[22]:


#Q6
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)
a = 2 * x * z
b = torch.sin(y)
c = a / b
d = 1 + c
e = torch.log(d)
f = torch.tanh(e)
f.backward()
print("Gradient",y.grad.item())
sech_e = torch.cosh(e) ** -2
analytical_gradient = sech_e * (1 / d) * (-2 * x * z * torch.cos(y) / (torch.sin(y) ** 2))

print("Analytical Gradient df/dy:", analytical_gradient.item())


# In[ ]:





# In[ ]:




