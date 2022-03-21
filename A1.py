#!/usr/bin/env python
# coding: utf-8

# # Assignment A1 [35 marks]
# 
# The assignment consists of 3 questions. Each exercise may contain coding and/or discussion questions.
# - Type your **code** in the **code cells** provided below each question.
# - For **non-code** questions, use the **Markdown cells** provided below each question, indicated by 📝. Double-click these cells to edit them, and run them to display your Markdown-formatted text. Please refer to the Week 1 tutorial notebook for Markdown syntax.

# ---
# ## Question 1: Estimating $\pi$ [8 marks]
# 
# Consider the 3 following formulas:
# 
# $$
# \begin{align}
# (1) \qquad &\prod_{n=1}^\infty \frac{4n^2}{4n^2-1} = \frac{\pi}{2} \\
# (2) \qquad &\sum_{n=0}^\infty \frac{2^n n!^2}{(2n+1)!} = \frac{\pi}{2} \\
# (3) \qquad &\sum_{n=1}^\infty \frac{(-1)^{n+1}}{n(n+1)(2n+1)} = \pi - 3
# \end{align}
# $$
# 
# Each of these formulas can be used to compute the value of $\pi$ to arbitrary precision, by computing as many terms of the partial sum or product as are required.
# 
# **1.1** Compute the sum or product of the first $m$ terms for each formula, with $m = 1, 2, 3, \dots, 30$.
# 
# Present your results graphically, using 2 plots, both with the total number of terms on the x-axis.
# 
# - The first plot should display the value of the partial sum or product for different numbers of terms, and clearly indicate the exact value of $\pi$ for reference.
# - The second plot should display the absolute value of the error between the partial sum or product and $\pi$, with the y-axis set to logarithmic scale.
# 
# **[5 marks]**

# In[2]:


import numpy as np
import math
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

#Formula 1 partial product:
product = np.ones(30)
product[0] = 4/3
for m in range(2,31):
    product[m-1] = product[m-2] * (4 * m ** 2) / (4 * m ** 2 - 1)
print(product[-1])    


#Formula 2 partial sums:
sum1 = np.ones(30)
sum1[0] = 1
for n in range(1,30):
    sum1[n] = sum1[n-1] + (2 ** n * (math.factorial(n)) ** 2) / math.factorial(2 * n + 1)
print(sum1[-1])

#Formula 3 partial sums:
sum2 = np.ones(30)
sum2[0] = 1/6
for i in range(2, 31):
    sum2[i-1] = sum2[i-2] + (-1) ** (i+1) / (i * (i+1) * (2*i+1))
print(sum2[-1])


x = np.arange(1,31)
plt.figure(1)
plt.plot(x, 2 * product, 'g-', label='formula (1)')
plt.plot(x, 2 * sum1, 'r-', label='fomula (2)')
plt.plot(x, sum2 + 3, 'b-', label='formula (3)')
plt.axhline(y=np.pi, color='c', linestyle='-', label='pi')
plt.xlabel('number of terms')
plt.ylabel('value of product, sums')
plt.legend(loc='lower right', fontsize=12)
plt.show()

error_product = abs(30 * [np.pi] - 2 * product)
error_product_log = np.log(error_product)
error_sum1 = abs(30 * [np.pi] - 2 * sum1)
error_sum1_log = np.log(error_sum1)
error_sum2 = abs(30 * [np.pi] - (sum2 + 3))
error_sum2_log = np.log(error_sum2)
plt.figure(2)
plt.plot(x, abs(error_product_log), '-', label='error of (1)')
plt.plot(x, abs(error_sum1_log), '-', label='error of (2)')
plt.plot(x, abs(error_sum2_log), '-', label='error of (3)')
plt.xlabel('number of terms')
plt.ylabel('error(log scale)')
plt.legend(loc='upper right', fontsize=12)
plt.show()


# **1.2** If you did not have access to e.g. `np.pi` or `math.pi`, which of these 3 formulas would you choose to efficiently calculate an approximation of $\pi$ accurate to within any given precision (down to machine accuracy -- i.e. not exceeding $\sim 10^{-16}$)?
# 
# Explain your reasoning in your own words, with reference to your plots from **1.1**, in no more than 200 words.
# 
# **[3 marks]**

# I will use formula (3) as an alternative of np.pi. From the first plot we can see that as we traverse the x axis, the dark blue line is always the nearest to pi. Formula (2) has almost the same level of accuracy as (3) but it is noticeable that it is quite far off from pi when number of terms are small. In contrast, the error between (3) and true value of pi is small at the beginning and it tends to 0 really fast as number of terms increases.

# ---
# ## Question 2: Numerical Linear Algebra [12 marks]
# 
# A **block-diagonal matrix** is a square matrix with square blocks of non-zero values along the diagonal, and zeros elsewhere. For example, the following matrix A is an example of a block-diagonal matrix of size $7\times 7$, with 3 diagonal blocks of size $2\times 2$, $3\times 3$, and $2 \times 2$ respectively:
# 
# $$
# A =
# \begin{pmatrix}
# 1 & 3 & 0 & 0 & 0 & 0 & 0 \\
# 2 & 2 & 0 & 0 & 0 & 0 & 0 \\
# 0 & 0 & -1 & 1 & 2 & 0 & 0 \\
# 0 & 0 & 2 & 1 & 0 & 0 & 0 \\
# 0 & 0 & 4 & 3 & 3 & 0 & 0 \\
# 0 & 0 & 0 & 0 & 0 & 4 & -2 \\
# 0 & 0 & 0 & 0 & 0 & 5 & 3
# \end{pmatrix}.
# $$
# 
# 
# **2.1** The code below creates a block-diagonal matrix with random non-zero values between 0 and 1, where all blocks have identical size. Study the following documentation sections:
# 
# - [`scipy.linalg.block_diag()` - SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.block_diag.html)
# - [`numpy.split()` - NumPy documentation](https://numpy.org/doc/stable/reference/generated/numpy.split.html)
# - [Unpacking Argument Lists - Python tutorial](https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists)
# 
# Then, write detailed code comments in the cell below to explain **each line** of code, in your own words.
# 
# **[3 marks]**

# In[3]:


import numpy as np
from scipy.linalg import block_diag
#import block_diag function from scipy.linalg library.
def random_blocks(m, shape):
    '''
    Returns a list of m random matrices of size shape[0] x shape[1].
    '''
    #create a function that takes two input arguments: an integer m and an 1 x 2 array 'shape'.
    mat = np.random.random([m * shape[0], shape[1]])
    #use the random function to generate a matrix with m * shape[0] rows and shape[1] columns.
    blocks = np.split(mat, m)
    #use the split function in numpy to split the array 'mat' into m equal length sub-arrays. 
    return blocks
    #returns a list of m random matrices of size shape[0] x shape[1].

blocks = random_blocks(4, (3, 2))
#generate a list of 4 random matrices each of size 3 x 2.
A = block_diag(*blocks)
#use the block_diag function to create a block diagonal matrix from provided arrays 'blocks'. 
#That is a matrix with our 4 randomly generated matrices on the diagonal.
print(np.round(A,3))
#print the diagonal matrix with each element round to third decimal places.


# **2.2** For the rest of Question 2, we consider only block-diagonal matrices with $m$ blocks, where all diagonal blocks have the same shape $n \times n$. A block-diagonal system $Ax = b$ can be written as
# 
# $$
# \begin{pmatrix}
# A_{1} & & & \\
# & A_{2} & & \\
# & & \ddots & \\
# & & & A_{m}
# \end{pmatrix}
# \begin{pmatrix}
# x_1 \\ x_2 \\ \vdots \\ x_m
# \end{pmatrix}
# =
# \begin{pmatrix}
# b_1 \\ b_2 \\ \vdots \\ b_m
# \end{pmatrix}
# \qquad \Leftrightarrow \qquad
# \begin{cases}
# A_{1} x_1 &= b_1 \\
# A_{2} x_2 &= b_2 \\
# &\vdots \\
# A_{m} x_m &= b_m
# \end{cases},
# $$
# 
# where $A_i$ is the $i$th diagonal block of $A$, and $x_i$, $b_i$ are blocks of length $n$ of the vectors $x$ and $b$ respectively, for $i=1, 2, \dots, m$. Note that when $m=1$, this is a diagonal system.
# 
# We assume that all diagonal blocks $A_i$ are invertible, and therefore that the matrix $A$ is also invertible.
# 
# Write a function `linsolve_block_diag(blocks, b)` which takes 2 input arguments:
# 
# - `blocks`, a list of length $m$ storing a collection of $n \times n$ NumPy arrays (e.g. as returned by `random_blocks()` from **2.1**) representing the blocks $A_i$,
# - a NumPy vector `b` of length $mn$.
# 
# Your function should solve the block-diagonal system $Ax = b$, by solving **each sub-system $A_i x_i = b_i$ separately**, and return the solution as a NumPy vector `x` of length $mn$. You should choose an appropriate method seen in the course (e.g. `np.linalg.solve()`) to solve each sub-system.
# 
# **[3 marks]**

# In[4]:


def linsolve_block_diag(blocks, b):
    '''
    Solves the block-diagonal system Ax=b,
    where the diagonal blocks are listed in "blocks".
    '''
    b_vec = np.split(b, len(blocks))
    solution = []
    for i in range(0,len(blocks)):
        solution.append(np.linalg.solve(blocks[i], b_vec[i])) 
    return solution 


# **2.3** We now wish to compare the computation time needed to solve a block-diagonal system $Ax = b$ using 2 different methods:
# 
# - solving the sub-systems one at a time, as in **2.2**,
# - solving the full system with a general method, not attempting to take the block-diagonal structure into account.
# 
# Consider block-diagonal systems with block sizes $n = 5, 10, 15, 20$, and a total number $m = 5, 10, 15, \dots, 40$ of blocks. For each combination of $n$ and $m$:
# 
# - Use the function `random_blocks()` from **2.1** to generate a list of $m$ random matrices of size $n\times n$.
# - Use the function `np.random.random()` to generate a random vector `b` of length $mn$.
# - Use your function `linsolve_block_diag()` from **2.2** to solve the system $Ax = b$, where $A$ is a block-diagonal matrix of size $mn \times mn$, with diagonal blocks given by the output of `random_blocks()`. Measure the computation time needed to solve the system.
# - Use the function `block_diag()` from `scipy.linalg` to form a NumPy array `A` of size $mn \times mn$, representing the block-diagonal matrix $A$.
# - Solve the full system $Ax = b$, using the same method you used in **2.2** for each individual sub-system. Measure the computation time needed to solve the system.
# 
# Create 4 plots, one for each value of $n$, to compare the computation time needed to solve $Ax=b$ with both methods, and how this varies with the total size of the system.
# 
# Summarise and discuss your observations in no more than 200 words.
# 
# **[6 marks]**

# In[5]:


import time
#n = 5
m = np.array([5, 10, 15, 20, 25, 30, 35, 40])
t_n_5_s1 = []
t_n_5_s2 = []
for i in range(0,8):
    blocks_n_5 = random_blocks(m[i], (5, 5))
    b_n_5 = np.random.random([m[i]*5])
    linsolve_block_diag(blocks_n_5, b_n_5)
    
    t0 = time.time()
    t_n_5_s1.append(time.time() - t0)
    
    A_n_5 = block_diag(*blocks_n_5)
    np.linalg.solve(A_n_5, b_n_5)

    t1 = time.time()
    t_n_5_s2.append(time.time() - t1)

    
t_n_10_s1 = []
t_n_10_s2 = []
for j in range(0,8):
    blocks_n_10 = random_blocks(m[j], (10, 10))
    b_n_10 = np.random.random([m[j]*10])
    linsolve_block_diag(blocks_n_10, b_n_10)
    
    t0 = time.time()
    t_n_10_s1.append(time.time() - t0)
    
    A_n_10 = block_diag(*blocks_n_10)
    np.linalg.solve(A_n_10, b_n_10)

    t1 = time.time()
    t_n_10_s2.append(time.time() - t1)
    
    
t_n_15_s1 = []
t_n_15_s2 = []
for k in range(0,8):
    blocks_n_15 = random_blocks(m[k], (15, 15))
    b_n_15 = np.random.random([m[k]*15])
    linsolve_block_diag(blocks_n_15, b_n_15)
    
    t0 = time.time()
    t_n_15_s1.append(time.time() - t0)
    
    A_n_15 = block_diag(*blocks_n_15)
    np.linalg.solve(A_n_15, b_n_15)

    t1 = time.time()
    t_n_15_s2.append(time.time() - t1)
    
    
t_n_20_s1 = []
t_n_20_s2 = []
for l in range(0,8):
    blocks_n_20 = random_blocks(m[l], (20, 20))
    b_n_20 = np.random.random([m[l]*20])
    linsolve_block_diag(blocks_n_20, b_n_20)
    
    t0 = time.time()
    t_n_20_s1.append(time.time() - t0)
    
    A_n_20 = block_diag(*blocks_n_20)
    np.linalg.solve(A_n_20, b_n_20)

    t1 = time.time()
    t_n_20_s2.append(time.time() - t1)
    
    
fig_1, ax_1 = plt.subplots(2, 2, figsize=(9,6))
ax_1[0,0].bar(m, t_n_5_s1, color='r', width=1.5, label='solution 1')
ax_1[0,0].bar(m+1.5, t_n_5_s2, color='b', width=1.5, label='solution 2')
ax_1[0,0].set_xlabel('value of m', fontsize=10)
ax_1[0,0].set_ylabel('time (s)', fontsize=10)
ax_1[0,0].legend(loc='upper left', fontsize=5)
ax_1[0,0].title.set_text("n = 5")

ax_1[0,1].bar(m, t_n_10_s1, color='r', width=1.5, label='solution 1')
ax_1[0,1].bar(m+1.5, t_n_10_s2, color='b', width=1.5, label='solution 2')
ax_1[0,1].set_xlabel('value of m', fontsize=10)
ax_1[0,1].set_ylabel('time (s)', fontsize=10)
ax_1[0,1].legend(loc='upper left', fontsize=5)
ax_1[0,1].title.set_text("n = 10")

ax_1[1,0].bar(m, t_n_15_s1, color='r', width=1.5, label='solution 1')
ax_1[1,0].bar(m+1.5, t_n_15_s2, color='b', width=1.5, label='solution 2')
ax_1[1,0].set_xlabel('value of m', fontsize=10)
ax_1[1,0].set_ylabel('time (s)', fontsize=10)
ax_1[1,0].legend(loc='upper left', fontsize=5)
ax_1[1,0].title.set_text("n = 15")

ax_1[1,1].bar(m, t_n_20_s1, color='r', width=1.5, label='solution 1')
ax_1[1,1].bar(m+1.5, t_n_20_s2, color='b', width=1.5, label='solution 2')
ax_1[1,1].set_xlabel('value of m', fontsize=10)
ax_1[1,1].set_ylabel('time (s)', fontsize=10)
ax_1[1,1].legend(loc='upper left', fontsize=5)
ax_1[1,1].title.set_text("n = 20")

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)


# From the 4 plots we can see that the time takes to compute the solution for diagonal matrix using both methods is quite unrelated to matrix size parameter m, n. Running the above code multiple times gives different results and these results presents a random pattern. However, it is noticeable that solution 2 (blue bar) gets averagely higher than solution 1 (red bar) across different values of n. From that we conclude method 1, which is solving the sub-systems one at a time is more efficient than solving the system directly. 

# ---
# ## Question 3: Numerical Integration [15 marks]
# 
# The integral of the function $f(x,y)= \sin(x) \cos\left(\frac{y}{5}\right)$ defined on the rectangle $\mathcal{D}\in\mathbb{R}^2 = (a,b)\times(c,d)$
# can be expressed in closed form as
# 
# $$
# I = \int_c^{d}\int_a^{b}  \sin(x)\cos\left(\frac{y}{5}\right) \ dx \ dy = 5\left(-\cos(b) + \cos(a)\right)\left(\sin\left(\frac{d}{5}\right) - \sin\left(\frac{c}{5}\right)\right).
# $$
# 
# for any $a<b$ and $c<d$.
# 
# **3.1** Create a surface plot of the function $f(x,y)$ on the interval $(-5, 5) \times (-5, 5)$.
# 
# **[3 marks]**

# In[6]:


def F(x, y):
    return np.sin(x) * np.cos(y/5)

x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000)

X, Y = np.meshgrid(x, y)
Z = F(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


# **3.2** Write a function `midpoint_I(D, N, M)` which takes 4 input arguments:
# 
# - a list `D` of length 4, to store the 4 interval bounds $a, b, c, d$,
# - a positive integer `N`,
# - a positive integer `M`,
# 
# and implements the 2D composite midpoint rule to compute and return an approximation of $I$, partitioning the rectangle $\mathcal{D}$ into $N\times M$ rectangles. (This translates to `N` nodes on the x-axis and `M` nodes on the y-axis.
# 
# You will need to adapt the 1D composite midpoint rule seen in Weeks 5 and 6 to extend it to 2 dimensions. Instead of approximating the integral by computing the sum of the surface areas of $N$ rectangles, you will need to sum the volumes of $N \times M$ cuboids.
# 
# **[3 marks]**

# In[7]:


def midpoint_I(D, N, M):
    delta_x = (D[1] - D[0]) / N
    delta_y = (D[3] - D[2]) / M
    
    I = 0
    for j in range(0,M):
        for i in range(0,N):
            I = I + (delta_x * delta_y) * F((D[0]+delta_x/2) + i*delta_x, (D[2]+delta_x/2) + j*delta_y)
    return I

D = [-1, 6, -3, 5]
print(midpoint_I(D, 100, 100))


# **3.3** Consider now the domain $\mathcal{D} = (0, 5)\times(0, 5)$. Compute the absolute error between the exact integral $I$ and the approximated integral computed with your `midpoint_I()` function from **3.2**, with all combinations of $M = 5, 10, 15, \dots, 300$ and $N = 5, 10, 15, \dots, 300$.
# 
# Store the error values in a $60\times 60$ NumPy array.
# 
# **[3 marks]**

# In[8]:


M1 = np.arange(5,305,5)
N1 = np.arange(5,305,5)
exact_I = 5*np.sin(1) * (-np.cos(5)+1)
D1 = [0, 5, 0, 5]
error_mat = np.eye(60)

for j in range(0,len(M1)):
    for i in range(0,len(N1)):
        error_mat[i, j] = abs(exact_I - midpoint_I(D1, N1[i], M1[j]))
        
print(error_mat)


# **3.4** Display the absolute error values as a function of $N$ and $M$ using a contour plot, with contour levels at $10^{-k}, k = 1, 2, \dots, 5$.
# 
# You may find the documentation for [`contour()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html#matplotlib.pyplot.contour), [`contourf()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html#matplotlib.pyplot.contourf), and [colormap normalization](https://matplotlib.org/stable/tutorials/colors/colormapnorms.html#logarithmic) useful to clearly display your results.
# 
# **[3 marks]**

# In[15]:


from matplotlib import cm
import matplotlib.colors as colors

X, Y = np.meshgrid(N1, M1)
Z = error_mat

fig,ax= plt.subplots()

levels = [0.00001, 0.0001, 0.001, 0.01, 0.1]


ax.contour(X, Y ,Z, levels)
ax.contourf(X, Y ,Z, levels)
ax.set_xlabel('N')
ax.set_ylabel('M')
pcm = ax.pcolor(X, Y, Z,
                   norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
                   cmap='PuBu_r', shading='auto')
fig.colorbar(pcm, ax=ax, extend='max')

plt.show()


# **3.5** Summarise and explain your observations from **3.4** in no more than 250 words. In particular, discuss how $\mathcal{D}$ should be partitioned to approximate $I$ with an error of at most $10^{-4}$, and comment on the symmetry or asymmetry of the partitioning.
# 
# **[3 marks]**

# From the above plot, we can see with reference to the colorbar on the right that in order to approximate I with an error of at most 10^(-4), we require the area in the plot with color deeper than that of 10^(-4). In addition, the purple lines are added to separete color area. We can also see that the color has a symmetrical pattern (around the line M = N). The closer M, N are to the symmetric line, the deeper the color is, which indicates more accuracy on integral estimation. We can get an accurate estimation if we choose N and M close
# enough.
# 

# In[ ]:




