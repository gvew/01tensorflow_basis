 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
import numpy as np

# 例子1，拟合y_data的函数，权重和偏置分别趋近0.1和0.3

# np.random.rand(100)生成100个[0,1]之间的随机数，构成1维数组
# np.random.rand(2,3)生成2行3列的二维数组
# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
# 权重偏置这些不断更新的值用tf变量存储，
# tf.random_uniform()的参数意义：(shape,min,max)
# tf.random_uniform([1], -1.0, 1.0) 产生-1.0到1.0的一个随机数，概率均匀
# 偏置初始化为0
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

# 损失函数。tf.reduce_mean()是取均值。square是平方
# 也就是去方差的均值。
loss = tf.reduce_mean(tf.square(y-y_data))
# 用梯度优化方法最小化损失函数。
# gradient 逐渐 descent 降低 optimizer 优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)
# minimize 最小化
train = optimizer.minimize(loss)
#也可以这样写 tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 变量Variable需要初始化并激活
init = tf.initialize_all_variables()
### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)          # Very important

# 循环训练201次，每20次做一次打印
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
        
        # 注意 打印的只能是 sess.run(Weights)，而不能直接打Weights；


