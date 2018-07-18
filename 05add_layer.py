 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
激励函数：及让某一部分的神经元先激活起来，让后把激活效益信息传递到下一层去；数值被提高；
"""
import tensorflow as tf

# 添加神经层的函数，它有四个参数：输入值、输入的形状、输出的形状和激励函数，
# Wx_plus_b是未激活的值，函数返回激活值。
def add_layer(inputs, in_size, out_size, activation_function=None):
# random_normal：从正态分布中输出随机值
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 矩阵乘法
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

	
	