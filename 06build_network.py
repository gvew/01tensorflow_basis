 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
import numpy as np

# ����񾭲�ĺ����������ĸ�����������ֵ���������״���������״�ͼ���������
# Wx_plus_b��δ�����ֵ���������ؼ���ֵ��
def add_layer(inputs, in_size, out_size, activation_function=None):
# random_normal������̬�ֲ���������ֵ
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # ����˷�
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Make up some real data
# ����ѵ������
# np.linspace()��-1��1֮��Ȳ�����300������
# noise����̬�ֲ���������ǰ������������̬�ֲ��Ĳ�����Ȼ����size
# np.newaxis�Ĺ����ǲ�����ά��
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
# noise = np.random.normal(0, 0.05,  x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
# ����ռλ�������������������������롣
# �ڶ�������Ϊshape��None��������������1��������
# ���������������������������ÿ����������������
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
# �����1����Ԫ����Ϊֻ��һ�������������ز�10���������1����
# ���ú����������ز������㣬����size����һ�����Ԫ������ȫ���ӣ������size�Ǳ��������
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# ����Ԥ��ֵprediction����ʵֵ�����Զ��߲��ƽ�������ȡƽ����Ϊ��ʧ������
# reduction_indices��ʾ������ݵ�ѹ��ά�ȣ�����һ�㲻�����������������0ά��һ����������
# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)   # ���ݶ��Ż�������С����ʧ����

# important step
# ��ʼ�����������ִ������
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

