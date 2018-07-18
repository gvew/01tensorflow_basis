 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
import numpy as np

# ����1�����y_data�ĺ�����Ȩ�غ�ƫ�÷ֱ�����0.1��0.3

# np.random.rand(100)����100��[0,1]֮��������������1ά����
# np.random.rand(2,3)����2��3�еĶ�ά����
# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
# Ȩ��ƫ����Щ���ϸ��µ�ֵ��tf�����洢��
# tf.random_uniform()�Ĳ������壺(shape,min,max)
# tf.random_uniform([1], -1.0, 1.0) ����-1.0��1.0��һ������������ʾ���
# ƫ�ó�ʼ��Ϊ0
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

# ��ʧ������tf.reduce_mean()��ȡ��ֵ��square��ƽ��
# Ҳ����ȥ����ľ�ֵ��
loss = tf.reduce_mean(tf.square(y-y_data))
# ���ݶ��Ż�������С����ʧ������
# gradient �� descent ���� optimizer �Ż���
optimizer = tf.train.GradientDescentOptimizer(0.5)
# minimize ��С��
train = optimizer.minimize(loss)
#Ҳ��������д tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# ����Variable��Ҫ��ʼ��������
init = tf.initialize_all_variables()
### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)          # Very important

# ѭ��ѵ��201�Σ�ÿ20����һ�δ�ӡ
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
        
        # ע�� ��ӡ��ֻ���� sess.run(Weights)��������ֱ�Ӵ�Weights��


