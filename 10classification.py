"""**
  ******************************************************************************
  * @file    classification.py
  * @author  phone
  * @version V1.0
  * @date    2018-07-24
  * @brief  classification test
  ******************************************************************************
  * @attention
  *
  * @{实验平台:
	* @{ 
	--
	-Please note, this code is only for python 3+. 
	-If you are using python 2+, please modify the code accordingly.
  ******************************************************************************
  *"""
  "" import ------------------------------------------------------------------""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

"""----------------------------------------------------------------------
* @brief:隐藏层
-----------------------------------------------------------------------"""
def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

 """----------------------------------------------------------------------
* @brief:compute accuracy
-----------------------------------------------------------------------"""
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    # argmax 输出矩阵最大值所在的下标，第二个参数0列比较，1行比较；
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    # 准确性
    # 预测值是一行十列的值，每个值介于01之间的小数
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

"""---------------------------------------------------------------------------------
  * @brief   start operation
  ---------------------------------------------------------------------------------"""
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)

# the error between prediction and real data
# #求交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
                                              # log--lne
                                              #reduce_sum :按照维度求和https://www.cnblogs.com/Ph-one/p/9253584.html
                                              #reduce_mean:按照1 维度求平均值；
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))

