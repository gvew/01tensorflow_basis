 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
placeholder: 占位符（只能是float32的形式），可以不定义初始值，用到feed_dict来给占位符实际的量；
"""
import tensorflow as tf

# 如果要传入值，用tensorflow的占位符，暂时存储变量，
# 以这种形式feed数据：sess.run(***, feed_dict={input: **})
# 只是占位符，在feed后占用内存；
# 在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
# mul = multiply 是将input1和input2 做乘法运算，并输出为 output 
ouput = tf.mul(input1, input2)

with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))
# 输出[ 14.]