 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
placeholder: ռλ����ֻ����float32����ʽ�������Բ������ʼֵ���õ�feed_dict����ռλ��ʵ�ʵ�����
"""
import tensorflow as tf

# ���Ҫ����ֵ����tensorflow��ռλ������ʱ�洢������
# ��������ʽfeed���ݣ�sess.run(***, feed_dict={input: **})
# ֻ��ռλ������feed��ռ���ڴ棻
# �� Tensorflow ����Ҫ���� placeholder �� type ��һ��Ϊ float32 ��ʽ
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
# mul = multiply �ǽ�input1��input2 ���˷����㣬�����Ϊ output 
ouput = tf.mul(input1, input2)

with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))
# ���[ 14.]