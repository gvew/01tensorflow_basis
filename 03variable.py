 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
variable:������һ��Ҫ initialize_all_variables
"""
import tensorflow as tf

# ����һ��tensorflow�ı�������ʼֵΪ0������Ϊcounter��
state = tf.Variable(0, name='counter')
# ��ӡ����������
#print(state.name)    
# ���峣��ֵΪ1��
one = tf.constant(1)

# ����ӷ����� (ע: �˲���û��ֱ�Ӽ���)
new_value = tf.add(state, one)
# �� State ���³� new_value
update = tf.assign(state, new_value)
# ����Variable��Ҫ��ʼ����������Ҵ�ӡ�Ļ�ֻ��ͨ��sess.run()��
init = tf.initialize_all_variables()  # must have if define variable

# ʹ�� Session ����
with tf.Session() as sess:
    sess.run(init)
    #ѭ��ѵ������
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

