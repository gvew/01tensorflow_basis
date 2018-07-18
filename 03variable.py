 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
variable:变量，一定要 initialize_all_variables
"""
import tensorflow as tf

# 定义一个tensorflow的变量，初始值为0，名字为counter：
state = tf.Variable(0, name='counter')
# 打印变量的名字
#print(state.name)    
# 定义常量值为1；
one = tf.constant(1)

# 定义加法步骤 (注: 此步并没有直接计算)
new_value = tf.add(state, one)
# 将 State 更新成 new_value
update = tf.assign(state, new_value)
# 变量Variable需要初始化并激活，并且打印的话只能通过sess.run()：
init = tf.initialize_all_variables()  # must have if define variable

# 使用 Session 计算
with tf.Session() as sess:
    sess.run(init)
    #循环训练三次
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

