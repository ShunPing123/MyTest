# 深度学习--全连接神经网络FC
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf  # 导包tensorflow
from tensorflow.examples.tutorials.mnist import input_data  # 导入 minist 的 input_data 文件，以便读入数据
minist = input_data.read_data_sets('./resource', one_hot=True)  # 读入 Minist 数据集存入当前目录下的datasets文件夹中
# from keras.datasets import mnist
# (train_x_image, train_y), (test_x_image, test_y) = tf.keras.datasets.mnist.load_data(path=r'/home/bian/.keras/datasets/mnist.npz')

# 准备工作
num_input = 28*28*1  # 784 表示一张图像输入格式为一个行向量1*784
num_classes = 10  # 最终输出格式与数据集标签一致：一个行向量1*10
# 神经网络每层包含的 神经元个数/超参数 512太多，一般256
num_hidden_1 = 256
num_hidden_2 = 256
learning_rate = 0.001  # 学习率 步长(每次调整的距离) alpha
num_steps = 7000  # 迭代次数
batch_size = 64  # 一个批次大小，一批批传入

X = tf.placeholder(tf.float32, shape=(None, num_input))  # 图片格式占位变量：数值以float存储，默认读入行数*784列
Y = tf.placeholder(tf.float32, shape=(None, num_classes))
# 权重
weights = {
    'h1': tf.Variable(tf.random_normal(shape=[num_input, num_hidden_1])),
    'h2': tf.Variable(tf.random_normal(shape=[num_hidden_1, num_hidden_2])),
    'h3': tf.Variable(tf.random_normal(shape=[num_hidden_2, num_classes]))
}
# 偏移量
biases = {
    'b1': tf.Variable(tf.random_normal(shape=[num_hidden_1])),
    'b2': tf.Variable(tf.random_normal(shape=[num_hidden_2])),
    'b3': tf.Variable(tf.random_normal(shape=[num_classes]))
}

def neural_net(x):
    layer1 = tf.nn.sigmoid( tf.add( tf.matmul( x, weights['h1'] ), biases['b1'] ) )  # 激活/激励函数 非线性变化
    layer2 = tf.nn.sigmoid( tf.add( tf.matmul( layer1, weights['h2'] ), biases['b2'] ) )
    out_layer = tf.nn.sigmoid( tf.add( tf.matmul( layer2, weights['h3'] ), biases['b3'] ) )
    return out_layer

# 全连接神经网络建模 优化op
logits = neural_net(x=X)
# 平均损失 误差
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))  # avg
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # AdamOptimizer自带冲量优化  GradientDescentOptimizer
train_op = optimizer.minimize(loss=loss_op)

current_pred = tf.equal((tf.argmax(logits, axis=1)), (tf.argmax(Y, axis=1)))  # 横1 竖0
accuracy = tf.reduce_mean(tf.cast(current_pred, tf.float32))  # 人眼～0.93

with tf.Session() as sess:  # main
    sess.run(tf.global_variables_initializer())
    for step in range(1, num_steps+1):
        batch_x, batch_y = minist.train.next_batch(batch_size=batch_size)
        sess.run(train_op, feed_dict={X: batch_x,
                                      Y: batch_y})
        if step % 100 == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print('Step' + str(step) + ',Minibatch Loss=' + '{:.4f}'.format(loss) + 'Training Accuracy=' + '{:.3f}'.format(acc))
    print('Optimization Finished!')
    print('Testing Accuracy=', sess.run(accuracy, feed_dict={X: minist.test.images[:64],  # 截断处理
                                                             Y: minist.test.labels[:64]}))
