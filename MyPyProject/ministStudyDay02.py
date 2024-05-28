# CNN
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
minist = input_data.read_data_sets('./resource', one_hot=True)

num_input = 784
num_classes = 10
learning_rate = 0.01
num_steps = 1000
batch_size = 64
# 解决过拟合
dropout = 0.8

X = tf.placeholder(tf.float32, shape=[None, num_input])
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

keep_prob = tf.placeholder(tf.float32)

weights = {
    # 卷积 2
    'wc1': tf.Variable(tf.random_normal(shape=[3, 3, 1, 16])),  # 图层1，个数32
    'wc2': tf.Variable(tf.random_normal(shape=[3, 3, 16, 32])),
    # 全连接 2
    'wd1': tf.Variable(tf.random_normal(shape=[7*7*32, 512])),
    'wd2': tf.Variable(tf.random_normal(shape=[512, num_classes]))  # 1024
}

biases = {
    'bc1': tf.Variable(tf.random_normal(shape=[16])),
    'bc2': tf.Variable(tf.random_normal(shape=[32])),

    'bd1': tf.Variable(tf.random_normal(shape=[512])),
    'bd2': tf.Variable(tf.random_normal(shape=[num_classes]))
}

# 卷
def conv2d(x, w, b, strides = 1):  # strides = 1表示向下移动一行
    x = tf.nn.conv2d(input=x, filter=w, strides=[1, strides, strides, 1], padding='SAME')  # SAME自动补零，形状恢复
    x = tf.add(x, b)
    # relu函数优点-正映射、缺点-0
    return tf.nn.relu(x)

def max_pool2d(x, k = 2):
    return tf.nn.max_pool(value=x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')  # full same valid

def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])  # -1自动获取
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = max_pool2d(conv1, k=2)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = max_pool2d(conv2, k=2)
    tc = tf.reshape(conv2, shape=[-1, weights['wd1'].get_shape().as_list()[0]])

    tc = tf.add(tf.matmul(tc, weights['wd1']), biases['bd1'])
    tc = tf.nn.relu(tc)

    tc = tf.nn.dropout(tc, keep_prob=dropout)

    out = tf.add(tf.matmul(tc, weights['wd2']), biases['bd2'])
    return out

# 调用
logits = conv_net(X, weights, biases, dropout=keep_prob)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)
prediction = tf.nn.softmax(logits)

current_pred = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(current_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1, num_steps+1):
        batch_x, batch_y = minist.train.next_batch(batch_size=batch_size)
        sess.run(train_op, feed_dict={X: batch_x,
                                      Y: batch_y,
                                      keep_prob: dropout})
        if step % 50 == 0 or step == 1:
            loss, acc = sess.run([loss_op,accuracy], feed_dict={X: batch_x,
                                                                Y: batch_y,
                                                                keep_prob: dropout})
            print('Step'+str(step)+',Minibatch loss='+'{:.4f}'.format(loss)+',Training Accuracy='+'{:.3f}'.format(acc))
    print('Optimization Finished!')
    print('Testing Accuracy=', sess.run(accuracy, feed_dict={X: minist.test.images[:64],
                                                             Y: minist.test.labels[:64],
                                                             keep_prob: dropout}))
