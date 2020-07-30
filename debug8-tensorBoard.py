import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import *
import time

np.random.seed(1)


def create_placeholders(n_x, n_y):
    """
    为TensorFlow会话创建占位符
    参数：
        n_x - 一个实数，图片向量的大小（64*64*3 = 12288）
        n_y - 一个实数，分类数（从0到5，所以n_y = 6）

    返回：
        X - 一个数据输入的占位符，维度为[n_x, None]，dtype = "float"
        Y - 一个对应输入的标签的占位符，维度为[n_Y,None]，dtype = "float"

    提示：
        使用None，因为它让我们可以灵活处理占位符提供的样本数量。事实上，测试/训练期间的样本数量是不同的。

    """
    with tf.name_scope('input'):
        X = tf.compat.v1.placeholder(tf.float32, [n_x, None], name="X")
        Y = tf.compat.v1.placeholder(tf.float32, [n_y, None], name="Y")

        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(X, [-1, 64, 64, 3])
        tf.summary.image('input', image_shaped_input, 10)
    return X, Y


# print("=======测试占位符=======")
# X, Y = create_placeholders(12288, 6)
# print("X = " + str(X))
# print("Y = " + str(Y))

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(input_tensor=var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(input_tensor=var))
        tf.summary.scalar('min', tf.reduce_min(input_tensor=var))
        tf.summary.histogram('histogram', var)


def initialize_parameters():
    """
    初始化神经网络的参数，参数的维度如下：
        W1 : [25, 12288]
        b1 : [25, 1]
        W2 : [12, 25]
        b2 : [12, 1]
        W3 : [6, 12]
        b3 : [6, 1]

    返回：
        parameters - 包含了W和b的字典


    """

    tf.compat.v1.set_random_seed(1)  # 指定随机种子

    with tf.name_scope('layer1'):
        with tf.name_scope('weight'):
            W1 = tf.compat.v1.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            variable_summaries(W1)
        with tf.name_scope('biases'):
            b1 = tf.compat.v1.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
            variable_summaries(b1)
    with tf.name_scope('layer2'):
        with tf.name_scope('weight'):
            W2 = tf.compat.v1.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            variable_summaries(W2)
        with tf.name_scope('biases'):
            b2 = tf.compat.v1.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
            variable_summaries(b2)
    with tf.name_scope('layer3'):
        with tf.name_scope('weight'):
            W3 = tf.compat.v1.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            variable_summaries(W3)
        with tf.name_scope('biases'):
            b3 = tf.compat.v1.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())
            variable_summaries(b3)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


# print("=======测试初始化=======")
# tf.compat.v1.reset_default_graph()  # 用于清除默认图形堆栈并重置全局默认图形。
# with tf.compat.v1.Session() as sess:
#     parameters = initialize_parameters()
#     print("W1 = " + str(parameters["W1"]))
#     print("b1 = " + str(parameters["b1"]))
#     print("W2 = " + str(parameters["W2"]))
#     print("b2 = " + str(parameters["b2"]))


def forward_propagation(X, parameters):
    """
    实现一个模型的前向传播，模型结构为LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    参数：
        X - 输入数据的占位符，维度为（输入节点数量，样本数量）
        parameters - 包含了W和b的参数的字典

    返回：
        Z3 - 最后一个LINEAR节点的输出

    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    with tf.name_scope('layer1'):
        with tf.name_scope('WX_plux_b'):
            Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
            # Z1 = tf.matmul(W1,X) + b1             #也可以这样写
            tf.summary.histogram('pre-activations', Z1)
        A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
        tf.summary.histogram('activations', A1)
    with tf.name_scope('layer2'):
        with tf.name_scope('WX_plux_b'):
            Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
            tf.summary.histogram('pre-activations', Z2)
        A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
        tf.summary.histogram('activations', A2)
    with tf.name_scope('layer3'):
        with tf.name_scope('WX_plux_b'):
            Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3
            tf.summary.histogram('pre-activations', Z3)

    return Z3


# print("=======测试向前传播=======")
# tf.compat.v1.reset_default_graph() #用于清除默认图形堆栈并重置全局默认图形。
# with tf.compat.v1.Session() as sess:
#     X,Y = create_placeholders(12288,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#     print("Z3 = " + str(Z3))


def compute_cost(Z3, Y):
    """
    计算成本

    参数：
        Z3 - 前向传播的结果
        Y - 标签，一个占位符，和Z3的维度相同

    返回：
        cost - 成本值


    """
    logits = tf.transpose(Z3)  # 转置
    labels = tf.transpose(Y)  # 转置

    with tf.name_scope('cross_entropy'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        tf.summary.scalar('cross_entropy', cost)
    return cost


# print("=======测试代价函数=======")
# tf.compat.v1.reset_default_graph()
# with tf.compat.v1.Session() as sess:
#     X,Y = create_placeholders(12288,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#     cost = compute_cost(Z3,Y)
#     print("cost = " + str(cost))


def model(X_train, Y_train, X_test, Y_test,
          learning_rate=0.0001, num_epochs=1500, minibatch_size=32,
          print_cost=True, is_plot=True):
    """
    实现一个三层的TensorFlow神经网络：LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX

    参数：
        X_train - 训练集，维度为（输入大小（输入节点数量） = 12288, 样本数量 = 1080）
        Y_train - 训练集分类数量，维度为（输出大小(输出节点数量) = 6, 样本数量 = 1080）
        X_test - 测试集，维度为（输入大小（输入节点数量） = 12288, 样本数量 = 120）
        Y_test - 测试集分类数量，维度为（输出大小(输出节点数量) = 6, 样本数量 = 120）
        learning_rate - 学习速率
        num_epochs - 整个训练集的遍历次数
        mini_batch_size - 每个小批量数据集的大小
        print_cost - 是否打印成本，每100代打印一次
        is_plot - 是否绘制曲线图

    返回：
        parameters - 学习后的参数

    """
    ops.reset_default_graph()  # 能够重新运行模型而不覆盖tf变量
    tf.compat.v1.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape  # 获取输入节点数量和样本数
    n_y = Y_train.shape[0]  # 获取输出节点数量
    costs = []  # 成本集

    # 给X和Y创建placeholder
    X, Y = create_placeholders(n_x, n_y)

    # 初始化参数
    parameters = initialize_parameters()

    # 前向传播
    Z3 = forward_propagation(X, parameters)

    # 计算成本
    cost = compute_cost(Z3, Y)

    # 反向传播，使用Adam优化
    with tf.name_scope('train'):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # 初始化所有的变量
    init = tf.compat.v1.global_variables_initializer()

    # 开始会话并计算
    with tf.compat.v1.Session() as sess:

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./log/train", sess.graph)
        test_writer = tf.summary.FileWriter("./log/test", sess.graph)

        # 初始化
        sess.run(init)

        # 正常训练的循环
        for epoch in range(num_epochs):

            epoch_cost = 0  # 每代的成本
            num_minibatches = int(m / minibatch_size)  # minibatch的总数量
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # 选择一个minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # 数据已经准备好了，开始运行session
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                # 计算这个minibatch在这一代中所占的误差
                epoch_cost = epoch_cost + minibatch_cost / num_minibatches

            # 记录并打印成本
            ## 记录成本
            if epoch % 5 == 0:
                costs.append(epoch_cost)
                # 是否打印：
                if print_cost and epoch % 100 == 0:
                    print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))

            # 计算当前的预测结果
            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

            # 计算准确率
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
            print("测试集的准确率: ", accuracy.eval({X: X_test, Y: Y_test}))

        # 是否绘制图谱
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # 保存学习后的参数
        parameters = sess.run(parameters)
        print("参数已经保存到session。")

        # 计算当前的预测结果
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
        print("测试集的准确率: ", accuracy.eval({X: X_test, Y: Y_test}))

        train_writer.close()
        test_writer.close()

        return parameters


def train(learning_rate=0.0001, num_epochs=1500, minibatch_size=32,
          print_cost=True, is_plot=False):
    # 开始时间
    start_time = time.clock()

    tf.compat.v1.set_random_seed(1)
    seed = 3

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # print(X_train_orig.shape)  # (1080, 64, 64, 3)
    # print(Y_train_orig.shape)  # (1, 1080)

    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T  # 每一列就是一个样本
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    # print(X_train_flatten.shape)  # (12288, 1080)

    # 归一化数据
    X_train = X_train_flatten / 255
    X_test = X_test_flatten / 255

    # 转换为独热矩阵
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)
    # print(Y_train.shape)  # (6, 1080)

    m = X_train.shape[1]
    with tf.name_scope('input'):
        X = tf.compat.v1.placeholder(tf.float32, [12288, None], name="X")
        Y = tf.compat.v1.placeholder(tf.float32, [6, None], name="Y")

        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(tf.transpose(X), [-1, 64, 64, 3])
        tf.summary.image('input', image_shaped_input, 10)

    with tf.name_scope('layer1'):
        with tf.name_scope('weight'):
            W1 = tf.compat.v1.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            variable_summaries(W1)
        with tf.name_scope('biases'):
            b1 = tf.compat.v1.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
            variable_summaries(b1)
        with tf.name_scope('WX_plux_b'):
            Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
            # Z1 = tf.matmul(W1,X) + b1             #也可以这样写
            tf.summary.histogram('pre-activations', Z1)
        A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
        tf.summary.histogram('activations', A1)

    with tf.name_scope('layer2'):
        with tf.name_scope('weight'):
            W2 = tf.compat.v1.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            variable_summaries(W2)
        with tf.name_scope('biases'):
            b2 = tf.compat.v1.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
            variable_summaries(b2)
        with tf.name_scope('WX_plux_b'):
            Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
            tf.summary.histogram('pre-activations', Z2)
        A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
        tf.summary.histogram('activations', A2)

    with tf.name_scope('layer3'):
        with tf.name_scope('weight'):
            W3 = tf.compat.v1.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            variable_summaries(W3)
        with tf.name_scope('biases'):
            b3 = tf.compat.v1.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())
            variable_summaries(b3)
        with tf.name_scope('WX_plux_b'):
            Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3
            tf.summary.histogram('pre-activations', Z3)

    with tf.name_scope('cross_entropy'):
        logits = tf.transpose(Z3)  # 转置
        labels = tf.transpose(Y)  # 转置
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        tf.summary.scalar('cross_entropy', cost)

    # 反向传播，使用Adam优化
    with tf.name_scope('train'):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # 计算当前的预测结果
            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        with tf.name_scope('accuracy'):
            # 计算准确率
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('accuracy', accuracy)

    # 初始化所有的变量
    init = tf.compat.v1.global_variables_initializer()

    # 开始会话并计算
    with tf.compat.v1.Session() as sess:

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./log/train", sess.graph)
        test_writer = tf.summary.FileWriter("./log/test", sess.graph)

        # 初始化
        sess.run(init)

        costs = []

        # 正常训练的循环
        for epoch in range(num_epochs):

            epoch_cost = 0  # 每代的成本
            num_minibatches = int(m / minibatch_size)  # minibatch的总数量
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            summary, acc = sess.run([merged, accuracy], feed_dict={X: X_test, Y: Y_test})
            test_writer.add_summary(summary, epoch * len(minibatches))

            for i in range(len(minibatches)):

                epochi = i + epoch * len(minibatches)
                # 选择一个minibatch
                (minibatch_X, minibatch_Y) = minibatches[i]
                if epochi % 50 == 49:

                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    # 数据已经准备好了，开始运行session
                    summary, _, minibatch_cost = sess.run([merged, optimizer, cost],
                                                          feed_dict={X: minibatch_X, Y: minibatch_Y},
                                                          options=run_options, run_metadata=run_metadata)

                    # 计算这个minibatch在这一代中所占的误差
                    epoch_cost = epoch_cost + minibatch_cost / num_minibatches

                    train_writer.add_run_metadata(run_metadata, 'step%03d' % epochi)
                    train_writer.add_summary(summary, epochi)
                    print('Adding run metadata for', epochi)

                else:
                    # 数据已经准备好了，开始运行session
                    summary, _, minibatch_cost = sess.run([merged, optimizer, cost],
                                                          feed_dict={X: minibatch_X, Y: minibatch_Y})

                    # 计算这个minibatch在这一代中所占的误差
                    epoch_cost = epoch_cost + minibatch_cost / num_minibatches

                    train_writer.add_summary(summary, epochi)

            print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
            print("测试集的准确率: ", accuracy.eval({X: X_test, Y: Y_test}))


            # 记录并打印成本
            ## 记录成本
            if epoch % 5 == 0:
                costs.append(epoch_cost)
                # 是否打印：
                if print_cost and epoch % 100 == 0:
                    print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))

        # 是否绘制图谱
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        train_writer.close()
        test_writer.close()

    # 结束时间
    end_time = time.clock()
    # 计算时差
    print("CPU的执行时间 = " + str(end_time - start_time) + " 秒")


if __name__ == '__main__':
    train(learning_rate=0.0001, num_epochs=1500)

# train accuracy = 0.99
# test accuracy = 0.8