import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import input_data
from tensorflow.keras.datasets.mnist import load_data

# Only log errors (to prevent unnecessary cluttering of the console)
tf.logging.set_verbosity(tf.logging.ERROR)

number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def onehotEncoding(instance1, class1):
    """
    独热编码
    """
    temp1 = [0.] * len(class1)
    temp1[class1.index(instance1)] = 1
    return temp1


# We use the TF helper function to pull down the data from the MNIST site
# mnist = learnDebugTF.input_data.read_data_sets("../mnist", one_hot=True)
(train_x_, train_y_), (test_x_, test_y_) = load_data("mnist")
# print(train_x_.shape)  # (60000, 28, 28)
# print(train_y_.shape)  # (60000,)
# print(test_x_.shape)  # (10000, 28, 28)
# print(test_y_.shape)  # (10000,)
train_x = train_x_.reshape((train_x_.shape[0], -1)) / 255.0
test_x = test_x_.reshape((test_x_.shape[0], -1)) /255.0

train_y = []
test_y = []
for i in range(len(train_y_)):
    tmp = onehotEncoding(train_y_[i], number)
    train_y.append(tmp)
for i in range(len(test_y_)):
    tmp = onehotEncoding(test_y_[i], number)
    test_y.append(tmp)

train_y = np.array(train_y)
test_y = np.array(test_y)
print(train_x.shape)  # (60000, 784)
print(train_y.shape)  # (60000, 10)
print(test_x.shape)  # (10000, 784)
print(test_y.shape)  # (10000, 10)
# print(mnist.train.labels[0]) # 7-[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
# print(train_y_[0], train_y[0])  # 5-[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

# x is the placeholder for the 28 x 28 image data (the input)
# y_ is a 10 element vector, containing the predicted probability of each digit (0-9) class
# Define the weights and balances (always keep the dimensions in mind)
x = tf.placeholder(tf.float32, shape=[None, 784], name="x_placeholder")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_placeholder")

W = tf.Variable(tf.zeros([784, 10]), name="weights_variable")
b = tf.Variable(tf.zeros([10]), name="bias_variable")

# Define the activation function = the real y. Do not use softmax here, as it will be applied in the next step
assert x.get_shape().as_list() == [None, 784]
assert y_.get_shape().as_list() == [None, 10]
assert W.get_shape().as_list() == [784, 10]
assert b.get_shape().as_list() == [10]
y = tf.add(tf.matmul(x, W), b)

# Loss is defined as cross entropy between the prediction and the real value
# Each training step in gradient descent we want to minimize the loss
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_, logits=y, name="lossFunction"
    ),
    name="loss",
)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss, name="gradDescent")

# Initialize all variables

# Perform the initialization which is only the initialization of all global variables
init = tf.global_variables_initializer()

# ------ Set Session or InteractiveSession
sess = tf.InteractiveSession()
sess.run(init)

costs = []  # 代价

# Perform 1000 training steps
# Feed the next batch and run the training
batch_size = 100
start_index = 5000
epochs_completed = 0
train_size = train_x.shape[0]
batch_xs = np.zeros((batch_size, train_x.shape[1]))
batch_ys = np.zeros((batch_size, train_y.shape[1]))

for i in range(1000):
    if start_index + batch_size > train_size:
        rest_num = train_size - start_index
        batch_xs[0:rest_num, :] = train_x[start_index:train_size, :]
        batch_ys[0:rest_num, :] = train_y[start_index:train_size, :]

        # shuffle the data
        perm = np.arange(train_size)
        np.random.shuffle(perm)
        train_x = train_x[perm]
        train_y = train_y[perm]

        start_index = 5000
        rest_rest_num = batch_size - rest_num + start_index
        batch_xs[rest_num:batch_size, :] = train_x[start_index:rest_rest_num, :]
        batch_ys[rest_num:batch_size, :] = train_y[start_index:rest_rest_num, :]
        epochs_completed += 1
        print(batch_xs.shape, (batch_size, train_x.shape[1]))
        print(batch_ys.shape, (batch_size, train_y.shape[1]))
    else:
        end_index = start_index + batch_size
        batch_xs[:, :] = train_x[start_index:end_index, :]
        batch_ys[:, :] = train_y[start_index:end_index, :]
        start_index = end_index
        print(batch_xs.shape, (batch_size, train_x.shape[1]))
        print(batch_ys.shape, (batch_size, train_y.shape[1]))

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    out = sess.run(x, feed_dict={x: batch_xs, y_: batch_ys})  # (100, 784)
    out1 = sess.run(y_, feed_dict={x: batch_xs, y_: batch_ys})  # (100,10)
    out2 = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})  # (100,10)
    cost = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})  # (100,10)
    print("run: ", cost)
    costs.append(cost)
    print("iter: ", i, " cost: ", cost)

plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(0.5))
plt.show()

# Evaluate the accuracy of the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

print("=====================================")
print(
    f"The bias parameter is: {sess.run(b, feed_dict={x: test_x, y_: test_y})}"
)
print(
    f"Accuracy of the model is: {sess.run(accuracy, feed_dict={x: test_x, y_: test_y}) * 100}%"
)
print(
    f"Loss of the model is: {sess.run(loss, feed_dict={x: test_x, y_: test_y})}"
)

sess.close()
