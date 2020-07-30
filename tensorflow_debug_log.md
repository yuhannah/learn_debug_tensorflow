# tensorflow debug 方法实践总结

## 环境

> - python 3.7.4
> - tensorflow 1.15.0
> - keras 2.3.1
> - opencv 4.3.0
> - mnist数据集

## 准备工作

原代码中使用下述方式获取mnist数据集，即从指定网址在线下载：

```python
from tensorflow.examples.tutorials.mnist import input_data
```

在tensorflow 1.15版本中已经没有example目录。

从tensorflow的github（<https://github.com/tensorflow/tensorflow>）上找到examples.tutorials.mnist.input_data文件，文件开头注释内容说明了该模块被弃用：

```text
Functions for downloading and reading MNIST data (deprecated).
This module and all its submodules are deprecated.
```

学习过程中将input_data文件单独拷贝出来，以供读取mnist压缩包，转换成指定格式。

从官网上下载mnist的四个数据包。（<http://yann.lecun.com/exdb/mnist/>）

- train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
- train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
- t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
- t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

调用input_data.read_data_sets()时，需要确保输入的mnist数据包的相对路径正确。返回找到数据包。

## 方法一：Session.run()

将需要查看的变量写在 Session.run() 的括号中，如果有输入数据，也需要提供相应的 feed_dict 。如果想打印运行结果，使用 print() 输出执行结果即可。

该方法既简单又快捷，不管从哪里都能获取任何求值。

```python
print(f"Loss of the model is: {sess.run(loss, feed_dict={x: mnist.test.images, y_: mnist.test.labels})}%")
```

如果代码更为复杂些，可以应用 session 的 partial_run 执行。但由于这是一种实验特性，这里不再进一步实现展示给大家看了。


## 方法二：eval()

另外一种特别用于评估张量的 .eval() 方法。

使用 eval() 函数取值，再用 print 打印。效果等同于方法一用 Session.run() 。

```python
print(loss.eval(session=sess, feed_dict={x: batch_xs, y_: batch_ys}))
```

## 方法三：tf.Print()

在运行时求值时，tf.Print 方法用起来非常方便，因为这时我们不想用 Session.run() 显式地取用代码。

这是一个 identity 操作，在求值时会打印出数据。它能让我们查看求值期间的值的变化。

它对配置的要求有限，所以能很容易地 clog 终端。

> 谷歌云 AI 团队成员 Yufeng Guo 写过一篇很不错的文章（<https://towardsdatascience.com/using-tf-print-in-tensorflow-aa26e1cff11e>），讲解了如何使用 tf.Print 语句。他指出：
>
> > 你实际上使用返回的节点是非常重要的，因为如果你没这么做，会很不稳定。

使用 tf.Print() ，将需要查看的变量写在括号里，其中，第一个参数 input 表示出入节点，第二个参数 [] 表示经过节点，第三个参数是额外的打印的内容。

该方法的关键在于将 print 语句放置到网络节点之间，确保希望输出的节点经过该语句后还有被使用的节点。

其打印内容在 stderr 中。

```python
lossprint = tf.Print(loss, [loss], message="loss")
cost = sess.run(lossprint, feed_dict={x: batch_xs, y_: batch_ys})
# iter:  0  cost:  1.5507867
# loss[1.55078673]
# loss[1.59501088]
# iter:  1  cost:  1.5950109
# iter:  2  cost:  1.6215112
# loss[1.62151122]
# loss[1.58505452]
# iter:  3  cost:  1.5850545
# loss[1.24310839]
# iter:  4  cost:  1.2431084
# loss[0.8465904]
# iter:  5  cost:  0.8465904
# iter:  6  cost:  0.8985159
# loss[0.89851588]
# loss[0.849024832]
# iter:  7  cost:  0.84902483
# loss[0.935509801]
# iter:  8  cost:  0.9355098
# iter:  9  cost:  0.7335552
```

经过测试，如果 tf.Print() 的返回值于输入值同名，每调用一次 tf.Print() ，就会多打印输出一次，原因不明。如下所示。

```python
loss = tf.Print(loss, [loss], message="loss")
cost = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
# iter:  0  cost:  1.7018133
# loss[1.70181334]
# loss[1.57541442]
# loss[1.57541442]
# iter:  1  cost:  1.5754144
# loss[1.44510198]
# loss[1.44510198]
# loss[1.44510198]
# iter:  2  cost:  1.445102
# iter:  3  cost:  1.338381
# loss[1.33838105]
# loss[1.33838105]
# loss[1.33838105]
# loss[1.33838105]
# iter:  4  cost:  1.0924
# loss[1.0924]
# loss[1.0924]
# loss[1.0924]
# loss[1.0924]
# loss[1.0924]
# iter:  5  cost:  0.9073091
# loss[0.907309115]
# loss[0.907309115]
# loss[0.907309115]
# loss[0.907309115]
# loss[0.907309115]
# loss[0.907309115]
# iter:  6  cost:  0.95042527
```

## 方法四：tensorboard 可视化

用法的关键就是数据的序列化。TensorFlow 提供总结性的操作，能让你导出模型的压缩后信息，它们就像锚点一样告诉可视化面板绘制什么样的图。

TensorFlow 官网上有篇很棒的教程（<https://www.tensorflow.org/tensorboard/migrate#in_tf_1x>），讲了怎么实现它和使用 TensorBoard。

分为四个步骤：

- a) 使用恰当的名称和名称作用域清理计算图

    首先我们需要使用 TensorFlow 提供的所有作用域方法将全部变量和运算组织起来。

    ```python
    with tf.name_scope("variables_scope"):
        x = tf.placeholder(tf.float32, shape=[None, 784], name="x_placeholder")
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_placeholder")
    ```

- b) 添加 tf.summaries

    ```python
    with tf.name_scope("weights_scope"):
        W = tf.Variable(tf.zeros([784, 10]), name="weights_variable")
    tf.summary.histogram("weight_histogram", W)
    ```

- c) 添加一个 tf.summary.FileWriter 创建日志文件

    Tips：一定要为每个日志创建一个子文件夹，以避免计算图挤在一起

    ```python
    merged_summary_op = tf.summary.merge_all()
    tbWriter = tf.summary.FileWriter(logdir)
    tbWriter.add_graph(sess.graph)
    ```

- d) 从你的终端启动 TensorBoard 服务器

    ```python
    tensorboard --logdir=./tfb_logs/ --port=8090 --host=127.0.0.1
    ```

    导航至 TensorBoard 服务器（这里是 http://127.0.0.1:8090 ）。

    PS：使用过程中，在 win10 环境下，在 cmd 或者 power shell 中输入上述语句，提示找不到指定模块 DLL 。最终在 Anaconda Prompt 中输入上述语句则成功。根据网络资源，原因可能是需要在 tensorflow 环境下才能调用 tensorboard 。

    PS：需要输入 tensorboard 的绝对路径和 log 文件的绝对路径。我的电脑中 tensorboard.exe 在 Anaconda3/Script/ 。

    PS：后面的 --port=8090 --host=127.0.0.1 不是必须的。

    PS：tensorboard 2.2.1

tensorboard 的功能需要令外补充。

## 方法五：tensorboard 调试工具

TensorFlow 内置的这个调试功能非常实用，可以看看对应的 GitHub 仓库（<https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins/debugger>）加深了解。

要想使用这种方法，需要向前面的代码添加 3 样东西：

- 导入 from tensorflow.python import debug as tf_debug
- 用 tf_debug.TensorBoardDebugWrapsperSession 添加你的 session
- 将 debugger_port 添加到你的 TensorBoard 服务器

现在我们就有了调试整个可视化后模型的选项，没有其它调试工具，而是一张很美观的图。可以选择特定的节点并检查它们，使用“step”和“continue”按钮控制代码的执行，可视化张量和它们的值。

## 方法六：tensorflow 调试工具

最后一种同样强大的方法就是 CLI TensorFlow 调试工具（<https://www.tensorflow.org/guide/function#debugging>）。

这个调试工具重点是使用 tfdbg 的命令行界面，和 tfdbg 的图形用户界面相对。

只需用 tf_debug.LocalCLIDebugWrapperSession(sess) 封装 session，然后执行文件开始调试。

基本上能让你运行和查看模型的执行步骤，并提供评估指标。

那么这里的重要特性就是命令 invoke_stepper，然后按 s 逐步查看每项操作。这是一项很基本的调试功能，不过是 CLI 中。