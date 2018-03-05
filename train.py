import os
import numpy as np
import tensorflow as tf
from alexnet import AlexNet
from input_data import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
import cv2

learning_rate_end = 0.0001
learning_rate = 0.0001
num_epochs = 500  # 代的个数
batch_size = 1000
dropout_rate = 0.5
num_classes = 8  # 类别标签
train_layers = ['fc8', 'fc7', 'fc6']
display_step = 50
current_dir = os.getcwd()
filewriter_path = current_dir + "/tensorboard/tensorboard1/"  # 存储tensorboard文件
checkpoint_path = current_dir + "/checkpoint/check1/"  # 训练好的模型和参数存放目录


if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

train_image_path = current_dir + '/train/'  # 指定训练集数据路径（根据实际情况指定训练数据集的路径）
test_image_path = current_dir + '/test/'  # 指定测试集数据路径（根据实际情况指定测试数据集的路径）

imgs_path = []
test_path = []
label_path = []
test_label = []

# 打开训练数据集目录，读取全部图片，生成图片路径列表
for file in os.listdir(train_image_path):
    name = file.split(sep = '_')
    if name[0] == '.DS' or name[0] == '.jpg' or name[0] == '':
        continue
    print(int(name[0]))
    label = int(name[0])
    face_path = np.array(train_image_path + file).tolist()
    img = cv2.imread(face_path)
    if img is None:
        continue
    imgs_path.append(img)
    label_path.append(label)

# 打开测试数据集目录，读取全部图片，生成图片路径列表
for file in os.listdir(test_image_path):
    name = file.split(sep = '_')
    if name[0] == '.DS' or name[0] == '.jpg' or name[0] == '':
        continue
    print(int(name[0]))
    label = int(name[0])
    face_path = np.array(test_image_path + file).tolist()
    img = cv2.imread(face_path)
    if img is None:
        continue
    test_path.append(face_path)
    test_label.append(label)

tr_data = ImageDataGenerator(
    images=imgs_path,
    labels=label_path,
    batch_size=batch_size,
    num_classes=num_classes)


test_data = ImageDataGenerator(
    images=test_path,
    labels=test_label,
    batch_size=batch_size,
    num_classes=num_classes,
    shuffle=False)

with tf.name_scope('input'):
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                   tr_data.data.output_shapes)

    training_initalize=iterator.make_initializer(tr_data.data)
    testing_initalize=iterator.make_initializer(test_data.data)

    # 定义每次迭代的数据
    next_batch = iterator.get_next()

x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
testacc = tf.placeholder(tf.float32,name="testacc")
testloss = tf.placeholder(tf.float32,name='testloss')
keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)
# 图片数据通过AlexNet网络处理
model = AlexNet(x, keep_prob, num_classes, train_layers)

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# 执行整个网络图
score = model.fc8

with tf.name_scope('loss'):
    # 损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                              labels=y))

gradients = tf.gradients(loss, var_list)

gradients = list(zip(gradients, var_list))

with tf.name_scope('optimizer'):
    # 优化器，采用梯度下降算法进行优化
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)


# 定义网络精确度
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 把精确度加入到Tensorboard
train_loss = tf.summary.scalar("train_loss", loss)
train_acc = tf.summary.scalar("train_acc",accuracy)
test_loss_end = tf.summary.scalar("test_loss", testloss)
test_acc_end = tf.summary.scalar("test_acc", testacc)
lr_decay = tf.summary.scalar("lr_decay",lr)
train_merged = tf.summary.merge([train_acc, train_loss])
writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver(max_to_keep=50)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_path)

# 定义一代的迭代次数
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))


def saveValid(loss, accuracy,test_step):
    print(loss)
    # get feed_dict
    feedDict = dict()
    feedDict[testloss] = loss
    feedDict[testacc] = accuracy
    feedDict[lr] = learning_rate
    # get runOps
    runOps = (test_loss_end, test_acc_end,lr_decay)
    session = tf.Session()
    # get summary
    summaries = session.run(runOps, feedDict)
    # write
    for summary in summaries:
        writer.add_summary(summary,global_step=test_step)
    return


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 把训练好的权重加入未训练的网络中
    #model.load_initial_weights(sess)


    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))
    print(" [*] Reading checkpoints...")
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" [*] Success to read {}".format(ckpt_name))
    else:
        print(" [*] Failed to find a checkpoint")
    # 总共训练10代
    for epoch in range(num_epochs):
        if learning_rate >= 0.0001:
            learning_rate = learning_rate_end * pow(0.9,epoch)
        elif learning_rate > 0.00001 and learning_rate < 0.0001:
            learning_rate = learning_rate_end * pow(0.9,epoch)
        else:
            learning_rate = 0.00001
        sess.run(training_initalize)
        print("{} Epoch number: {} start".format(datetime.now(), epoch + 1))
        sess.run(tr_data.labels)
        #开始训练每一代
        for step in range(train_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)
            sess.run(train_op, feed_dict={x: img_batch,
                                           y: label_batch,
                                           keep_prob: dropout_rate})
            if step % display_step == 0:
                s = sess.run(train_merged, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch * train_batches_per_epoch + step)
                loss_train, acc = sess.run([loss, accuracy],feed_dict={x: img_batch, y: label_batch, keep_prob: 1.})

                #print(str(loss_train) + " " + str(acc))
                print("Epoch: " + str(epoch + 1) + ", Batch: " + str(step) +
                  ", Loss= " + "{:.4f}".format(loss_train) + ", Training Accuracy= " + "{:.4f}".format(acc))
        # 测试模型精确度
        print("{} Start validation".format(datetime.now()))
        sess.run(testing_initalize)
        test_loss = 0.
        test_acc = 0.
        test_count = 0

        for _ in range(test_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)
            loss_test,acc = sess.run([loss,accuracy], feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.0})
            test_loss += loss_test
            test_acc += acc
            test_count += 1

        test_acc /= test_count
        test_loss /= test_count
        saveValid(test_loss, test_acc, epoch * test_batches_per_epoch)
        print("Epoch: " + str(epoch + 1) + ", Test Loss= " + "{:.4f}".format(test_loss) + ", Test Accuracy= " + "{:.4f}".format(test_acc))
        # 把训练好的模型存储起来
        print("{} Saving checkpoint of model...".format(datetime.now()))

        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Epoch number: {} end".format(datetime.now(), epoch + 1))
