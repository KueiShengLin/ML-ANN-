import os
import tensorflow as tf
import pandas as pd
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_data = pd.read_csv("top100.csv")

ANS = 0
ALL_WORD = []
CV = 5
a2c = {'EAP': [1, 0, 0], 'HPL': [0, 1, 0], 'MWS': [0, 0, 1]}
train_data['_author'] = [a2c[a] for a in train_data['_author'].values]
print('')


def cross_validation():
    global CV
    split_num = int(math.ceil(len(train_data)/CV))
    split_list = []
    for num in range(0, len(train_data), split_num):
        sp = []
        # sp = [train_data.values[num + i].tolist() for i in range(split_num) if num+i < len(train_data)]
        for i in range(split_num):

            try:
                sp.append(list(train_data.values[num + i]))
                if i % 100 == 0:
                    print(i)
            except:
                break
        split_list.append(sp)
    return split_list


# tensorflow add layer
def add_layer(inputs, input_tensors, output_tensors, activation_function=None):
    w = tf.Variable(tf.random_normal([input_tensors, output_tensors]))
    b = tf.Variable(tf.truncated_normal([output_tensors]))
    formula = tf.add(tf.matmul(inputs, w), b)  # matmul = dot
    if activation_function is None:
        outputs = formula
    else:
        outputs = activation_function(formula)
    return outputs


def get_ans(y_feed, output_layer):
    global ANS
    if y_feed.index(max(y_feed)) == output_layer.index(max(output_layer)):
        ANS += 1

# cv_list = cross_validation()
cv_list = []
# layer units define
INPUT = len(train_data.values[0])-1
HIDDEN = INPUT * 2

OUTPUT = 3
Y_HAT = train_data['_author'].values[:].tolist()


y_feed = tf.placeholder(tf.float32, [None, OUTPUT])  # no:0 yes:1
input_feed = tf.placeholder(tf.float32, [None, INPUT])

# layer define you can add more hidden in there
hidden_layer = add_layer(input_feed, input_tensors=INPUT, output_tensors=HIDDEN, activation_function=tf.nn.sigmoid)
hidden_layer2 = add_layer(hidden_layer, input_tensors=HIDDEN, output_tensors=HIDDEN, activation_function=tf.nn.sigmoid)
# hidden_layer3 = add_layer(hidden_layer2, input_tensors=HIDDEN, output_tensors=HIDDEN, activation_function=tf.nn.sigmoid)
output_layer = add_layer(hidden_layer2, input_tensors=HIDDEN, output_tensors=OUTPUT, activation_function=tf.nn.softmax)

loss = tf.losses.mean_squared_error(y_feed, output_layer)   # loss function use mean squared
optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01)  # use adadeleta(adagrand的加強版) change learning rate
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print('training')

split_len = int(math.ceil(len(train_data) / CV))
del train_data['_author']
print('')
for cv in range(0, 1):
    print('cv' + str(cv))

    # if cv != 0:
    #     cv_list[0], cv_list[cv] = cv_list[cv], cv_list[0]
    for iteration in range(5):
        for fid, feature in enumerate(train_data.values):
            if fid < split_len:
                continue
            try:
                sess.run(train, feed_dict={input_feed: [feature.tolist()], y_feed: [Y_HAT[fid]]})
            except IndexError:
                break
        print(iteration)
        print(sess.run(output_layer, feed_dict={input_feed: [train_data.values[split_len].tolist()], y_feed: [Y_HAT[split_len]]}))
        print(Y_HAT[split_len])
        print(sess.run(loss, feed_dict={input_feed: [train_data.values[split_len].tolist()], y_feed: [Y_HAT[split_len]]}))

    for fid, feature in enumerate(train_data.values):
        if fid > split_len:
            break
        try:
            out = sess.run(output_layer, feed_dict={input_feed: [feature[0:-1].tolist()], y_feed: [Y_HAT[fid]]})
            get_ans(Y_HAT[fid], out.tolist())
        except IndexError:
            break
    print('Acc = ' + str(ANS / split_len))
#


