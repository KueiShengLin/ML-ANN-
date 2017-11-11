import csv
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


f = open('bank-full.csv', 'r')
FEATURE = []    # list of the member and it's feature
ATTRIBUTE = []  # list of the feature
ATTRIBUTE_ID = []   # cluster id of the each feature
INPUT_DATA = []     # convert the feature which is str to the cluster id
CV = 5


def catch_feature():
    for row_id, row in enumerate(csv.reader(f, delimiter=';')):
        FEATURE.append(row)
        if row_id == 0:
            for i in range(len(FEATURE[0])):
                ATTRIBUTE.append([])
                ATTRIBUTE_ID.append({})
            continue
        for ele_id, element in enumerate(row):
            ATTRIBUTE[ele_id].append(element)
    f.close()

    for i in range(len(FEATURE[0])):
        try:
            for ele_id, element in enumerate(ATTRIBUTE[i]):
                ATTRIBUTE[i][ele_id] = int(element)
            ATTRIBUTE_ID[i]['int'] = 0
        except Exception:
            a_id = 0
            for v, k in enumerate(ATTRIBUTE[i]):
                if k in ATTRIBUTE_ID[i]:
                    continue
                else:
                    ATTRIBUTE_ID[i][k] = a_id
                    a_id += 1

    for f_id in range(len(FEATURE)):
        del FEATURE[f_id][-1]

    for feature_id, feature_ele in enumerate(FEATURE[1:]):
        input_t = []
        for ele_id, ele in enumerate(feature_ele):
            try:
                FEATURE[feature_id + 1][ele_id] = int(ele)
                input_t.append(int(ele))
            except Exception:
                FEATURE[feature_id + 1][ele_id] = ATTRIBUTE_ID[ele_id][ele]
                for i in range(len(ATTRIBUTE_ID[ele_id])):
                    if i == ATTRIBUTE_ID[ele_id][ele]:
                        input_t.append(1)
                    else:
                        input_t.append(0)
        INPUT_DATA.append(input_t)


def cross_validation():
    global CV, INPUT_DATA
    split_num = int(len(INPUT_DATA)/CV)
    split_list = []
    for num in range(0, len(INPUT_DATA), split_num):
        split_list.append(INPUT_DATA[(num-split_num):num])
    del split_list[0]
    return split_list


def add_layer(inputs, input_tensors, output_tensors, activation_function=None):
    w = tf.Variable(tf.random_normal([input_tensors, output_tensors]))
    b = tf.Variable(tf.truncated_normal([output_tensors]))
    formula = tf.add(tf.matmul(inputs, w), b)  # matmul = dot
    if activation_function is None:
        outputs = formula
    else:
        outputs = activation_function(formula)
    return outputs

# def accuary():

catch_feature()
print('catch feature down')
# INPUT = sum(len(a_id) for a_id in ATTRIBUTE_ID)
INPUT = len(INPUT_DATA[0])
HIDDEN = 100
OUTPUT = 1
Y_HAT = [ATTRIBUTE_ID[-1][name] for name in ATTRIBUTE[-1]]

WI = tf.Variable(tf.truncated_normal([INPUT, HIDDEN]))
WO = tf.Variable(tf.truncated_normal([HIDDEN, OUTPUT]))
BH = tf.Variable(tf.truncated_normal([HIDDEN]))
BO = tf.Variable(tf.truncated_normal([OUTPUT]))

y_feed = tf.placeholder(tf.float32, [None, OUTPUT])  # no:0 yes:1
input_feed = tf.placeholder(tf.float32, [None, INPUT])

hidden_layer = add_layer(input_feed, input_tensors=INPUT, output_tensors=HIDDEN, activation_function=tf.nn.sigmoid)
output_layer = add_layer(hidden_layer, input_tensors=HIDDEN, output_tensors=OUTPUT, activation_function=tf.nn.sigmoid)

loss = (y_feed - output_layer) ** 2
optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

cv_list = cross_validation()
cv_list_amount = int(len(cv_list[0]) / CV)
print('training')

for iteration in range(10):
    for i in range(1, len(cv_list)):
        for fid, feature in enumerate(cv_list[i]):
            sess.run(train, feed_dict={input_feed: [feature], y_feed: [[Y_HAT[cv_list_amount * i + fid]]]})
            # print(sess.run(output_layer, feed_dict={input_feed: [feature], y_feed: [[Y_HAT[fid]]]}))
    print(sess.run(loss, feed_dict={input_feed: [cv_list[0][0]], y_feed: [[Y_HAT[cv_list_amount]]]}))
print('down')
#
