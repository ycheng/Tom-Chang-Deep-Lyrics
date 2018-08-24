#coding:utf-8
import tensorflow as tf
import sys,time
import numpy as np
import os, json
import pickle as cPickle
import Config
import Model
import pprint as pp

if not os.path.exists('./model'):
    os.makedirs('./model')

config_tf = tf.ConfigProto()
# config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

def read_json():
    # file = sys.argv[1]
    file = "process.json"
    data = []
    with open(file,'r',encoding="utf8") as f:
        while True:
            l = f.readline()
            if len(l) == 0: break
            j = json.loads(l)
            data.append(j[0] + ":" + j[1] + ".")
    return data

def get_target_set(data):
    max_len = 0
    chars_set = set()
    for d in data:
        chars_set.update(d)
        max_len = max(max_len, len(d))
    return chars_set, max_len

def padding_data(data, target_len):
    rdata = []
    for d in data:
        rdata.append(d + "." * (target_len - len(d)))
    return rdata

data = read_json()
chars_set, max_len = get_target_set(data)
data = padding_data(data, max_len)

# for d in data:
#     print(len(d), d)

chars = list(chars_set) #char vocabulary
# pp.pprint(chars)

_vocab_size = len(chars)
print('data has %d sentences, each lenth %d and %d unique.' % (len(data), max_len, _vocab_size))
char_to_idx = { ch:i for i,ch in enumerate(chars) }
idx_to_char = { i:ch for i,ch in enumerate(chars) }

config = Config.Config()
config.vocab_size = _vocab_size

cPickle.dump((char_to_idx, idx_to_char), open(config.model_path+'.voc','wb'), protocol=cPickle.HIGHEST_PROTOCOL)

context_of_idx = []
for d in data:
    context_of_idx.append(
        [char_to_idx[c] for c in d]
    )

# max_len = 25 # for testing smaller memory size

# pp.pprint(context_of_idx)

def data_iterator(raw_data, epoch_size, batch_size, num_steps):
    data = np.array(raw_data, dtype=np.int32)

    # print("DI:batch_size", batch_size)
    # print("DI:num_steps", num_steps)
    
    for i in range(epoch_size):
        b1 = i * batch_size
        b2 = b1 + batch_size
        x = data[b1:b2, 0:num_steps]
        y = data[b1:b2, 1:num_steps+1] # y 就是 x 的錯一位，即下一個詞
        # print(x.shape, y.shape)
        yield (x, y)

def run_epoch(session, m, data, num_steps, eval_op):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1)
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for step, (x, y) in enumerate(data_iterator(data, epoch_size, m.batch_size,
                                                    num_steps)):
        cost, state, _ = session.run([m.cost, m.final_state, eval_op], # x 和 y 的 shape 都是 (batch_size, num_steps)
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
        costs += cost
        iters += 1

        # print("epoch_size", epoch_size)
        #if step and step % (epoch_size // 2) == 0:
        if False:
            print("%.2f perplexity: %.3f cost-time: %.2f s" %
                (step * 1.0 / epoch_size, np.exp(costs / iters),
                 (time.time() - start_time)))
            start_time = time.time()

    return np.exp(costs / (iters * num_steps))

def main(_):
    train_data = context_of_idx

    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model.Model(is_training=True, config=config, num_steps=max_len - 1)

        tf.global_variables_initializer().run()

        model_saver = tf.train.Saver(tf.global_variables())

        for i in range(config.iteration):
            print("Training Epoch: %d ..." % (i+1))
            train_perplexity = run_epoch(session, m, train_data, max_len - 1, m.train_op)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            if (i+1) % config.save_freq == 0:
                print('model saving ...')
                model_saver.save(session, config.model_path+'-%d'%(i+1))
                print('Done!')

if __name__ == "__main__":
    tf.app.run()
