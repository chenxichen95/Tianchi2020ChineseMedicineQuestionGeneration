import json, os, re, sys
import numpy as np
from bert4keras.backend import keras, K, search_layer
from bert4keras.layers import Loss, Layer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam, extend_with_gradient_accumulation, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf

sys.path.append('.')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
K.clear_session() # 清空当前 session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
K.set_session(sess)

class config:
    max_t_len = 384
    max_q_len = 32
    max_a_len = 96
    max_qa_len = max_q_len + max_a_len

    train_batch_size = 4
    gradient_accumulation_steps = 8
    EPOCHS = 5

    config_path = {
        'wobert': '../user_data/chinese_wobert_L-12_H-768_A-12/bert_config.json',
        'wonezha': '../user_data/chinese_wonezha_L-12_H-768_A-12/bert_config.json',
    }
    checkpoint_path = {
        'wobert': '../user_data/chinese_wobert_L-12_H-768_A-12/bert_model.ckpt',
        'wonezha': '../user_data/chinese_wonezha_L-12_H-768_A-12/bert_model.ckpt',
    }
    dict_path = '../user_data/chinese_wonezha_L-12_H-768_A-12/vocab.txt'
    SEED = {
        'wobert': 42,
        'wonezha': 43,
    }
    data_path = {
        'train_data_json': '../data/round1_train_0907.json',
        'test_data_json': '../data/juesai_1011.json',
    }
    lr = 3e-5               # 初始学习率
    label_weight = 0.1      # 标签平滑的平滑因子

    ADV_epsilon = {
        'wobert': 0.3,      # 针对 wobert 的 Embedding 层进行对抗扰动的扰动系数
        'wonezha': 0.1,     # 针对 wonezha 的 Embedding 层进行对抗扰动的扰动系数
    }

    teacher_rate = 0.5      # Teache model 在 Student loss 中所占的权重
    temperature = 10.0      # 温度系数

    Train_Flag = False      # True ， 执行训练代码
    Test_Flag = True        # True ,执行测试代码，必须 要有 训练好的 模型权重 文件


def softmax(x, axis=-1):
    # 计算每行的最大值
    row_max = x.max(axis=axis, keepdims=True)

    x = x - row_max

    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


def load_train_json():
    D = []

    data = json.load(fp=open(config.data_path.get('train_data_json'), 'r', encoding='utf-8'))
    uni_id = 0

    for item in data:
        text = item['text']
        id = item['id']
        annotations = item['annotations']
        for qa in annotations:
            question = qa['Q']
            answer = qa['A']

            D.append({
                'question': question,
                'answer': answer,
                'text': text,
                'id': int(id),
                'uni_id': uni_id,
            })
            uni_id += 1

    return D


class data_generator_T(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        """单条样本格式：[CLS]篇章[SEP]问题[SEP]答案[SEP]
        """
        batch_token_ids, batch_segment_ids = [], []
        for is_end, D in self.sample(random):
            question = D['question']
            answer = D['answer']
            text = D['text']
            aq_token_ids, aq_segment_ids = tokenizer.encode(
                answer, question, maxlen=config.max_qa_len + 1
            )
            t_token_ids, t_segment_ids = tokenizer.encode(
                text, maxlen=config.max_t_len
            )
            token_ids = t_token_ids + aq_token_ids[1:]
            segment_ids = t_segment_ids + aq_segment_ids[1:]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class data_generator_S(DataGenerator):
    """
        Student Model 的数据生成器
    """
    def __init__(self, data, tokenizer=None, padding_max_len=None, teachers=None, **kwargs):
        super(data_generator_S, self).__init__(data, **kwargs)
        self.tokenizer = tokenizer
        self.padding_max_len = padding_max_len
        self.teachers = teachers


    def __iter__(self, random=False):
        """单条样本格式：[CLS]篇章[SEP]问题[SEP]答案[SEP]
        """
        batch_token_ids, batch_segment_ids = [], []
        for is_end, D in self.sample(random):
            question = D['question']
            answer = D['answer']
            text = D['text']
            aq_token_ids, aq_segment_ids = self.tokenizer.encode(
                answer, question, maxlen=config.max_qa_len + 1
            )
            t_token_ids, t_segment_ids = self.tokenizer.encode(
                text, maxlen=config.max_t_len
            )
            token_ids = t_token_ids + aq_token_ids[1:]
            segment_ids = t_segment_ids + aq_segment_ids[1:]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids, length=self.padding_max_len)
                batch_segment_ids = sequence_padding(batch_segment_ids, length=self.padding_max_len)
                teachers_label = []
                for teacher in self.teachers:
                    teachers_label.append(teacher.predict([batch_token_ids, batch_segment_ids]))
                batch_teachers_label = np.stack(teachers_label, axis=1)
                yield [batch_token_ids, batch_segment_ids, batch_teachers_label], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy_ls_T(Loss):
    """
        用于 Teacher 的 loss 函数，加了 label smoothing
    """
    def __init__(self, label_weight=None, vocab_size=None, **kwargs):
        super(CrossEntropy_ls_T, self).__init__(**kwargs)
        self.label_weight = label_weight
        self.vocab_size = vocab_size

        self.smoothing_value = label_weight / (vocab_size - 2)  # 多减掉一个 [PAD]
        smoothing_value = K.constant([self.smoothing_value])
        one_hot = K.tile(smoothing_value, [vocab_size])
        self.one_hot = K.expand_dims(K.concatenate([K.zeros([1]), one_hot[1:]]), 0)
        self.confidence = 1.0 - label_weight

    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs

        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位

        batch_size = K.shape(y_true)[0]
        num_pos = K.shape(y_true)[1]
        y_pred = K.reshape(y_pred, (-1, self.vocab_size))
        y_pred = K.softmax(y_pred, axis=-1)  # 原来 BERT 输出层的 softmax 已经被删除
        y_true = K.reshape(y_true, (-1, ))

        model_prob = K.tile(self.one_hot, [K.shape(y_true)[0], 1])
        target_one_hot = K.one_hot(K.cast(y_true, dtype='int32'), self.vocab_size)

        mask1 = tf.where(tf.equal(target_one_hot, 1), K.zeros_like(target_one_hot), K.ones_like(target_one_hot))
        model_prob = model_prob * mask1 + target_one_hot * self.confidence

        loss = tf.keras.losses.kullback_leibler_divergence(model_prob, y_pred)
        loss = K.reshape(loss, (batch_size, num_pos))

        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


class CrossEntropy_ls_S(Loss):
    """
        Student Model 使用的 loss ，加了 label smoothing 和 teacher label
    """
    def __init__(self, label_weight=None, vocab_size=None, temperature=1.0, teacher_rate=0.0, **kwargs):
        super(CrossEntropy_ls_S, self).__init__(**kwargs)
        self.label_weight = label_weight
        self.vocab_size = vocab_size

        self.smoothing_value = label_weight / (vocab_size - 2)  # 多减掉一个 [PAD]
        smoothing_value = K.constant([self.smoothing_value])
        one_hot = K.tile(smoothing_value, [vocab_size])
        self.one_hot = K.expand_dims(K.concatenate([K.zeros([1]), one_hot[1:]]), 0)
        self.confidence = 1.0 - label_weight
        self.temperature = temperature
        self.teacher_rate = teacher_rate

    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, teachers_pred, y_pred = inputs

        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        teachers_pred = teachers_pred[:, :, :-1]


        # 计算 student 与真实标签的 loss
        batch_size = K.shape(y_true)[0]
        num_pos = K.shape(y_true)[1]
        y_pred1 = K.reshape(y_pred, (-1, self.vocab_size))
        y_pred1 = K.softmax(y_pred1, axis=-1)  # 原来 BERT 输出层的 softmax 已经被删除
        y_true1 = K.reshape(y_true, (-1, ))
        model_prob = K.tile(self.one_hot, [K.shape(y_true1)[0], 1])
        target_one_hot = K.one_hot(K.cast(y_true1, dtype='int32'), self.vocab_size)

        mask1 = tf.where(tf.equal(target_one_hot, 1), K.zeros_like(target_one_hot), K.ones_like(target_one_hot))
        model_prob = model_prob * mask1 + target_one_hot * self.confidence

        loss0 = tf.keras.losses.kullback_leibler_divergence(model_prob, y_pred1)
        loss0 = K.reshape(loss0, (batch_size, num_pos))
        loss0 = K.sum(loss0 * y_mask) / K.sum(y_mask)

        # 计算 student 和 teacher 之间的 loss

        y_pred2 = K.reshape(K.stack([y_pred for _ in range(config.teacher_num)], axis=1), (-1, self.vocab_size))
        y_pred2 = K.softmax(y_pred2 / self.temperature, axis=-1)
        teachers_pred2 = K.reshape(teachers_pred, (-1, self.vocab_size))
        teachers_pred2 = K.softmax(teachers_pred2 / self.temperature, axis=-1)

        y_mask2 = K.reshape(K.stack([y_mask for _ in range(config.teacher_num)], axis=1), (-1, num_pos))

        loss_kd = self.temperature ** 2 * tf.keras.losses.kullback_leibler_divergence(teachers_pred2, y_pred2)
        loss_kd = K.reshape(loss_kd, (-1, num_pos))
        loss_kd = K.sum(loss_kd * y_mask2) / K.sum(y_mask2)

        total_loss = (1.0 - self.teacher_rate) * loss0 + self.teacher_rate * loss_kd

        return total_loss


class Rouge(object):
    def __init__(self, beta=1.0):

        self.beta = beta
        self.inst_scores = []

    def lcs(self, string, sub):
        if len(string) < len(sub):
            sub, string = string, sub
        lengths = np.zeros((len(string) + 1, len(sub) + 1))
        for j in range(1, len(sub) + 1):
            for i in range(1, len(string) + 1):
                if string[i - 1] == sub[j - 1]:
                    lengths[i][j] = lengths[i - 1][j - 1] + 1
                else:
                    lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
        return lengths[len(string)][len(sub)]

    def add_inst(self, cand, ref):

        basic_lcs = self.lcs(cand, ref)
        p_denom = len(cand)
        r_denom = len(ref)
        prec = basic_lcs / (p_denom if p_denom > 0. else 0.)
        recall = basic_lcs / (r_denom if r_denom > 0. else 0.)

        if prec != 0 and recall != 0:
            score = ((1 + self.beta ** 2) * prec * recall) / \
                    float(recall + self.beta ** 2 * prec)
        else:
            score = 0.0
        self.inst_scores.append(score)

    def score(self):
        return 1. * sum(self.inst_scores) / len(self.inst_scores)

    def score_one(self, cand, ref):
        basic_lcs = self.lcs(cand, ref)
        p_denom = len(cand)
        r_denom = len(ref)
        prec = basic_lcs / (p_denom if p_denom > 0. else 0.)
        recall = basic_lcs / (r_denom if r_denom > 0. else 0.)

        if prec != 0 and recall != 0:
            score = ((1 + self.beta ** 2) * prec * recall) / \
                    float(recall + self.beta ** 2 * prec)
        else:
            score = 0.0

        return score


class ReadingComprehension_T(AutoRegressiveDecoder):
    """
        用于 Teacher 的解码器
    """
    def __init__(self, model, **kwargs,):
        super(ReadingComprehension_T, self).__init__(**kwargs)
        self.model = model

    def get_ngram_set(self, x, n):
        """生成ngram合集，返回结果格式是:
        {(n-1)-gram: set([n-gram的第n个字集合])}
        """
        result = {}
        for i in range(len(x) - n + 1):
            k = tuple(x[i:i + n])
            if k[:-1] not in result:
                result[k[:-1]] = set()
            result[k[:-1]].add(k[-1])
        return result

    @AutoRegressiveDecoder.wraps(default_rtype='probas', use_states=True)
    def predict(self, inputs, output_ids, states):
        inputs = [i for i in inputs if i[0, 0] > -1]  # 过滤掉无答案篇章
        topk = len(inputs[0])
        all_token_ids, all_segment_ids = [], []
        for token_ids in inputs:  # inputs里每个元素都代表一个篇章
            token_ids = np.concatenate([token_ids, output_ids], 1)
            segment_ids = np.zeros_like(token_ids)
            if states > 0:
                segment_ids[:, -output_ids.shape[1]:] = 1
            all_token_ids.extend(token_ids)
            all_segment_ids.extend(segment_ids)
        padded_all_token_ids = sequence_padding(all_token_ids)
        padded_all_segment_ids = sequence_padding(all_segment_ids)
        probas = self.model.predict([padded_all_token_ids, padded_all_segment_ids])
        probas = softmax(probas, axis=-1)
        probas = [
            probas[i, len(ids) - 1] for i, ids in enumerate(all_token_ids)
        ]
        probas = np.array(probas).reshape((len(inputs), topk, -1))
        if states == 0:
            # 这一步主要是排除没有答案的篇章
            # 如果一开始最大值就为end_id，那说明该篇章没有答案
            argmax = probas[:, 0].argmax(axis=1)
            available_idxs = np.where(argmax != self.end_id)[0]
            if len(available_idxs) == 0:
                scores = np.zeros_like(probas[0])
                scores[:, self.end_id] = 1
                return scores, states + 1
            else:
                for i in np.where(argmax == self.end_id)[0]:
                    inputs[i][:, 0] = -1  # 无答案篇章首位标记为-1
                probas = probas[available_idxs]
                inputs = [i for i in inputs if i[0, 0] > -1]  # 过滤掉无答案篇章
        return (probas**2).sum(0) / (probas.sum(0) + 1), states + 1  # 某种平均投票方式

    def q_generate(self, answer, text, topk=1):
        token_ids = []
        t_token_ids = tokenizer.encode(text, maxlen=config.max_t_len)[0]
        a_token_ids = tokenizer.encode(answer, maxlen=config.max_a_len + 1)[0]
        token_ids.append(t_token_ids + a_token_ids[1:])
        output_ids = self.beam_search(
            token_ids, topk, states=0
        )  # 基于beam search
        return tokenizer.decode(output_ids)


class Evaluator_T(keras.callbacks.Callback):
    def __init__(self, eval_data, model_name):
        self.best_rouge = -1
        self.eval_data = eval_data
        self.model_name = model_name

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) >= 3:
            reader = ReadingComprehension_T(
                start_id=None,
                end_id=tokenizer._token_end_id,
                maxlen=config.max_q_len,
                model=self.model,
            )
            rouge = Rouge(beta=1.0)
            with tqdm(total=len(self.eval_data)) as pbar:
                for item in self.eval_data:
                    text = item['text']
                    answer = item['answer']
                    question = item['question']
                    predict_q = reader.q_generate(answer, text, topk=1)
                    rouge.add_inst(predict_q, question)
                    pbar.set_postfix(test_rouge=rouge.score())
                    pbar.update(1)
            new_rouge = rouge.score()
            print(f'Epoch {epoch + 1}/{self.epochs}, rouge:{new_rouge}')
            # 保存最优
            if new_rouge > self.best_rouge:
                print(f'Epoch:{epoch + 1}, Validation score improved ({self.best_rouge} --> {new_rouge}). Saving model!')
                self.best_rouge = new_rouge
                self.model.save_weights(f'../user_data/{self.model_name}_best_model.weights')


class ReadingComprehension_S(AutoRegressiveDecoder):
    """beam search解码来生成答案
    passages为多篇章组成的list，从多篇文章中自动决策出最优的答案，
    如果没答案，则返回空字符串。
    mode是extractive时，按照抽取式执行，即答案必须是原篇章的一个片段。
    """
    def __init__(self, model, **kwargs,):
        super(ReadingComprehension_S, self).__init__(**kwargs)
        self.model = model

    def get_ngram_set(self, x, n):
        """生成ngram合集，返回结果格式是:
        {(n-1)-gram: set([n-gram的第n个字集合])}
        """
        result = {}
        for i in range(len(x) - n + 1):
            k = tuple(x[i:i + n])
            if k[:-1] not in result:
                result[k[:-1]] = set()
            result[k[:-1]].add(k[-1])
        return result

    @AutoRegressiveDecoder.wraps(default_rtype='probas', use_states=True)
    def predict(self, inputs, output_ids, states):
        inputs = [i for i in inputs if i[0, 0] > -1]  # 过滤掉无答案篇章
        topk = len(inputs[0])
        all_token_ids, all_segment_ids = [], []
        for token_ids in inputs:  # inputs里每个元素都代表一个篇章
            token_ids = np.concatenate([token_ids, output_ids], 1)
            segment_ids = np.zeros_like(token_ids)
            if states > 0:
                segment_ids[:, -output_ids.shape[1]:] = 1
            all_token_ids.extend(token_ids)
            all_segment_ids.extend(segment_ids)
        padded_all_token_ids = sequence_padding(all_token_ids)
        padded_all_segment_ids = sequence_padding(all_segment_ids)
        bs, num = padded_all_token_ids.shape
        padded_all_teacher_labals = np.zeros(shape=(bs, config.teacher_num, num, tokenizer._vocab_size))
        probas = self.model.predict([padded_all_token_ids, padded_all_segment_ids, padded_all_teacher_labals])
        probas = softmax(probas, -1)
        probas = [
            probas[i, len(ids) - 1] for i, ids in enumerate(all_token_ids)
        ]
        probas = np.array(probas).reshape((len(inputs), topk, -1))
        if states == 0:
            # 这一步主要是排除没有答案的篇章
            # 如果一开始最大值就为end_id，那说明该篇章没有答案
            argmax = probas[:, 0].argmax(axis=1)
            available_idxs = np.where(argmax != self.end_id)[0]
            if len(available_idxs) == 0:
                scores = np.zeros_like(probas[0])
                scores[:, self.end_id] = 1
                return scores, states + 1
            else:
                for i in np.where(argmax == self.end_id)[0]:
                    inputs[i][:, 0] = -1  # 无答案篇章首位标记为-1
                probas = probas[available_idxs]
                inputs = [i for i in inputs if i[0, 0] > -1]  # 过滤掉无答案篇章
        return (probas**2).sum(0) / (probas.sum(0) + 1), states + 1  # 某种平均投票方式

    def q_generate(self, answer, text, topk=1):
        token_ids = []
        t_token_ids = tokenizer.encode(text, maxlen=config.max_t_len)[0]
        a_token_ids = tokenizer.encode(answer, maxlen=config.max_a_len + 1)[0]
        token_ids.append(t_token_ids + a_token_ids[1:])
        output_ids = self.beam_search(
            token_ids, topk, states=0
        )  # 基于beam search
        return tokenizer.decode(output_ids)


class Evaluator_S(keras.callbacks.Callback):
    def __init__(self, eval_data, model_name):
        self.best_rouge = -1
        self.eval_data = eval_data
        self.model_name = model_name

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) >= 3:
            reader = ReadingComprehension_S(
                start_id=None,
                end_id=tokenizer._token_end_id,
                maxlen=config.max_q_len,
                model=self.model,
            )
            rouge = Rouge(beta=1.0)
            with tqdm(total=len(self.eval_data)) as pbar:
                for item in self.eval_data:
                    text = item['text']
                    answer = item['answer']
                    question = item['question']
                    predict_q = reader.q_generate(answer, text, topk=1)
                    rouge.add_inst(predict_q, question)
                    pbar.set_postfix(test_rouge=rouge.score())
                    pbar.update(1)
            new_rouge = rouge.score()
            print(f'Epoch {epoch + 1}/{self.epochs}, rouge:{new_rouge}')
            # 保存最优
            if new_rouge > self.best_rouge:
                print(f'Epoch:{epoch + 1}, Validation score improved ({self.best_rouge} --> {new_rouge}). Saving model!')
                self.best_rouge = new_rouge
                self.model.save_weights(f'../user_data/{self.model_name}_best_model.weights')


class Keras_Model:
    def __init__(self, config_path, weight_file, model_type, keep_tokens):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph)
            with self.session.as_default():
                if model_type == 'nezha' or model_type == 'wonezha':
                    self.model = build_transformer_model(
                        config_path,
                        model='nezha',
                        application='unilm',
                        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
                    )
                else:
                    self.model = build_transformer_model(
                        config_path,
                        application='unilm',
                        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
                    )
                self.model.load_weights(weight_file)

    def predict(self, *inputs):
        with self.graph.as_default():
            with self.session.as_default():
                return self.model.predict(*inputs)


def adversarial_training(model, embedding_name, epsilon=1.0):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
            model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads ** 2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


class ReadingComprehension_test(AutoRegressiveDecoder):
    """beam search解码来生成答案
    passages为多篇章组成的list，从多篇文章中自动决策出最优的答案，
    如果没答案，则返回空字符串。
    mode是extractive时，按照抽取式执行，即答案必须是原篇章的一个片段。
    """
    def __init__(self, models, **kwargs,):
        super(ReadingComprehension_test, self).__init__(**kwargs)
        self.models = models

    def get_ngram_set(self, x, n):
        """生成ngram合集，返回结果格式是:
        {(n-1)-gram: set([n-gram的第n个字集合])}
        """
        result = {}
        for i in range(len(x) - n + 1):
            k = tuple(x[i:i + n])
            if k[:-1] not in result:
                result[k[:-1]] = set()
            result[k[:-1]].add(k[-1])
        return result

    @AutoRegressiveDecoder.wraps(default_rtype='probas', use_states=True)
    def predict(self, inputs, output_ids, states):
        inputs = [i for i in inputs if i[0, 0] > -1]  # 过滤掉无答案篇章
        topk = len(inputs[0])
        all_token_ids, all_segment_ids = [], []
        for token_ids in inputs:  # inputs里每个元素都代表一个篇章
            token_ids = np.concatenate([token_ids, output_ids], 1)
            segment_ids = np.zeros_like(token_ids)
            if states > 0:
                segment_ids[:, -output_ids.shape[1]:] = 1
            all_token_ids.extend(token_ids)
            all_segment_ids.extend(segment_ids)
        padded_all_token_ids = sequence_padding(all_token_ids)
        padded_all_segment_ids = sequence_padding(all_segment_ids)
        probas1 = self.models[0].predict([padded_all_token_ids, padded_all_segment_ids])
        probas2 = self.models[1].predict([padded_all_token_ids, padded_all_segment_ids])

        probas1 = softmax(probas1, -1)
        probas2 = softmax(probas2, -1)
        probas = (probas1 + probas2) / 2
        probas = [
            probas[i, len(ids) - 1] for i, ids in enumerate(all_token_ids)
        ]
        probas = np.array(probas).reshape((len(inputs), topk, -1))
        if states == 0:
            # 这一步主要是排除没有答案的篇章
            # 如果一开始最大值就为end_id，那说明该篇章没有答案
            argmax = probas[:, 0].argmax(axis=1)
            available_idxs = np.where(argmax != self.end_id)[0]
            if len(available_idxs) == 0:
                scores = np.zeros_like(probas[0])
                scores[:, self.end_id] = 1
                return scores, states + 1
            else:
                for i in np.where(argmax == self.end_id)[0]:
                    inputs[i][:, 0] = -1  # 无答案篇章首位标记为-1
                probas = probas[available_idxs]
                inputs = [i for i in inputs if i[0, 0] > -1]  # 过滤掉无答案篇章
        return (probas**2).sum(0) / (probas.sum(0) + 1), states + 1  # 某种平均投票方式

    def q_generate(self, answer, text, topk=1):
        token_ids = []
        t_token_ids = tokenizer.encode(text, maxlen=config.max_t_len)[0]
        a_token_ids = tokenizer.encode(answer, maxlen=config.max_a_len + 1)[0]
        token_ids.append(t_token_ids + a_token_ids[1:])
        output_ids = self.beam_search(
            token_ids, topk, states=0
        )  # 基于beam search
        return tokenizer.decode(output_ids)


def test(models, topk):
    test_data = json.load(fp=open(config.data_path.get('test_data_json'), 'r', encoding='utf-8'))
    reader = ReadingComprehension_test(
        start_id=None,
        end_id=tokenizer._token_end_id,
        maxlen=config.max_q_len,
        models=models,
    )
    predict_result = []
    with tqdm(total=len(test_data)) as pbar:
        for item in test_data:
            text = item['text']
            id = item['id']
            annotations = item['annotations']
            new_annotations = []
            for qa in annotations:
                answer = qa['A']
                q_generated = reader.q_generate(answer, text, topk=topk)
                new_annotations.append({
                    "Q": q_generated,
                    "A": answer
                })
            predict_result.append({
                "id": id,
                'text': text,
                "annotations": new_annotations,
            })
            pbar.update(1)

    json.dump(predict_result, fp=open(f'../prediction_result/result.json', 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    model_types = ['wobert', 'wonezha']
    token_dict, keep_tokens = load_vocab(
        dict_path=config.dict_path,
        simplified=True,
        startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )
    tokenizer = Tokenizer(token_dict, do_lower_case=True)

    if config.Train_Flag:
        config.teacher_num = 1
        train_data = load_train_json()
        train_data, eval_data = train_test_split(train_data, test_size=0.1, random_state=42, shuffle=True)
        #train Teacher Model
        for i, model_type in enumerate(model_types):
            np.random.seed(config.SEED[f'{model_type}'])
            evaluator = Evaluator_T(eval_data, f'T_{model_type}')
            train_generator = data_generator_T(train_data, config.train_batch_size)

            model = build_transformer_model(
                config.config_path[f'{model_type}'],
                config.checkpoint_path[f'{model_type}'],
                model='nezha' if 'nezha' in model_type else 'bert',
                application='unilm',
                keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
            )
            output = CrossEntropy_ls_T(
                label_weight=config.label_weight,
                vocab_size=tokenizer._vocab_size,
                output_axis=2
            )(model.inputs + model.outputs)

            model = Model(model.inputs, output)

            optimizer = extend_with_piecewise_linear_lr(Adam)
            optimizer = extend_with_gradient_accumulation(optimizer)
            optimizer_params = {
                'learning_rate': config.lr,
                'lr_schedule': {0: 1.0, len(train_generator) * config.EPOCHS: 0.5},
                'grad_accum_steps': config.gradient_accumulation_steps,
            }
            optimizer = optimizer(**optimizer_params)
            model.compile(optimizer=optimizer)

            # 对 Embedding 层进行对抗扰动
            adversarial_training(model, 'Embedding-Token', config.ADV_epsilon[f'{model_type}'])

            print(f'training teacher model: {model_type}')
            model.fit_generator(
                train_generator.forfit(),
                steps_per_epoch=len(train_generator),
                epochs=config.EPOCHS,
                callbacks=[evaluator]
            )
            K.clear_session()
            tf.reset_default_graph()

        # train Student Model
        for i, model_type in enumerate(model_types):
            np.random.seed(config.SEED[f'{model_type}'])
            # load teacher model
            teacher_models = []
            teacher_models.append(
                Keras_Model(
                    config_path=config.config_path[f'{model_type}'],
                    weight_file=f'../user_data/T_{model_type}_best_model.weights',
                    model_type=model_type,
                    keep_tokens=keep_tokens,
                )
            )

            evaluator = Evaluator_S(eval_data, f'S_{model_type}')
            train_generator = data_generator_S(train_data, batch_size=config.train_batch_size, tokenizer=tokenizer, teachers=teacher_models)

            graph = tf.Graph()
            with graph.as_default():
                session = tf.Session(graph=graph)
                with session.as_default():
                    model = build_transformer_model(
                        config.config_path[f'{model_type}'],
                        config.checkpoint_path[f'{model_type}'],
                        model='nezha' if 'nezha' in model_type else 'bert',
                        application='unilm',
                        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
                    )
                    teachers_label = keras.layers.Input(shape=(config.teacher_num, None, None))
                    output = CrossEntropy_ls_S(
                        label_weight=config.label_weight,
                        vocab_size=tokenizer._vocab_size,
                        output_axis=3,
                        teacher_rate=config.teacher_rate,
                        temperature=config.temperature,
                    )(model.inputs + [teachers_label] + model.outputs)

                    model = Model(model.inputs + [teachers_label], output)

                    optimizer = extend_with_piecewise_linear_lr(Adam)
                    optimizer = extend_with_gradient_accumulation(optimizer)
                    optimizer_params = {
                        'learning_rate': config.lr,
                        'lr_schedule': {0: 1.0, len(train_generator) * config.EPOCHS: 0.5},
                        'grad_accum_steps': config.gradient_accumulation_steps,
                    }
                    optimizer = optimizer(**optimizer_params)
                    model.compile(optimizer=optimizer)

                    adversarial_training(model, 'Embedding-Token', config.ADV_epsilon[f'{model_type}'])
                    print(f'training student model: {model_type}')
                    model.fit_generator(
                        train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=config.EPOCHS,
                        callbacks=[evaluator]
                    )
            K.clear_session()
            tf.reset_default_graph()

    if config.Test_Flag:
        # 检测 student model 是否存在
        S_weight_files = [f'../user_data/S_{model_type}_best_model.weights' for model_type in model_types]
        for S_weight_file in S_weight_files:
            if not os.path.exists(S_weight_file):
                raise Exception(f'{S_weight_file} not exist ~')

        models = []
        for i, model_type in enumerate(model_types):
            models.append(
                Keras_Model(
                    config_path=config.config_path[f'{model_type}'],
                    weight_file=f'../user_data/S_{model_type}_best_model.weights',
                    model_type=model_type,
                    keep_tokens=keep_tokens,
                )
            )
        test(models, topk=5)
