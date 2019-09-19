import json
import collections
import os
import pickle
import tensorflow as tf
from bert import modeling, tokenization, optimization
from model_config import joint_config as config
from tensorflow.contrib.layers.python.layers import initializers
from lstm_crf_layer import BLSTM_CRF
import numpy as np

def get_slot_name(text, slot_label):
    slots = {}
    for i, slot in enumerate(slot_label):
        if slot == 'O':
            continue
        else:
            _, slot_name = slot.split('-')
            if slot_name in slots:
                slots[slot_name] += text[i]
            else:
                slots[slot_name] = text[i]

    return slots


class InputExample(object):
    def __init__(self, guid, text, domain=None, intent=None, slots=None):
        self.guid = guid
        self.text = text
        self.slots = slots
        self.domain = domain
        self.intent = intent

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 domain_id,
                 intent_id,
                 slot_id,
                 is_real_example=True):
      self.input_ids = input_ids
      self.input_mask = input_mask
      self.segment_ids = segment_ids
      self.domain_id = domain_id
      self.intent_id = intent_id
      self.slot_id = slot_id
      self.is_real_example = is_real_example


class Joint_Processor(object):
    def get_train_examples(self, data_dir):
        return self._create_examples(\
            self._read_json(os.path.join(data_dir, config['train_file'])), "train")

    def get_test_examples(self, test_file):
        """读取指定的测试文件"""
        return self._create_examples(\
                self._read_json(os.path.join("", test_file)), "test")

    def get_dev_examples(self, data_dir):
        pass

    def get_labels(self, data_dir, set_type):
        '''
        根据训练数据获取domain, intent, slots labels
        '''
        if set_type == 'train':
            data_list = self._read_json(os.path.join(data_dir, config['train_file']))
            domain_labels = set([data['domain'] for data in data_list])
            intent_labels = set([str(data['intent']) for data in data_list])

            slots_labels = set()
            for data in data_list:
                for slot in data['slots']:
                    slots_labels.add("B-%s" % slot)
                    #slots_labels.add("I-%s" % slot)
            slots_labels = list(slots_labels)

            id2domain = {i : label for i, label in enumerate(domain_labels)}
            domain2id = {label : i for i, label in id2domain.items()}

            id2intent = {i : label for i, label in enumerate(intent_labels)}
            intent2id = {label : i for i, label in id2intent.items()}

            #
            domain_d = {}
            intent_d = {}
            for data in data_list:
                if data['domain'] not in domain_d:
                    domain_d[data['domain']] = 1
                else:
                    domain_d[data['domain']] += 1

            for data in data_list:
                if data['intent'] not in intent_d:
                    intent_d[str(data['intent'])] = 1
                else:
                    intent_d[str(data['intent'])] += 1

            domain_w = [1] * len(domain2id)
            intent_w = [1] * len(intent2id)
            for key in domain2id:
                domain_w[domain2id[key]] = len(data_list) / (len(domain2id) + domain_d[key])
            for key in intent2id:
                intent_w[intent2id[key]] = len(data_list) / (len(intent2id) + intent_d[key])


            id2slot = {i : label for i, label in enumerate(slots_labels, 4)}
            id2slot[0] = '[PAD]'
            id2slot[1] = '[CLS]'
            id2slot[2] = '[SEP]'
            id2slot[3] = 'O'
            slot2id = {label : i for i, label in id2slot.items()}

            #保存
            with open(config['label_file'], 'wb') as fw:
                pickle.dump([id2domain, domain2id, id2intent, intent2id, id2slot, slot2id], fw)
        else:
            #预测时读取labels
            with open(config['label_file'], 'rb') as fr:
                id2domain, domain2id, id2intent, intent2id, id2slot, slot2id = pickle.load(fr)

            domain_w = [1] * len(domain2id)
            intent_w = [1] * len(intent2id)

        '''
        print("bert load %d domain labels, %d intent labels, %d slot labels" % \
                (len(id2domain), len(id2intent), len(id2slot)))
        '''

        return id2domain, domain2id, id2intent, intent2id, id2slot, slot2id, domain_w, intent_w


    @classmethod
    def _read_json(cls, input_file):
        """read json data """
        with open(input_file, "r") as f:
            return json.load(f)
 
    @classmethod
    def _get_slot_label(cls, text, slot):
        tag = ['O'] * len(text)
        for k, v in slot.items():
            index = text.find(v)
            if index == -1:
                continue
            tag[index] = 'B-%s' % k
            if len(v) > 1:
                for i in range(len(v) - 1):
                    tag[index + i + 1] = 'I-%s' % k
                    tag[index + i + 1] = 'B-%s' % k

        return ' '.join(tag)

    def _create_examples(self, data_list, set_type):
        examples = []
        for (i, data) in enumerate(data_list):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(data['text'])
            if set_type == "test":
                domain = "label-test"
                intent = "lable-test"
                slots = "label-test"
            else:
                domain = tokenization.convert_to_unicode(data['domain'])
                intent = tokenization.convert_to_unicode(str(data['intent']))
                slots = self._get_slot_label(data['text'], data['slots'])
                slots = tokenization.convert_to_unicode(slots)
            examples.append(\
                InputExample(guid=guid, text=text, domain=domain, intent=intent, slots=slots))
        if set_type == "test":
            pass
        else:
            import numpy as np
            np.random.shuffle(examples)

        return examples

def convert_single_example(ex_index, example, domain2id, intent2id, slot2id, max_seq_length,\
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids = [0] * max_seq_length,
            input_mask = [0] * max_seq_length,
            segment_ids = [0] * max_seq_length,
            domain_id = 0,
            intent_id = 0,
            slot_id = [0] * max_seq_length,
            is_real_example = False)

    #tokens_text = tokenizer.tokenize(example.text)
    tokens_text = [t if t != ' ' else '$' for t in example.text]
    #对测试和训练label分别处理
    slot_label = []
    if example.domain == 'label-test':
        domain_id = 0
        intent_id = 0
        slot_label = ['O'] * len(tokens_text)
    else:
        domain_id = domain2id[example.domain]
        intent_id = intent2id[example.intent]
        slot_label = example.slots.split()
    
    if len(tokens_text) > max_seq_length - 2:
        tokens_text = tokens_text[0 : max_seq_length - 2]
        slot_label = slot_label[0 : max_seq_length - 2]

    assert len(slot_label) == len(tokens_text)

    tokens = []
    segment_ids = []
    slot_id = []
    tokens.append('[CLS]')
    segment_ids.append(0)
    slot_id.append(slot2id['[CLS]'])
    for i, token in enumerate(tokens_text):
        tokens.append(tokenizer.tokenize(token)[0])
        segment_ids.append(0)
        slot_id.append(slot2id[slot_label[i]])
    tokens.append('[SEP]')
    segment_ids.append(0)
    slot_id.append(slot2id['[SEP]'])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    #padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        slot_id.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(slot_id) == max_seq_length
    
    '''
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("domain_ids: %s" % domain_id)
        tf.logging.info("intent_ids: %s" % intent_id)
        tf.logging.info("slot label: %s (id = %s)" % (example.slots, " ".join([str(x) for x in slot_id])))
    '''

    feature = InputFeatures(
        input_ids = input_ids,
        input_mask = input_mask,
        segment_ids = segment_ids,
        domain_id = domain_id,
        intent_id = intent_id,
        slot_id = slot_id,
        is_real_example = True)

    return feature

def file_based_convert_examples_to_features(examples, domain2id, intent2id, slot2id,\
            max_seq_length, tokenizer, output_file, task_name="domain"):
    """Convert a set of `InputExample`s to a TFRecord file."""
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_id, example) in enumerate(examples):
        if ex_id % 500 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_id, len(examples)))
        feature = convert_single_example(ex_id, example, domain2id, intent2id, slot2id,\
                                         max_seq_length, tokenizer)
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features['input_ids'] = create_int_feature(feature.input_ids)
        features['input_mask'] = create_int_feature(feature.input_mask)
        features['segment_ids'] = create_int_feature(feature.segment_ids)
        features['domain_id'] = create_int_feature([feature.domain_id])
        features['intent_id'] = create_int_feature([feature.intent_id])
        features['slot_id'] = create_int_feature(feature.slot_id)
        features['is_real_example'] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    writer.close()

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder, task_name="domain"):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "domain_id": tf.FixedLenFeature([], tf.int64),
            "intent_id": tf.FixedLenFeature([], tf.int64),
            "slot_id": tf.FixedLenFeature([seq_length], tf.int64),
            "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
        example[name] = t

        return example

    def input_fn(params):
         """The actual input function."""
         #任务batch size都设置相同
         batch_size = config['train_batch_size'] if is_training else config['predict_batch_size']
    
         # For training, we want a lot of parallel reading and shuffling.
         # For eval, we want no shuffling and parallel reading doesn't matter.
         d = tf.data.TFRecordDataset(input_file)
         if is_training:
             d = d.repeat()
             d = d.shuffle(buffer_size=100)

         d = d.apply(
                 tf.contrib.data.map_and_batch(
                     lambda record: _decode_record(record, name_to_features),
                     batch_size=batch_size,
                     drop_remainder=drop_remainder))
         return d

    return input_fn

def convert_examples_to_features(examples, domain2id, intent2id, slot2id,\
                                 max_seq_length, tokenizer):
    """在测试阶段将读取的InputExample数据转成features格式"""
    features_list = []
    for ex_index, example in enumerate(examples):
        feature = convert_single_example(ex_index, example, domain2id, \
                      intent2id, slot2id, max_seq_length, tokenizer)

        features_list.append(feature)

    return features_list

def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """在测试过程中读取features的函数"""
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_domain_ids = []
    all_intent_ids = []
    all_slot_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_domain_ids.append(feature.domain_id)
        all_intent_ids.append(feature.intent_id)
        all_slot_ids.append(feature.slot_id)

    def input_fn(params):
        """the actual input function"""
        batch_size = config['predict_batch_size']
        num_examples = len(features)
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids": tf.constant(all_input_ids, \
                         shape=[num_examples, seq_length], dtype=tf.int32),
            "input_mask": tf.constant(all_input_mask,\
                         shape=[num_examples, seq_length], dtype=tf.int32),
            "segment_ids": tf.constant(all_segment_ids, \
                         shape=[num_examples, seq_length], dtype=tf.int32),
            "domain_id": tf.constant(all_domain_ids, \
                         shape=[num_examples], dtype=tf.int32),
            "intent_id": tf.constant(all_intent_ids, \
                         shape=[num_examples], dtype=tf.int32),
            "slot_id": tf.constant(all_slot_ids, \
                         shape=[num_examples, seq_length], dtype=tf.int32)
        })
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size = 100)
        d = d.batch(batch_size = batch_size, drop_remainder = drop_remainder)

        return d

    return input_fn


def domain_classification(model, domain_id, num_domain, is_training, domain_w):
    '''domain classification'''
    #[batch, hidden_size]
    print("domain_classification...")
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    domain_output_weights = tf.get_variable("domain_output_weights",\
            [num_domain, hidden_size],\
            initializer = tf.truncated_normal_initializer(stddev=0.02))
    domain_output_bias = tf.get_variable("domain_output_bias",\
            [num_domain], initializer=tf.zeros_initializer())
    
    domain_w = tf.get_variable("domain_w", initializer=domain_w, dtype=tf.float32, trainable=False)

    with tf.variable_scope("domain_loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob = config['dropout_rate'])
        domain_logits = tf.matmul(output_layer, domain_output_weights, transpose_b=True)
        domain_logits = tf.nn.bias_add(domain_logits, domain_output_bias)
        domain_probabilities = tf.nn.softmax(domain_logits, axis=-1)
        domain_log_probs = tf.nn.log_softmax(domain_logits, axis=-1)
        domain_predictions = tf.argmax(domain_logits, axis=-1)
        domain_one_hot_lables = tf.one_hot(domain_id, depth=num_domain, dtype=tf.float32)
        domain_per_example_loss = -tf.reduce_sum(domain_one_hot_lables * domain_log_probs * domain_w, axis=-1)
        
        domain_loss = tf.reduce_mean(domain_per_example_loss)

        return domain_loss, domain_probabilities, domain_predictions

def intent_classification(model, intent_id, num_intent, is_training, intent_w):
    '''intent classification'''
    #[batch, hidden_size]
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    intent_output_weights = tf.get_variable("intent_output_weights",\
            [num_intent, hidden_size],\
            initializer = tf.truncated_normal_initializer(stddev=0.02))
    intent_output_bias = tf.get_variable("intent_output_bias",\
            [num_intent], initializer=tf.zeros_initializer())

    intent_w = tf.get_variable("intent_w", initializer=intent_w, dtype=tf.float32, trainable=False)
    
    with tf.variable_scope("intent_loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob = config['dropout_rate'])
        intent_logits = tf.matmul(output_layer, intent_output_weights, transpose_b=True)
        intent_logits = tf.nn.bias_add(intent_logits, intent_output_bias)
        intent_probabilities = tf.nn.softmax(intent_logits, axis=-1)
        intent_log_probs = tf.nn.log_softmax(intent_logits, axis=-1)
        intent_predictions = tf.argmax(intent_logits, axis=-1)
        intent_one_hot_lables = tf.one_hot(intent_id, depth=num_intent, dtype=tf.float32)
        #类别不均匀，损失加权
        intent_per_example_loss = -tf.reduce_sum(intent_one_hot_lables * intent_log_probs * intent_w, axis=-1)
        
        intent_loss = tf.reduce_mean(intent_per_example_loss)

        return intent_loss, intent_probabilities, intent_predictions

def slot_filling(model, lengths, slot_id, num_slot, is_training):
    '''
        slot filling
    '''
    #获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
    embedding = model.get_sequence_output()

    max_seq_length = embedding.shape[1].value

    #添加CRF out
    blstm_crf_config = {
        "embedded_chars": embedding,
        "hidden_unit": config['lstm_size'],
        "cell_type": config['cell'],
        "num_layers": config['num_layers'],
        "dropout_rate": config['dropout_rate'],
        "initializers": initializers,
        "num_labels": num_slot,
        "seq_length": max_seq_length,
        "labels": slot_id,
        "lengths": lengths,
        "is_training": is_training
    }

    blstm_crf = BLSTM_CRF(blstm_crf_config)
    loss, logits, trans, pred_ids = blstm_crf.add_blstm_crf_layer(crf_only=False)

    return loss, logits, trans, pred_ids


def create_model(bert_config, is_training, input_ids, input_mask,\
                 segment_ids, domain_id, intent_id, slot_id, num_domain,\
                 num_intent, num_slot, use_one_hot_embeddings, domain_w, intent_w):
    '''create a sequence labeling and classification model'''
    model = modeling.BertModel(
        config = bert_config,
        is_training = is_training,
        input_ids = input_ids,
        input_mask = input_mask,
        token_type_ids = segment_ids,
        use_one_hot_embeddings = use_one_hot_embeddings)

    #算序列真实长度
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices = 1) #

    #领域分类
    domain_loss, domain_probabilities, domain_pred =\
            domain_classification(model, domain_id, num_domain, is_training, domain_w)

    #意图识别
    intent_loss, intent_probabilities, intent_pred =\
            intent_classification(model, intent_id, num_intent, is_training, intent_w)

    #槽位填充
    slot_loss, slot_logits, trans, slot_pred = slot_filling(model, lengths, slot_id, num_slot, is_training)


    '''
    if is_training:
        total_loss = domain_loss + intent_loss + slot_loss
        return total_loss, domain_pred, intent_pred, slot_pred
    else:
        return None, domain_pred, intent_pred, slot_pred
    '''
    return domain_loss, intent_loss, slot_loss, domain_pred, intent_pred, slot_pred
    
def model_fn_builder(bert_config, num_domain, num_intent, num_slot, init_checkpoint,\
                     learning_rate, num_train_steps, num_warmup_steps, use_tpu,\
                     use_one_hot_embeddings, do_serve, domain_w, intent_w):

    #为什么会有一个labels？
    def model_fn(features, labels, mode, params):
        tf.logging.info("***features***")
        #print(features)
        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
        domain_id = features['domain_id']
        intent_id = features['intent_id']
        slot_id = features['slot_id']
        is_real_example = None #含义
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(domain_id), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        domain_loss, intent_loss, slot_loss, domain_pred, intent_pred, slot_pred = \
                create_model(bert_config, is_training, input_ids, input_mask, segment_ids, \
                    domain_id, intent_id, slot_id, num_domain, num_intent, num_slot,\
                    use_one_hot_embeddings, np.array(domain_w, dtype=np.float32), np.array(intent_w, dtype=np.float32))

        '''
        total_loss, domain_pred, intent_pred, slot_pred = \
                create_model(bert_config, is_training, input_ids, input_mask, segment_ids, \
                    domain_id, intent_id, slot_id, num_domain, num_intent, num_slot,\
                    use_one_hot_embeddings)
        '''

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        #加载模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            global_step = tf.train.get_global_step()
            #print("global_step: ", global_step)
            '''
            if num_train_steps < 1000:
                total_loss = domain_loss + intent_loss + slot_loss
            else:
                total_loss = domain_loss + intent_loss + (domain_loss + intent_loss) / slot_loss * slot_loss
            '''
            total_loss = domain_loss + intent_loss + 2 * slot_loss

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            #EstimatorSpec的使用
            output_spec = tf.estimator.EstimatorSpec(
                    mode = mode,
                    loss = total_loss,
                    train_op = train_op,
                    scaffold = scaffold_fn)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode = mode,
                predictions = {"domain_pred" : domain_pred,
                               "intent_pred" : intent_pred,
                               "slot_pred" : slot_pred},
                scaffold = scaffold_fn)   

        return output_spec

    return model_fn
