import tensorflow as tf
import json
import os
import sys
import numpy as np
from model_config import joint_config as config
from utils import Joint_Processor
from utils import file_based_convert_examples_to_features, file_based_input_fn_builder
from utils import model_fn_builder, convert_examples_to_features
from utils import input_fn_builder, get_slot_name
from bert import modeling, tokenization, optimization
from patterns import code_pattern
import json
import re

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(test_file = 'test.json'):
    tf.logging.set_verbosity(tf.logging.INFO)
    #1.设置数据处理器
    processors = {
        'joint': Joint_Processor
    }

    task_name = config['task_name'].lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)
    processor = processors[task_name]()

    #1.1获取标签
    id2domain, domain2id, id2intent, intent2id, id2slot, slot2id = \
            processor.get_labels(config["data_dir"],\
                                 "train" if config['do_train'] else "test")

    #print(domain2id)
    #print(intent2id)
    #print(slot2id)
    #获取分词器
    tokenizer = tokenization.FullTokenizer(\
                    vocab_file=config['vocab_file'], do_lower_case=config['do_lower_case'])

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    save_checkpoints_steps = config['save_checkpoints_steps']

    #1.2读取训练数据，并转成example格式
    if config['do_train']:
        tf.logging.info("***** Loading training examples *****")
        train_examples = processor.get_train_examples(config['data_dir'])
        num_train_steps = int(len(train_examples) / config['train_batch_size'] * config['num_train_epochs'])
        num_warmup_steps = int(num_train_steps * config['warmup_proportion'])
        save_checkpoints_steps = int(len(train_examples) / config['train_batch_size']) + 1

    if config['do_train']:
        train_file = os.path.join(config['data_dir'], 'train.tf_record')
        #将example写入tf方便读取的文件
        file_based_convert_examples_to_features(train_examples, domain2id, intent2id, slot2id,\
            config['max_seq_length'], tokenizer, train_file)

        #文件读取模块
        train_input_fn = file_based_input_fn_builder(
            input_file = train_file,
            seq_length = config['max_seq_length'],
            is_training = True,
            drop_remainder = False)
    #2.创建模型
    #2.1设置模型运行参数
    bert_config = modeling.BertConfig.from_json_file(config['bert_config_file'])

    tf_cfg = tf.ConfigProto()
    tf_cfg.gpu_options.per_process_gpu_memory_fraction = 0.8

    run_config = tf.estimator.RunConfig(
        model_dir = config['output_dir'],
        save_checkpoints_steps = save_checkpoints_steps,
        keep_checkpoint_max = 1,
        session_config = tf_cfg,
        log_step_count_steps = 100,)
    #2.1构建模型
    model_fn = model_fn_builder(
        bert_config = bert_config,
        num_domain = len(domain2id),
        num_intent = len(intent2id),
        num_slot = len(slot2id),
        init_checkpoint = config['init_checkpoint'],
        learning_rate = config['learning_rate'],
        num_train_steps = num_train_steps,
        num_warmup_steps = num_warmup_steps,
        use_tpu = config['use_tpu'],
        use_one_hot_embeddings = config['use_tpu'],
        do_serve = config['do_serve'])

    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        config = run_config,
    )

    #3训练
    if config['do_train']:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", config['train_batch_size'])
        tf.logging.info("  Num steps = %d", num_train_steps)
        if config['do_eval']:
            #没有eval环节
            train_spec = tf.estimator.TrainSpec(input_fn = train_input_fn,\
                                                max_steps = num_train_steps)
            eval_spec = tf.estimator.EvalSpec(input_fn = eval_input_fn,\
                                              steps = eval_steps, start_delay_secs=60, throttle_secs=0)
            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)    
        else:    
            estimator.train(input_fn = train_input_fn, max_steps = num_train_steps)

        return None

    #4预测
    #4.1加载预测数据
    if config['do_predict']:
        tf.logging.info("***** Loading training examples *****")
        test_examples = processor.get_test_examples(test_file)
        num_actual_predict_examples = len(test_examples)
        tf.logging.info("the number of test_examples is %d" % len(test_examples))
        test_features = convert_examples_to_features(test_examples, domain2id,\
                intent2id, slot2id, config['max_seq_length'], tokenizer)
        tf.logging.info("the number of test_features is %d" % len(test_features))

    if config['do_predict']:
        predict_input_fn = input_fn_builder(
            features = test_features,
            seq_length = config['max_seq_length'],
            is_training = False,
            drop_remainder = False,
        )
        result = estimator.predict(input_fn=predict_input_fn)
        print(result)
        pred_results = []
        for pred_line, prediction in zip(test_examples, result):
            data = {}
            #print(pred_line.text)
            data['text'] = pred_line.text
            domain_pred = prediction["domain_pred"]
            intent_pred = prediction["intent_pred"]
            slot_pred = prediction["slot_pred"]
            data['domain'] = id2domain[domain_pred] 

            data['intent'] = id2intent[intent_pred] if id2intent[intent_pred] != 'NaN' else np.nan
            idx = 0
            len_seq = len(pred_line.text)
            slot_labels = []
            for sid in slot_pred:
                if idx >= len_seq:
                    break
                if sid == 0:
                    continue
                cur_slot = id2slot[sid]
                if cur_slot in ['[CLS]', '[SEP]']:
                    continue
                slot_labels.append(cur_slot)
                idx += 1

            data['slots'] = get_slot_name(pred_line.text, slot_labels)

            for p in code_pattern:
                result = re.match(p, data['text'])
                if result:
                    #print(result.group(1))
                    #print(result.group(0), result.group(1))
                    data['slots']['code'] = result.group(1)
                    break
            

            pred_results.append(data)

            #print(domain_pred, intent_pred, slot_pred)
    json.dump(pred_results, open(sys.argv[2], 'w', encoding='utf8'), ensure_ascii=False)


if __name__ == '__main__':
    test_file = sys.argv[1]
    print(test_file)
    main(test_file)

