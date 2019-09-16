#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/5/19 10:54 AM
# @Author  : zchai
# -*- coding: utf-8 -*-
import json
import codecs
import sys

'''
    Calculate the sentence accuracy
    Json file format: {
        "text": "",
        "domain": "",
        "intent": "",
        "slots": {
          "name": ""
        }
    }
'''
def sentence_acc(truth_dict_list, pred_dict_list):
    assert len(truth_dict_list) == len(pred_dict_list)

    acc_num = 0
    total_num = len(truth_dict_list)
    for truth_dic, pred_dic in zip(truth_dict_list, pred_dict_list):

        # Determine if the domain and intent are correct
        if truth_dic['domain'] != pred_dic['domain'] \
                or truth_dic['intent'] != pred_dic['intent'] \
                or len(truth_dic['slots']) != len(pred_dic['slots']):
            print("true: ", truth_dic)
            print("pre: ", pred_dic)
            continue
        else:
            # Determine if the slots_key and slots_value are correct
            flag = True
            for key, value in truth_dic['slots'].items():
                if key not in pred_dic['slots']:
                    flag = False
                    break # if there is a key not in predict, flag set as false
                elif pred_dic['slots'][key] != truth_dic['slots'][key]:
                    flag = False # if one not match, flag set as false
                    break

            if flag:
                acc_num += 1
            #else:
                #print("true: ", truth_dic)
                #print("pre: ", pred_dic)

    return float(acc_num) / float(total_num)

def domain_acc(truth_dict_list, pred_dict_list):
    assert len(truth_dict_list) == len(pred_dict_list)
    acc_num = 0
    total_num = len(truth_dict_list)
    for truth_dic, pred_dic in zip(truth_dict_list, pred_dict_list):
        if truth_dic['domain'] == pred_dic['domain']:
            acc_num += 1

    return float(acc_num) / float(total_num)


def intent_acc(truth_dict_list, pred_dict_list):
    assert len(truth_dict_list) == len(pred_dict_list)
    acc_num = 0
    total_num = len(truth_dict_list)
    for truth_dic, pred_dic in zip(truth_dict_list, pred_dict_list):
        if truth_dic['intent'] == pred_dic['intent'] and truth_dic['domain'] == pred_dic['domain']:
            acc_num += 1

    return float(acc_num) / float(total_num)

def slots_acc(truth_dict_list, pred_dict_list):
    assert len(truth_dict_list) == len(pred_dict_list)
    acc_num = 0
    total_num = 0
    for truth_dic, pred_dic in zip(truth_dict_list, pred_dict_list):
        total_num += len(truth_dic['slots'])
        for key, value in truth_dic['slots'].items():
            if key not in pred_dic['slots']:
                continue
            elif pred_dic['slots'][key] == truth_dic['slots'][key]:
                acc_num+=1

    return float(acc_num) / float(total_num)

def slots_f(truth_dict_list, pred_dict_list):
    assert len(truth_dict_list) == len(pred_dict_list)
    correct, p_denominator, r_denominator = 0, 0, 0
    for truth_dic, pred_dic in zip(truth_dict_list, pred_dict_list):
        r_denominator += len(truth_dic['slots'])
        p_denominator += len(pred_dic['slots'])
        for key, value in truth_dic['slots'].items():
            if key not in pred_dic['slots']:
                continue
            elif pred_dic['slots'][key] == truth_dic['slots'][key] and \
                truth_dic['domain'] == pred_dic['domain'] and \
                truth_dic['intent'] == pred_dic['intent']:
                correct += 1
    precision = float(correct) / p_denominator
    recall = float(correct) / r_denominator
    f1 = 2 * precision * recall / (precision + recall) * 1.0

    return f1

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Too few args for this script')
        exit(1)

    with codecs.open(sys.argv[1], 'r', encoding='utf-8') as f:
        fp_truth = json.loads(f.read())

    with codecs.open(sys.argv[2], 'r', encoding='utf-8') as f_pred:
        fp_pred = json.loads(f_pred.read())

    domain_accuracy = domain_acc(fp_truth, fp_pred)
    intent_accuracy = intent_acc(fp_truth, fp_pred)
    slots_f = slots_f(fp_truth, fp_pred)

    sentence_accuracy = sentence_acc(fp_truth, fp_pred)

    print('Domain sentence accuracy : %f' % domain_accuracy)
    print('Intent sentence accuracy : %f' % intent_accuracy)
    print('Slots f score : %f' % slots_f)
    print('Avg sentence accuracy : %f' % sentence_accuracy)
