# -*- coding: utf-8 -*-
# @Time : 2021/1/14 11:26
# @Author : Jclian91
# @File : tf_hub_usage.py
# @Place : Yangpu, Shanghai
import tensorflow_text
import tensorflow_hub as hub

# Load BERT and the preprocessing model from TF Hub.
preprocess = hub.load('bert_en_uncased_preprocess_2')
encoder = hub.load('bert_en_uncased_L-12_H-768_A-12_3')

# Use BERT on a batch of raw text inputs.
input = preprocess(['Batch of inputs', 'TF Hub makes BERT easy!', 'More text.'])
pooled_output = encoder(input)["pooled_output"]
print(pooled_output)

text_test = ['this is such an amazing movie!']
text_preprocessed = preprocess(text_test)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')