# -*- coding: utf-8 -*-
# @Time : 2021/1/15 10:03
# @Author : Jclian91
# @File : model_predict.py
# @Place : Yangpu, Shanghai
import tensorflow_text as text
import tensorflow as tf

# load model
dataset_name = 'imdb'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))
reloaded_model = tf.saved_model.load(saved_model_path)


# model predict
def print_my_examples(inputs, results):
    result_for_printing = [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}' for i in range(len(inputs))]
    print(*result_for_printing, sep='\n')
    print()


examples = [
    'this is such an amazing movie!',
    'The movie was great!',
    'The movie was meh.',
    'The movie was okish.',
    'The movie was terrible...'
]

reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))

print('Results from the saved model:')
print_my_examples(examples, reloaded_results)


"""
output: 

Results from the saved model:
input: this is such an amazing movie! : score: 0.998060
input: The movie was great!           : score: 0.994865
input: The movie was meh.             : score: 0.319709
input: The movie was okish.           : score: 0.759805
input: The movie was terrible...      : score: 0.003196
"""