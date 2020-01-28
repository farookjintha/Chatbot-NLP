# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:52:10 2020

@author: farookjintha
"""

import numpy as np
import tensorflow as tf
import re
import time


#Importing the dataset
#lines = open(r"C:\Users\farookjintha\Desktop\Chatbot-NLP\movie_lines.txt", encoding = 'utf-8', errors = 'ignore').read().split('\n')
#conversations = open(r"C:\Users\farookjintha\Desktop\Chatbot-NLP\movie_conversations.txt", encoding = 'utf-8', errors = 'ignore').read().split('\n')

lines = open(r"C:\Users\751026\Desktop\Projects\PYTHON\Chatbot-NLP\movie_lines.txt", encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open(r"C:\Users\751026\Desktop\Projects\PYTHON\Chatbot-NLP\movie_conversations.txt", encoding = 'utf-8', errors = 'ignore').read().split('\n')


id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

#Creating List of all conversations        
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ", "")
    conversations_ids.append(_conversation.split(','))
    
#Getting questions and answers seperately
questions = []
answers = []

for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
#Cleaning 1
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'v", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"!@#$&<>+=~|.,?'_]", "", text)
    return text

#Cleaning questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

#Cleaning answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

#Creating a dictionary that actually maps each word to its no. of occurences!
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

#Creating 2 dictionaries that map question words and answer words to a unique integer
threshold = 20
questionswords2int = {}
word_number = 0

for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1

answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1

#Adding last tokens to these two dictionaries   
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1

for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

#Creating the inverse dictionary of the answerswords2int dictionary
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}
    
#Adding <EOS> to the end of every answers
for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>"
    
#Translating all the words in questions and answers to integers
#Replacing the filtered words with <OUT> tag.
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
    
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)

#Sorting questions and answers by the length of the questions
sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1, 25+1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])

#BUILDING SEQ2SEQ MODEL

#Creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input') #input placeholder
    targets = tf.placeholder(tf.int32, [None, None], name = 'target') #target placeholder
    lr = tf.placeholder(tf.float32, name = 'learning_rate') #learning_rate placeholder
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob') #keep_prob placeholder
    return inputs, targets, lr, keep_prob

#Preprocessing the target
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1) #AXIS : 1 for horizontal concat, 0 for vertical concat
    return preprocessed_targets

#Creating the Encoder RNN Layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32) #Only second variable we needed i.e., encoder_state
    return encoder_state

#Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input,
                        sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state  = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                               training_decoder_function,
                                                                                                               decoder_embedded_input,
                                                                                                               sequence_length,
                                                                                                               scope = decoding_scope) #We only needed the first variable i.e., decoder_output
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

#Decoding the test set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words,
                    sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_infer")
    test_predictions, decoder_final_state, decoder_final_context_state  = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                 test_decoder_function,
                                                                                                                 scope = decoding_scope) #We only changes decoder_output to test_predictions that is what we calculated.
    return test_predictions

#Creating the Decoder RNN Layer
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state,
                num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope('decoding') as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]* num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x : tf.contrib.layers.fully_connected(x,
                                                                       num_words,
                                                                       None,
                                                                       scope = decoding_scope,
                                                                       weights_initializer = weights,
                                                                       biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state, 
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length-1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
        return training_predictions, test_predictions
    
#Building the seq2seq model with RNN layers
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words,
                  encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs, 
                                                              answers_num_words+1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0,1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words+1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions

#TRAINING SEQ2SEQ MODEL

#Setting the hyperparameters
epochs = 100  #process of getting the batches into neural networks
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5


#Defining the tf session
tf.reset_default_graph()
session = tf.InteractiveSession()

#Loading the inputs to the seq2seq model
inputs, targets, lr, keep_prob = model_inputs()

#Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

#Getting the shape of  the inputs sensors
input_shape = tf.shape(inputs)

#Getting the training & test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)



