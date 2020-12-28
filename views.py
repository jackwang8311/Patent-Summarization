from django.shortcuts import render

# Create your views here.
#花費時間
import os
os_path = os.getcwd()

from django.http import HttpResponse, HttpResponseRedirect
import time

import zipfile
import pandas as pd
import numpy as np
from gensim.models import word2vec
import tensorflow as tf
import re
import jieba
import nltk
from nltk import tokenize
from pandas import Series, DataFrame
from nltk.corpus import stopwords
import time
import csv
from gensim.models import Phrases
from tensorflow.python.layers.core import Dense




def main_jack():
    import glob
    
    file = open(path + 'stopwords.csv')
    stopword = csv.reader(file,delimiter = ';')
    stopwords1 = []
    for i in stopword:
        ww = i[0]
        stopwords1.append(ww)
    patentNumber = []
    title = []
    summary = []
    claim = []
    novelty = []
    use = []
    advantange = []
    assignee = []
    for file1 in glob.glob(path + 'testing.csv'):
        file2 = open(file1, 'rU',encoding ='latin-1')
        patent= csv.reader(file2,delimiter=';')
        for column in patent:
              aa = column[0]
              bb = column[1]
              cc = column[2]
              dd = column[3]
              ee = column[4]
              ff = column[5]
              gg = column[6]
              patentNumber.append(aa)
              title.append(bb)
              summary.append(cc)
              claim.append(dd)
              novelty.append(ee)
              use.append(ff)
              advantange.append(gg)
    patentNumber = pd.DataFrame(patentNumber, columns= {'Patentnumber'})
    title = pd.DataFrame(title, columns={'title'})
    summary = pd.DataFrame(summary, columns={'summary'})
    claim = pd.DataFrame(claim, columns={'claim'})
    novelty = pd.DataFrame(novelty, columns = {'novelty'})
    use = pd.DataFrame(use, columns = {'use'})
    advantange = pd.DataFrame(advantange, columns = {'advantange'})
    #assignee = pd.DataFrame(assignee, columns = {'assignee'})
    testing = pd.concat([patentNumber,title, summary,claim, novelty,use,advantange],axis=1)
    testing.to_csv(path + 'testing sets.csv')
    #print (testing)
    file2.close()

    import glob
    import os
    file = open(path + 'stopwords.csv')
    stopword = csv.reader(file,delimiter = ';')
    stopwords1 = []
    for i in stopword:
        ww = i[0]
        stopwords1.append(ww)
    patentNumber = []
    title = []
    summary = []
    claim = []
    novelty = []
    use = []
    advantange = []
    assignee = []
    for file1 in glob.glob(path + 'training.csv'):
        file2 = open(file1, 'rU',encoding ='latin-1')
        patent= csv.reader(file2,delimiter=';')
        for column in patent:
              aa = column[0]
              bb = column[1]
              cc = column[2]
              dd = column[3]
              ee = column[4]
              ff = column[5]
              gg = column[6]
              patentNumber.append(aa)
              title.append(bb)
              summary.append(cc)
              claim.append(dd)
              novelty.append(ee)
              use.append(ff)
              advantange.append(gg)
    patentNumber = pd.DataFrame(patentNumber, columns= {'Patentnumber'})
    title = pd.DataFrame(title, columns={'title'})
    summary = pd.DataFrame(summary, columns={'summary'})
    claim = pd.DataFrame(claim, columns={'claim'})
    novelty = pd.DataFrame(novelty, columns = {'novelty'})
    use = pd.DataFrame(use, columns = {'use'})
    advantange = pd.DataFrame(advantange, columns = {'advantange'})
    #assignee = pd.DataFrame(assignee, columns = {'assignee'})
    training = pd.concat([patentNumber,title, summary,claim, novelty,use,advantange],axis=1)
    training.to_csv(path + 'training sets.csv')
    #print (training)
    file2.close()


    def clean_text(text, remove_stopwords = True):
        '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
        
        # Convert words to lower case
        text = str(text)
        text = text.lower()
        #text = text.split()
        
        
        
        # Format words and remove unwanted characters
        text = re.sub(r'https?:\/\/.*[\r\n]*,', '', text, flags=re.MULTILINE)
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'&amp;', '', text) 
        text = re.sub('\d',' ', text)
        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'e.g.', ' ', text)
        text = re.sub(r'\'', ' ', text)
        stops = set(stopwords.words("english"))
        text = text.split()
        text = [w for w in text if not w in stops]
        text = [w for w in text if not w in stopwords1]
        text = " ".join(text)
        ##print (text)
        aa = [text.split()]
        ##print (aa)
        bigram = Phrases(aa, min_count= 1, threshold=2)
        text = bigram[str(text).split()]
        trigram = Phrases(text,min_count=1, threshold=2)
        text = trigram[text]
        text = ' '.join(text)
        return text


    clean_summaries = []
    for summary in training.novelty:
        clean_summaries.append(clean_text(summary))
    #print (clean_summaries)
    clean_texts = []
    for text in training.summary:
          clean_texts.append(clean_text(text))
    #print (clean_texts)


    def count_words(count_dict, text):
        '''Count the number of occurrences of each word in a set of text'''
        for sentence in text:
            for word in sentence.split():
                if word not in count_dict:
                    count_dict[word] = 1
                else:
                    count_dict[word] += 1
    # Find the number of times each word was used and the size of the vocabulary
    word_counts = {}

    count_words(word_counts, clean_summaries)
    count_words(word_counts, clean_texts)
                
    #print("Size of Vocabulary:", len(word_counts))



    # Load Conceptnet Numberbatch's (CN) embeddings, similar to GloVe, but probably better 
    # (https://github.com/commonsense/conceptnet-numberbatch)
    embeddings_index = {}
    ff = pd.read_csv(path + 'word_embedding.csv',encoding = 'latin-1') 
    ff.index = ff.Name


    ff = ff.drop('Name', axis=1)
    ##print (ff)
    embeddings_index = ff
    values = ff.values
    #word = ff.index
    #word = str(word)



    #fe = values.spilt()
    ##print (fe)
    ##print (fe)
    embedding = np.asarray(values[0:], dtype='float32')
    ##print (embedding)
    #embedding = np.asarray(values, dtype='float32')
    ##print (embeddings_index[word])

    #print ('Word embeddings:', len(embeddings_index))
    #print (embeddings_index)


    # Find the number of words that are missing from CN, and are used more than our threshold.
    missing_words = 0
    threshold = 0

    for word, count in word_counts.items():
        if count > threshold:
            if word not in embeddings_index:
                missing_words += 1
                
    missing_ratio = round(missing_words/len(word_counts),4)*100
                
    #print("Number of words missing from CN:", missing_words)
    #print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

    # Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe

    #dictionary to convert words to integers
    vocab_to_int = {} 

    value = 0
    for word, count in word_counts.items():
        if count >= threshold or word in embeddings_index:
            vocab_to_int[word] = value
            value += 1

    # Special tokens that will be added to our vocab
    codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   

    # Add codes to vocab
    for code in codes:
        vocab_to_int[code] = len(vocab_to_int)

    # Dictionary to convert integers to words
    int_to_vocab = {}
    for word, value in vocab_to_int.items():
        int_to_vocab[value] = word

    usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

    #print("Total number of unique words:", len(word_counts))
    #print("Number of words we will use:", len(vocab_to_int))
    #print("Percent of words we will use: {}%".format(usage_ratio))
    # Need to use 300 for embedding dimensions to match CN's vectors.
    embedding_dim = 300
    nb_words = len(vocab_to_int)

    # Create matrix with default values of zero
    word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
    for word, i in vocab_to_int.items():
        if word in embeddings_index:
            word_embedding_matrix[i] = embeddings_index[word]
        else:
            # If word not in CN, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            #embeddings_index[word] = new_embedding
            word_embedding_matrix[i] = new_embedding

    # Check if value matches len(vocab_to_int)
    #print(len(word_embedding_matrix))


    # In[99]:


    def convert_to_ints(text, word_count, unk_count, eos=False):
        '''Convert words in text to an integer.
           If word is not in vocab_to_int, use UNK's integer.
           Total the number of words and UNKs.
           Add EOS token to the end of texts'''
        ints = []
        for sentence in text:
            sentence_ints = []
            for word in sentence.split():
                word_count += 1
                if word in vocab_to_int:
                    sentence_ints.append(vocab_to_int[word])
                else:
                    sentence_ints.append(vocab_to_int["<UNK>"])
                    unk_count += 1
            if eos:
                sentence_ints.append(vocab_to_int["<EOS>"])
            ints.append(sentence_ints)
        return ints, word_count, unk_count
    # Apply convert_to_ints to clean_summaries and clean_texts
    word_count = 0
    unk_count = 0

    int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count)
    int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, eos=True)

    unk_percent = round(unk_count/word_count,4)*100

    #print("Total number of words in headlines:", word_count)
    #print("Total number of UNKs in headlines:", unk_count)
    #print("Percent of words that are UNK: {}%".format(unk_percent))


    # In[100]:


    def create_lengths(text):
        '''Create a data frame of the sentence lengths from a text'''
        lengths = []
        for sentence in text:
            lengths.append(len(sentence))
        return pd.DataFrame(lengths, columns=['counts'])
    lengths_summaries = create_lengths(int_summaries)
    lengths_texts = create_lengths(int_texts)

    #print("Summaries:")
    #print(lengths_summaries.describe())
    #print()
    #print("Texts:")
    #print(lengths_texts.describe())# Inspect the length of texts
    #print(np.percentile(lengths_texts.counts, 90))
    #print(np.percentile(lengths_texts.counts, 95))
    #print(np.percentile(lengths_texts.counts, 99)) #探討門檻值的相關文獻


    # In[101]:


    def unk_counter(sentence):
        '''Counts the number of time UNK appears in a sentence.'''
        unk_count = 0
        for word in sentence:
            if word == vocab_to_int["<UNK>"]:
                unk_count += 1
        return unk_count
    # Sort the summaries and texts by the length of the texts, shortest to longest
    # Limit the length of summaries and texts based on the min and max ranges.
    # Remove reviews that include too many UNKs

    sorted_summaries = []
    sorted_texts = []
    max_text_length = 180
    max_summary_length = 60
    min_length = 2
    unk_text_limit = 0
    unk_summary_limit = 0

    for length in range(min(lengths_texts.counts), max_text_length): 
        for count, words in enumerate(int_summaries):
            if (len(int_summaries[count]) >= min_length and
                len(int_summaries[count]) <= max_summary_length and
                len(int_texts[count]) >= min_length and
                unk_counter(int_summaries[count]) <= unk_summary_limit and
                unk_counter(int_texts[count]) <= unk_text_limit and
                length == len(int_texts[count])
               ):
                sorted_summaries.append(int_summaries[count])
                sorted_texts.append(int_texts[count])
            
    # Compare lengths to ensure they match
    #print(len(sorted_summaries))
    #print(len(sorted_texts))


    # In[102]:


    def model_inputs():
        '''Create palceholders for inputs to the model'''
        
        input_data = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
        max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
        text_length = tf.placeholder(tf.int32, (None,), name='text_length')

        return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length

    def process_encoding_input(target_data, vocab_to_int, batch_size):
        '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
        
        ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

        return dec_input

    def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
        '''Create the encoding layer'''
        
        for layer in range(num_layers):
            with tf.variable_scope('encoder_{}'.format(layer)):
                cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                                  initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
                                                        input_keep_prob = keep_prob)

                cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                                  initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
                                                        input_keep_prob = keep_prob)

                enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                        cell_bw, 
                                                                        rnn_inputs,
                                                                        sequence_length,
                                                                        dtype=tf.float32)
        # Join outputs since we are using a bidirectional RNN
        enc_output = tf.concat(enc_output,2)
        
        return enc_output, enc_state

    def training_decoding_layer(dec_embed_input, summary_length, dec_cell, initial_state, output_layer, 
                                vocab_size, max_summary_length):
        '''Create the training logits'''
        
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=summary_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer) 

        training_logits, *_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                               output_time_major=False,
                                                               impute_finished=True,
                                                               maximum_iterations=max_summary_length)
        return training_logits

    def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                                 max_summary_length, batch_size):
        '''Create the inference logits'''
        
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
        
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    end_token)
                    
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            initial_state,
                                                            output_layer)
                    
        inference_logits, *_ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                output_time_major=False,
                                                                impute_finished=True,
                                                                maximum_iterations=max_summary_length)
        
        return inference_logits

    def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length, 
                       max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
        '''Create the decoding cell and attention for the training and inference decoding layers'''
        
        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                lstm = tf.contrib.rnn.LSTMCell(rnn_size,
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                         input_keep_prob = keep_prob)
        
        output_layer = Dense(vocab_size,
                             kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
        
        attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                      enc_output,
                                                      text_length,
                                                      normalize=False,
                                                      name='BahdanauAttention')

        dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attn_mech, rnn_size)
                
        #initial_state = tf.contrib.seq2seq.AttentionWrapperState(enc_state[0], _zero_state_tensors(rnn_size, batch_size, tf.float32)) 
        initial_state = dec_cell.zero_state(batch_size, tf.float32)
        with tf.variable_scope("decode"):
            training_logits = training_decoding_layer(dec_embed_input, 
                                                      summary_length, 
                                                      dec_cell, 
                                                      initial_state,
                                                      output_layer,
                                                      vocab_size, 
                                                      max_summary_length)
        with tf.variable_scope("decode", reuse=True):
            inference_logits = inference_decoding_layer(embeddings,  
                                                        vocab_to_int['<GO>'], 
                                                        vocab_to_int['<EOS>'],
                                                        dec_cell, 
                                                        initial_state, 
                                                        output_layer,
                                                        max_summary_length,
                                                        batch_size)

        return training_logits, inference_logits

    def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length, 
                      vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):
        '''Use the previous functions to create the training and inference logits'''
        
        # Use Numberbatch's embeddings and the newly created ones as our embeddings
        embeddings = word_embedding_matrix
        
        enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
        enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)
        
        dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)
        dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
        
        training_logits, inference_logits  = decoding_layer(dec_embed_input, 
                                                            embeddings,
                                                            enc_output,
                                                            enc_state, 
                                                            vocab_size, 
                                                            text_length, 
                                                            summary_length, 
                                                            max_summary_length,
                                                            rnn_size, 
                                                            vocab_to_int, 
                                                            keep_prob, 
                                                            batch_size,
                                                            num_layers)
        
        return training_logits, inference_logits

    def pad_sentence_batch(sentence_batch):
        """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def get_batches(summaries, texts, batch_size):
        """Batch summaries, texts, and the lengths of their sentences together"""
        for batch_i in range(0, len(texts)//batch_size):
            start_i = batch_i * batch_size
            summaries_batch = summaries[start_i:start_i + batch_size]
            texts_batch = texts[start_i:start_i + batch_size]
            pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
            pad_texts_batch = np.array(pad_sentence_batch(texts_batch))
            
            # Need the lengths for the _lengths parameters
            pad_summaries_lengths = []
            for summary in pad_summaries_batch:
                pad_summaries_lengths.append(len(summary))
            
            pad_texts_lengths = []
            for text in pad_texts_batch:
                pad_texts_lengths.append(len(text))
            
            yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths


    # In[103]:


    # Set the Hyperparameters
    epochs = 100
    batch_size = 20
    rnn_size = 200
    num_layers = 2
    learning_rate = 0.005
    keep_probability = 0.75

    # Build the graph
    train_graph = tf.Graph()
    # Set the graph to default to ensure that it is ready for training
    with train_graph.as_default():
        
        # Load the model inputs    
        input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()

        # Create the training and inference logits
        training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                          targets, 
                                                          keep_prob,   
                                                          text_length,
                                                          summary_length,
                                                          max_summary_length,
                                                          len(vocab_to_int)+1,
                                                          rnn_size, 
                                                          num_layers, 
                                                          vocab_to_int,
                                                          batch_size)
        
        # Create tensors for the training logits and inference logits
        training_logits = tf.identity(training_logits.rnn_output, 'logits')
        inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
        
        # Create the weights for sequence_loss
        masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
    #print("Graph is built.")


    # In[104]:


    # Train the Model
    learning_rate_decay = 0.95
    min_learning_rate = 0.0005
    display_step = 20 # Check training loss after every 20 batches
    stop_early = 0 
    stop = 10 # If the update loss does not decrease in 3 consecutive update checks, stop training
    per_epoch = 3 # Make 3 update checks per epoch
    update_check = (len(sorted_texts)//batch_size//per_epoch)-1

    update_loss = 0 
    batch_loss = 0
    summary_update_loss = [] # Record the update losses for saving improvements in the model

    checkpoint = "Summarization/static/Summarization/TF//./best_model.ckpt"
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        
        # If we want to continue training a previous session
        #loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
        #loader.restore(sess, checkpoint)
        
        for epoch_i in range(1, epochs+1):
            update_loss = 0
            batch_loss = 0
            for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                    get_batches(sorted_summaries, sorted_texts, batch_size)):
                start_time = time.time()
                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: texts_batch,
                     targets: summaries_batch,
                     lr: learning_rate,
                     summary_length: summaries_lengths,
                     text_length: texts_lengths,
                     keep_prob: keep_probability})

                batch_loss += loss
                update_loss += loss
                end_time = time.time()
                batch_time = end_time - start_time
                '''
                if batch_i % display_step == 0 and batch_i > 0:
                    #print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs, 
                                  batch_i, 
                                  len(sorted_texts) // batch_size, 
                                  batch_loss / display_step, 
                                  batch_time*display_step))
                    batch_loss = 0'''

                if batch_i % update_check == 0 and batch_i > 0:
                    #print("Average loss for this update:", round(update_loss/update_check,3))
                    summary_update_loss.append(update_loss)
                    
                # If the update loss is at a new minimum, save the model
                    if update_loss <= min(summary_update_loss):
                        #print('New Record!') 
                        stop_early = 0
                        saver = tf.train.Saver() 
                        saver.save(sess, checkpoint)

                    else:
                        #print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break
                    update_loss = 0
                
                        
            # Reduce learning rate, but not below its minimum value
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            
            if stop_early == stop:
                #print("Stopping Training.")
                break


    # In[190]:


    def text_to_seq(text):
        text = clean_text(text, remove_stopwords = True)
         
        return [vocab_to_int.get(word, vocab_to_int['<UNK>'])for word in text.split()]
    from pandas import Series, DataFrame
    final = []
    for i in range(0,len(testing.summary)):  
       
            text = text_to_seq(testing.summary[i])
            #Summarization/static/Summarization/TF//./best_model.ckpt
            #checkpoint = "TF//./best_model.ckpt"
            checkpoint = "Summarization/static/Summarization/TF//./best_model.ckpt"
            loaded_graph = tf.Graph()
            with tf.Session(graph=loaded_graph) as sess:
                loader = tf.train.import_meta_graph(checkpoint + '.meta')
                loader.restore(sess, checkpoint)
                input_data = loaded_graph.get_tensor_by_name('input:0')
                logits = loaded_graph.get_tensor_by_name('predictions:0')
                text_length = loaded_graph.get_tensor_by_name('text_length:0')
                summary_length =  loaded_graph.get_tensor_by_name('summary_length:0')
                keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
                answer_logits = sess.run(logits, {input_data: [text]*batch_size, summary_length: [np.random.randint(40,45)], text_length: [len(text)]*batch_size, keep_prob: 1.0})[0]
                pad = vocab_to_int["<PAD>"]
                aaa = ([int_to_vocab[i] for i in  answer_logits if i != pad])
                aaa += '.'
                final.extend(aaa)
    #print(final)


    # In[191]:


    finals = []
    import numpy
    ww = ' '.join(final)
    ww = ww.split('.')
    for i in ww:
        i=i.strip()
        i+= '.'
        finals.append(i)
    attentionword = ' '.join(set(final))
    finals = re.sub(r'[_"\;%()|+&=*%!?:#$@\[\]/]', '', str(finals))
    finals = re.sub(r'\'','',finals)
    keyword = finals.split('.')
    inputtext = testing.title + '. ' +  testing.summary + '. ' + testing.claim
    novelty = []
    advantange = []
    for x in range(0,len(testing.summary)):
        aa = keyword[x]
        kk=set(aa.split())

        ee  = inputtext[x].lower()
        ee = re.sub(r'\d','',ee)
        ee = re.sub(r'[_"\;%()|+&=*%!?:#$@\[\]/]', '', ee)
        ee = ee.split('.')
        jj =[]
        attention = []
        for i in ee:                                                      
            dd=set(i.split())
            ##print (i)
            ##print (numpy.float64(len(dd&kk))/numpy.float64(len(dd)))
            jj.append(numpy.float64(len(dd&kk))/numpy.float64(len(dd)))
            attention.append(' '.join(dd&kk))
        qq = pd.DataFrame(ee,index = jj)   
        qq = qq.sort_index(axis = 0, ascending=False)
        qq = qq.iloc[0] + '. ' + qq.iloc[1] + '. ' + qq.iloc[2]
        for i in qq:
            i += '.'
            novelty.append(i)
    novelty = ' '.join(novelty)
    #print (novelty)
    novelty = pd.DataFrame([novelty], index = {'Novelty'})
    keywords = pd.DataFrame([attentionword], index = {'Keyword'})


    # In[192]:


    def clean_text(text, remove_stopwords = True):
        '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
        
        # Convert words to lower case
        text = str(text)
        text = text.lower()
        #text = text.split()
        
        
        
        # Format words and remove unwanted characters
        text = re.sub(r'https?:\/\/.*[\r\n]*,', '', text, flags=re.MULTILINE)
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'&amp;', '', text) 
        text = re.sub('\d',' ', text)
        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'e.g.', ' ', text)
        text = re.sub(r'\'', ' ', text)
        stops = set(stopwords.words("english"))
        text = text.split()
        text = [w for w in text if not w in stops]
        text = [w for w in text if not w in stopwords1]
        text = " ".join(text)
        ##print (text)
        aa = [text.split()]
        ##print (aa)
        bigram = Phrases(aa, min_count= 2, threshold=4)
        text = bigram[str(text).split()]
        trigram = Phrases(text,min_count=1, threshold=2)
        text = trigram[text]
        text = ' '.join(text)
        return text


    clean_summaries = []
    for summary in training.use:
        clean_summaries.append(clean_text(summary))

    clean_texts = []
    for text in training.title:
          clean_texts.append(clean_text(text))
    #print (clean_texts)


    # In[193]:


    def count_words(count_dict, text):
        '''Count the number of occurrences of each word in a set of text'''
        for sentence in text:
            for word in sentence.split():
                if word not in count_dict:
                    count_dict[word] = 1
                else:
                    count_dict[word] += 1
    # Find the number of times each word was used and the size of the vocabulary
    word_counts = {}

    count_words(word_counts, clean_summaries)
    count_words(word_counts, clean_texts)
                
    #print("Size of Vocabulary:", len(word_counts))


    # In[194]:


    # Load Conceptnet Numberbatch's (CN) embeddings, similar to GloVe, but probably better 
    # (https://github.com/commonsense/conceptnet-numberbatch)
    embeddings_index = {}
    ff = pd.read_csv(path + 'word_embedding.csv',encoding = 'latin-1') 
    ff.index = ff.Name


    ff = ff.drop('Name', axis=1)
    ##print (ff)
    embeddings_index = ff
    values = ff.values
    #word = ff.index
    #word = str(word)



    #fe = values.spilt()
    ##print (fe)
    ##print (fe)
    embedding = np.asarray(values[0:], dtype='float32')
    ##print (embedding)
    #embedding = np.asarray(values, dtype='float32')
    ##print (embeddings_index[word])

    #print ('Word embeddings:', len(embeddings_index))
    #print (embeddings_index)


    # In[68]:


    # Find the number of words that are missing from CN, and are used more than our threshold.
    missing_words = 0
    threshold = 0

    for word, count in word_counts.items():
        if count > threshold:
            if word not in embeddings_index:
                missing_words += 1
                
    missing_ratio = round(missing_words/len(word_counts),4)*100
                
    #print("Number of words missing from CN:", missing_words)
    #print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

    # Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe

    #dictionary to convert words to integers
    vocab_to_int = {} 

    value = 0
    for word, count in word_counts.items():
        if count >= threshold or word in embeddings_index:
            vocab_to_int[word] = value
            value += 1

    # Special tokens that will be added to our vocab
    codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   

    # Add codes to vocab
    for code in codes:
        vocab_to_int[code] = len(vocab_to_int)

    # Dictionary to convert integers to words
    int_to_vocab = {}
    for word, value in vocab_to_int.items():
        int_to_vocab[value] = word

    usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

    #print("Total number of unique words:", len(word_counts))
    #print("Number of words we will use:", len(vocab_to_int))
    #print("Percent of words we will use: {}%".format(usage_ratio))
    # Need to use 300 for embedding dimensions to match CN's vectors.
    embedding_dim = 300
    nb_words = len(vocab_to_int)

    # Create matrix with default values of zero
    word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
    for word, i in vocab_to_int.items():
        if word in embeddings_index:
            word_embedding_matrix[i] = embeddings_index[word]
        else:
            # If word not in CN, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            #embeddings_index[word] = new_embedding
            word_embedding_matrix[i] = new_embedding

    # Check if value matches len(vocab_to_int)
    #print(len(word_embedding_matrix))


    # In[196]:


    def convert_to_ints(text, word_count, unk_count, eos=False):
        '''Convert words in text to an integer.
           If word is not in vocab_to_int, use UNK's integer.
           Total the number of words and UNKs.
           Add EOS token to the end of texts'''
        ints = []
        for sentence in text:
            sentence_ints = []
            for word in sentence.split():
                word_count += 1
                if word in vocab_to_int:
                    sentence_ints.append(vocab_to_int[word])
                else:
                    sentence_ints.append(vocab_to_int["<UNK>"])
                    unk_count += 1
            if eos:
                sentence_ints.append(vocab_to_int["<EOS>"])
            ints.append(sentence_ints)
        return ints, word_count, unk_count
    # Apply convert_to_ints to clean_summaries and clean_texts
    word_count = 0
    unk_count = 0

    int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count)
    int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, eos=True)

    unk_percent = round(unk_count/word_count,4)*100

    #print("Total number of words in headlines:", word_count)
    #print("Total number of UNKs in headlines:", unk_count)
    #print("Percent of words that are UNK: {}%".format(unk_percent))


    # In[197]:


    def create_lengths(text):
        '''Create a data frame of the sentence lengths from a text'''
        lengths = []
        for sentence in text:
            lengths.append(len(sentence))
        return pd.DataFrame(lengths, columns=['counts'])
    lengths_summaries = create_lengths(int_summaries)
    lengths_texts = create_lengths(int_texts)

    #print("Summaries:")
    #print(lengths_summaries.describe())
    #print()
    #print("Texts:")
    #print(lengths_texts.describe())# Inspect the length of texts
    #print(np.percentile(lengths_texts.counts, 90))
    #print(np.percentile(lengths_texts.counts, 95))
    #print(np.percentile(lengths_texts.counts, 99)) #探討門檻值的相關文獻


    # In[198]:


    def unk_counter(sentence):
        '''Counts the number of time UNK appears in a sentence.'''
        unk_count = 0
        for word in sentence:
            if word == vocab_to_int["<UNK>"]:
                unk_count += 1
        return unk_count
    # Sort the summaries and texts by the length of the texts, shortest to longest
    # Limit the length of summaries and texts based on the min and max ranges.
    # Remove reviews that include too many UNKs

    sorted_summaries = []
    sorted_texts = []
    max_text_length = 30
    max_summary_length = 20
    min_length = 2
    unk_text_limit = 0
    unk_summary_limit = 0

    for length in range(min(lengths_texts.counts), max_text_length): 
        for count, words in enumerate(int_summaries):
            if (len(int_summaries[count]) >= min_length and
                len(int_summaries[count]) <= max_summary_length and
                len(int_texts[count]) >= min_length and
                unk_counter(int_summaries[count]) <= unk_summary_limit and
                unk_counter(int_texts[count]) <= unk_text_limit and
                length == len(int_texts[count])
               ):
                sorted_summaries.append(int_summaries[count])
                sorted_texts.append(int_texts[count])
            
    # Compare lengths to ensure they match
    #print(len(sorted_summaries))
    #print(len(sorted_texts))


    # In[71]:


    def model_inputs():
        '''Create palceholders for inputs to the model'''
        
        input_data = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
        max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
        text_length = tf.placeholder(tf.int32, (None,), name='text_length')

        return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length

    def process_encoding_input(target_data, vocab_to_int, batch_size):
        '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
        
        ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

        return dec_input

    def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
        '''Create the encoding layer'''
        
        for layer in range(num_layers):
            with tf.variable_scope('encoder_{}'.format(layer)):
                cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                                  initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
                                                        input_keep_prob = keep_prob)

                cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                                  initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
                                                        input_keep_prob = keep_prob)

                enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                        cell_bw, 
                                                                        rnn_inputs,
                                                                        sequence_length,
                                                                        dtype=tf.float32)
        # Join outputs since we are using a bidirectional RNN
        enc_output = tf.concat(enc_output,2)
        
        return enc_output, enc_state

    def training_decoding_layer(dec_embed_input, summary_length, dec_cell, initial_state, output_layer, 
                                vocab_size, max_summary_length):
        '''Create the training logits'''
        
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=summary_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer) 

        training_logits, *_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                               output_time_major=False,
                                                               impute_finished=True,
                                                               maximum_iterations=max_summary_length)
        return training_logits

    def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                                 max_summary_length, batch_size):
        '''Create the inference logits'''
        
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
        
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    end_token)
                    
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            initial_state,
                                                            output_layer)
                    
        inference_logits, *_ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                output_time_major=False,
                                                                impute_finished=True,
                                                                maximum_iterations=max_summary_length)
        
        return inference_logits

    def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length, 
                       max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
        '''Create the decoding cell and attention for the training and inference decoding layers'''
        
        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                lstm = tf.contrib.rnn.LSTMCell(rnn_size,
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                         input_keep_prob = keep_prob)
        
        output_layer = Dense(vocab_size,
                             kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
        
        attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                      enc_output,
                                                      text_length,
                                                      normalize=False,
                                                      name='BahdanauAttention')

        dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                              attn_mech,
                                                              rnn_size)
        initial_state = dec_cell.zero_state(batch_size, tf.float32)     
        #initial_state = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=enc_state)
        with tf.variable_scope("decode"):
            training_logits = training_decoding_layer(dec_embed_input, 
                                                      summary_length, 
                                                      dec_cell, 
                                                      initial_state,
                                                      output_layer,
                                                      vocab_size, 
                                                      max_summary_length)
        with tf.variable_scope("decode", reuse=True):
            inference_logits = inference_decoding_layer(embeddings,  
                                                        vocab_to_int['<GO>'], 
                                                        vocab_to_int['<EOS>'],
                                                        dec_cell, 
                                                        initial_state, 
                                                        output_layer,
                                                        max_summary_length,
                                                        batch_size)

        return training_logits, inference_logits

    def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length, 
                      vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):
        '''Use the previous functions to create the training and inference logits'''
        
        # Use Numberbatch's embeddings and the newly created ones as our embeddings
        embeddings = word_embedding_matrix
        
        enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
        enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)
        
        dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)
        dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
        
        training_logits, inference_logits  = decoding_layer(dec_embed_input, 
                                                            embeddings,
                                                            enc_output,
                                                            enc_state, 
                                                            vocab_size, 
                                                            text_length, 
                                                            summary_length, 
                                                            max_summary_length,
                                                            rnn_size, 
                                                            vocab_to_int, 
                                                            keep_prob, 
                                                            batch_size,
                                                            num_layers)
        
        return training_logits, inference_logits

    def pad_sentence_batch(sentence_batch):
        """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def get_batches(summaries, texts, batch_size):
        """Batch summaries, texts, and the lengths of their sentences together"""
        for batch_i in range(0, len(texts)//batch_size):
            start_i = batch_i * batch_size
            summaries_batch = summaries[start_i:start_i + batch_size]
            texts_batch = texts[start_i:start_i + batch_size]
            pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
            pad_texts_batch = np.array(pad_sentence_batch(texts_batch))
            
            # Need the lengths for the _lengths parameters
            pad_summaries_lengths = []
            for summary in pad_summaries_batch:
                pad_summaries_lengths.append(len(summary))
            
            pad_texts_lengths = []
            for text in pad_texts_batch:
                pad_texts_lengths.append(len(text))
            
            yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths


    # In[72]:


    # Set the Hyperparameters
    epochs = 100
    batch_size = 20
    rnn_size = 200
    num_layers = 2
    learning_rate = 0.005
    keep_probability = 0.75

    # Build the graph
    train_graph = tf.Graph()
    # Set the graph to default to ensure that it is ready for training
    with train_graph.as_default():
        
        # Load the model inputs    
        input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()

        # Create the training and inference logits
        training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                          targets, 
                                                          keep_prob,   
                                                          text_length,
                                                          summary_length,
                                                          max_summary_length,
                                                          len(vocab_to_int)+1,
                                                          rnn_size, 
                                                          num_layers, 
                                                          vocab_to_int,
                                                          batch_size)
        
        # Create tensors for the training logits and inference logits
        training_logits = tf.identity(training_logits.rnn_output, 'logits')
        inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
        
        # Create the weights for sequence_loss
        masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
    #print("Graph is built.")


    # In[ ]:


    # Train the Model
    learning_rate_decay = 0.95
    min_learning_rate = 0.0005
    display_step = 20 # Check training loss after every 20 batches
    stop_early = 0 
    stop = 5 # If the update loss does not decrease in 3 consecutive update checks, stop training
    per_epoch = 3 # Make 3 update checks per epoch
    update_check = (len(sorted_texts)//batch_size//per_epoch)-1

    update_loss = 0 
    batch_loss = 0
    summary_update_loss = [] # Record the update losses for saving improvements in the model

    checkpoint = "Summarization/static/Summarization/TF//./best_model.ckpt"
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        
        # If we want to continue training a previous session
        #loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
        #loader.restore(sess, checkpoint)
        
        for epoch_i in range(1, epochs+1):
            update_loss = 0
            batch_loss = 0
            for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                    get_batches(sorted_summaries, sorted_texts, batch_size)):
                start_time = time.time()
                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: texts_batch,
                     targets: summaries_batch,
                     lr: learning_rate,
                     summary_length: summaries_lengths,
                     text_length: texts_lengths,
                     keep_prob: keep_probability})

                batch_loss += loss
                update_loss += loss
                end_time = time.time()
                batch_time = end_time - start_time
                '''if batch_i % display_step == 0 and batch_i > 0:
                    #print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs, 
                                  batch_i, 
                                  len(sorted_texts) // batch_size, 
                                  batch_loss / display_step, 
                                  batch_time*display_step))
                    batch_loss = 0'''

                if batch_i % update_check == 0 and batch_i > 0:
                    #print("Average loss for this update:", round(update_loss/update_check,3))
                    summary_update_loss.append(update_loss)
                    
                # If the update loss is at a new minimum, save the model
                    if update_loss <= min(summary_update_loss):
                        #print('New Record!') 
                        stop_early = 0
                        saver = tf.train.Saver() 
                        saver.save(sess, checkpoint)

                    else:
                        #print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break
                    update_loss = 0
                
                        
            # Reduce learning rate, but not below its minimum value
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            
            if stop_early == stop:
                #print("Stopping Training.")
                break


    # In[202]:


    def text_to_seq(text):
        text = clean_text(text, remove_stopwords = True)
         
        return [vocab_to_int.get(word, vocab_to_int['<UNK>'])for word in text.split()]
    from pandas import Series, DataFrame
    final = []
    for i in range(0,len(testing.title)):  
       
            text = text_to_seq(testing.title[i])
            checkpoint = "Summarization/static/Summarization/TF//./best_model.ckpt"
            loaded_graph = tf.Graph()
            with tf.Session(graph=loaded_graph) as sess:
                loader = tf.train.import_meta_graph(checkpoint + '.meta')
                loader.restore(sess, checkpoint)
                input_data = loaded_graph.get_tensor_by_name('input:0')
                logits = loaded_graph.get_tensor_by_name('predictions:0')
                text_length = loaded_graph.get_tensor_by_name('text_length:0')
                summary_length =  loaded_graph.get_tensor_by_name('summary_length:0')
                keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
                answer_logits = sess.run(logits, {input_data: [text]*batch_size, summary_length: [np.random.randint(40,45)], text_length: [len(text)]*batch_size, keep_prob: 1.0})[0]
                pad = vocab_to_int["<PAD>"]
                aaa = ([int_to_vocab[i] for i in  answer_logits if i != pad])
                aaa += '.'
                final.extend(aaa)
    #print(final)


    # In[203]:


    finals = []
    ww = ' '.join(final)
    ww = ww.split('.')
    for i in ww:
        i=i.strip()
        i+= '.'
        finals.append(i)
    finals = re.sub(r'[_"\;%()|+&=*%!?:#$@\[\]/]', '', str(finals))
    finals = re.sub(r'\'','',finals)
    inputtext = testing.title + '. ' +  testing.summary + '. ' + testing.claim
    keyword = finals.split('.')
    use = []
    for x in range(0,len(testing.summary)):
        aa = keyword[x]
        kk=set(aa.split())

        ee  = inputtext[x].lower()
        ee = re.sub(r'\d','',ee)
        ee = re.sub(r'[_"\;%()|+&=*%!?:#$@\[\]/]', '', ee)
        ee = ee.split('.')
        jj =[]
        for i in ee:
            ##print (i)
            dd=set(i.split())
            ##print (numpy.float64(len(dd&kk))/numpy.float64(len(dd)))
            jj.append (numpy.float64(len(dd&kk))/numpy.float64(len(dd)))
        
        qq = pd.DataFrame(ee,index = jj)   
        qq = qq.sort_index(axis = 0, ascending=False)
        
        for i in qq.iloc[0]:
            i += '.'
            use.append(i)
        
    use = ' '.join(use)
    #print (use)
    use = pd.DataFrame([use], index = {'Use'})
    #summary = pd.concat([novelty,use,advantange],axis = 0)
    #summary.to_csv('D:/Documents/Desktop/data/output/decision_making_intelligent_summary_planning.csv')


    # In[204]:


    def clean_text(text, remove_stopwords = True):
        '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
        
        # Convert words to lower case
        text = str(text)
        text = text.lower()
        #text = text.split()
        
        
        
        # Format words and remove unwanted characters
        text = re.sub(r'https?:\/\/.*[\r\n]*,', '', text, flags=re.MULTILINE)
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'&amp;', '', text) 
        text = re.sub('\d',' ', text)
        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'e.g.', ' ', text)
        text = re.sub(r'\'', ' ', text)
        stops = set(stopwords.words("english"))
        text = text.split()
        text = [w for w in text if not w in stops]
        text = [w for w in text if not w in stopwords1]
        text = " ".join(text)
        ##print (text)
        aa = [text.split()]
        ##print (aa)
        bigram = Phrases(aa, min_count= 2, threshold=4)
        text = bigram[str(text).split()]
        trigram = Phrases(text,min_count=1, threshold=2)
        text = trigram[text]
        text = ' '.join(text)
        return text


    clean_summaries = []
    for summary in training.advantange:
        clean_summaries.append(clean_text(summary))

    clean_texts = []
    for text in training.summary:
          clean_texts.append(clean_text(text))
    #print (clean_texts)


    # In[205]:


    def count_words(count_dict, text):
        '''Count the number of occurrences of each word in a set of text'''
        for sentence in text:
            for word in sentence.split():
                if word not in count_dict:
                    count_dict[word] = 1
                else:
                    count_dict[word] += 1
    # Find the number of times each word was used and the size of the vocabulary
    word_counts = {}

    count_words(word_counts, clean_summaries)
    count_words(word_counts, clean_texts)
                
    #print("Size of Vocabulary:", len(word_counts))


    # In[206]:


    # Load Conceptnet Numberbatch's (CN) embeddings, similar to GloVe, but probably better 
    # (https://github.com/commonsense/conceptnet-numberbatch)
    embeddings_index = {}
    ff = pd.read_csv(path + 'word_embedding.csv',encoding = 'latin-1') 
    ff.index = ff.Name


    ff = ff.drop('Name', axis=1)
    ##print (ff)
    embeddings_index = ff
    values = ff.values
    #word = ff.index
    #word = str(word)



    #fe = values.spilt()
    ##print (fe)
    ##print (fe)
    embedding = np.asarray(values[0:], dtype='float32')
    ##print (embedding)
    #embedding = np.asarray(values, dtype='float32')
    ##print (embeddings_index[word])

    #print ('Word embeddings:', len(embeddings_index))
    #print (embeddings_index)


    # In[61]:


    # Find the number of words that are missing from CN, and are used more than our threshold.
    missing_words = 0
    threshold = 0

    for word, count in word_counts.items():
        if count > threshold:
            if word not in embeddings_index:
                missing_words += 1
                
    missing_ratio = round(missing_words/len(word_counts),4)*100
                
    #print("Number of words missing from CN:", missing_words)
    #print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

    # Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe

    #dictionary to convert words to integers
    vocab_to_int = {} 

    value = 0
    for word, count in word_counts.items():
        if count >= threshold or word in embeddings_index:
            vocab_to_int[word] = value
            value += 1

    # Special tokens that will be added to our vocab
    codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   

    # Add codes to vocab
    for code in codes:
        vocab_to_int[code] = len(vocab_to_int)

    # Dictionary to convert integers to words
    int_to_vocab = {}
    for word, value in vocab_to_int.items():
        int_to_vocab[value] = word

    usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

    #print("Total number of unique words:", len(word_counts))
    #print("Number of words we will use:", len(vocab_to_int))
    #print("Percent of words we will use: {}%".format(usage_ratio))
    # Need to use 300 for embedding dimensions to match CN's vectors.
    embedding_dim = 300
    nb_words = len(vocab_to_int)

    # Create matrix with default values of zero
    word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
    for word, i in vocab_to_int.items():
        if word in embeddings_index:
            word_embedding_matrix[i] = embeddings_index[word]
        else:
            # If word not in CN, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            #embeddings_index[word] = new_embedding
            word_embedding_matrix[i] = new_embedding

    # Check if value matches len(vocab_to_int)
    #print(len(word_embedding_matrix))


    # In[62]:


    def convert_to_ints(text, word_count, unk_count, eos=False):
        '''Convert words in text to an integer.
           If word is not in vocab_to_int, use UNK's integer.
           Total the number of words and UNKs.
           Add EOS token to the end of texts'''
        ints = []
        for sentence in text:
            sentence_ints = []
            for word in sentence.split():
                word_count += 1
                if word in vocab_to_int:
                    sentence_ints.append(vocab_to_int[word])
                else:
                    sentence_ints.append(vocab_to_int["<UNK>"])
                    unk_count += 1
            if eos:
                sentence_ints.append(vocab_to_int["<EOS>"])
            ints.append(sentence_ints)
        return ints, word_count, unk_count
    # Apply convert_to_ints to clean_summaries and clean_texts
    word_count = 0
    unk_count = 0

    int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count)
    int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, eos=True)

    unk_percent = round(unk_count/word_count,4)*100

    #print("Total number of words in headlines:", word_count)
    #print("Total number of UNKs in headlines:", unk_count)
    #print("Percent of words that are UNK: {}%".format(unk_percent))


    # In[63]:





    def create_lengths(text):
        '''Create a data frame of the sentence lengths from a text'''
        lengths = []
        for sentence in text:
            lengths.append(len(sentence))
        return pd.DataFrame(lengths, columns=['counts'])
    lengths_summaries = create_lengths(int_summaries)
    lengths_texts = create_lengths(int_texts)

    #print("Summaries:")
    #print(lengths_summaries.describe())
    #print()
    #print("Texts:")
    #print(lengths_texts.describe())# Inspect the length of texts
    #print(np.percentile(lengths_texts.counts, 90))
    #print(np.percentile(lengths_texts.counts, 95))
    #print(np.percentile(lengths_texts.counts, 99)) #探討門檻值的相關文獻


    # In[64]:



















    def unk_counter(sentence):
        '''Counts the number of time UNK appears in a sentence.'''
        unk_count = 0
        for word in sentence:
            if word == vocab_to_int["<UNK>"]:
                unk_count += 1
        return unk_count
    # Sort the summaries and texts by the length of the texts, shortest to longest
    # Limit the length of summaries and texts based on the min and max ranges.
    # Remove reviews that include too many UNKs

    sorted_summaries = []
    sorted_texts = []
    max_text_length = 150
    max_summary_length = 50
    min_length = 2
    unk_text_limit = 0
    unk_summary_limit = 0

    for length in range(min(lengths_texts.counts), max_text_length): 
        for count, words in enumerate(int_summaries):
            if (len(int_summaries[count]) >= min_length and
                len(int_summaries[count]) <= max_summary_length and
                len(int_texts[count]) >= min_length and
                unk_counter(int_summaries[count]) <= unk_summary_limit and
                unk_counter(int_texts[count]) <= unk_text_limit and
                length == len(int_texts[count])
               ):
                sorted_summaries.append(int_summaries[count])
                sorted_texts.append(int_texts[count])
            
    # Compare lengths to ensure they match
    #print(len(sorted_summaries))
    #print(len(sorted_texts))


    # In[65]:


    def model_inputs():
        '''Create palceholders for inputs to the model'''
        
        input_data = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
        max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
        text_length = tf.placeholder(tf.int32, (None,), name='text_length')

        return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length

    def process_encoding_input(target_data, vocab_to_int, batch_size):
        '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
        
        ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

        return dec_input

    def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
        '''Create the encoding layer'''
        
        for layer in range(num_layers):
            with tf.variable_scope('encoder_{}'.format(layer)):
                cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                                  initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
                                                        input_keep_prob = keep_prob)

                cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                                  initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
                                                        input_keep_prob = keep_prob)

                enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                        cell_bw, 
                                                                        rnn_inputs,
                                                                        sequence_length,
                                                                        dtype=tf.float32)
        # Join outputs since we are using a bidirectional RNN
        enc_output = tf.concat(enc_output,2)
        
        return enc_output, enc_state

    def training_decoding_layer(dec_embed_input, summary_length, dec_cell, initial_state, output_layer, 
                                vocab_size, max_summary_length):
        '''Create the training logits'''
        
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=summary_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer) 

        training_logits, *_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                               output_time_major=False,
                                                               impute_finished=True,
                                                               maximum_iterations=max_summary_length)
        return training_logits

    def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                                 max_summary_length, batch_size):
        '''Create the inference logits'''
        
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
        
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    end_token)
                    
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            initial_state,
                                                            output_layer)
                    
        inference_logits, *_ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                output_time_major=False,
                                                                impute_finished=True,
                                                                maximum_iterations=max_summary_length)
        
        return inference_logits

    def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length, 
                       max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
        '''Create the decoding cell and attention for the training and inference decoding layers'''
        
        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                lstm = tf.contrib.rnn.LSTMCell(rnn_size,
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                         input_keep_prob = keep_prob)
        
        output_layer = Dense(vocab_size,
                             kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
        
        attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                      enc_output,
                                                      text_length,
                                                      normalize=False,
                                                      name='BahdanauAttention')

        dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                              attn_mech,
                                                              rnn_size)
        initial_state = dec_cell.zero_state(batch_size, tf.float32) 
        #initial_state = dec_cell.zero_state(batch_size=batch_size, dtype = tf.float32).clone(cell_state=enc_state) 
        with tf.variable_scope("decode"):
            training_logits = training_decoding_layer(dec_embed_input, 
                                                      summary_length, 
                                                      dec_cell, 
                                                      initial_state,
                                                      output_layer,
                                                      vocab_size, 
                                                      max_summary_length)
        with tf.variable_scope("decode", reuse=True):
            inference_logits = inference_decoding_layer(embeddings,  
                                                        vocab_to_int['<GO>'], 
                                                        vocab_to_int['<EOS>'],
                                                        dec_cell, 
                                                        initial_state, 
                                                        output_layer,
                                                        max_summary_length,
                                                        batch_size)

        return training_logits, inference_logits

    def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length, 
                      vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):
        '''Use the previous functions to create the training and inference logits'''
        
        # Use Numberbatch's embeddings and the newly created ones as our embeddings
        embeddings = word_embedding_matrix
        
        enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
        enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)
        
        dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)
        dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
        
        training_logits, inference_logits  = decoding_layer(dec_embed_input, 
                                                            embeddings,
                                                            enc_output,
                                                            enc_state, 
                                                            vocab_size, 
                                                            text_length, 
                                                            summary_length, 
                                                            max_summary_length,
                                                            rnn_size, 
                                                            vocab_to_int, 
                                                            keep_prob, 
                                                            batch_size,
                                                            num_layers)
        
        return training_logits, inference_logits

    def pad_sentence_batch(sentence_batch):
        """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def get_batches(summaries, texts, batch_size):
        """Batch summaries, texts, and the lengths of their sentences together"""
        for batch_i in range(0, len(texts)//batch_size):
            start_i = batch_i * batch_size
            summaries_batch = summaries[start_i:start_i + batch_size]
            texts_batch = texts[start_i:start_i + batch_size]
            pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
            pad_texts_batch = np.array(pad_sentence_batch(texts_batch))
            
            # Need the lengths for the _lengths parameters
            pad_summaries_lengths = []
            for summary in pad_summaries_batch:
                pad_summaries_lengths.append(len(summary))
            
            pad_texts_lengths = []
            for text in pad_texts_batch:
                pad_texts_lengths.append(len(text))
            
            yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths


    # In[66]:


    # Set the Hyperparameters
    epochs = 100
    batch_size = 20
    rnn_size = 200
    num_layers = 2
    learning_rate = 0.005
    keep_probability = 0.75

    # Build the graph
    train_graph = tf.Graph()
    # Set the graph to default to ensure that it is ready for training
    with train_graph.as_default():
        
        # Load the model inputs    
        input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()

        # Create the training and inference logits
        training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                          targets, 
                                                          keep_prob,   
                                                          text_length,
                                                          summary_length,
                                                          max_summary_length,
                                                          len(vocab_to_int)+1,
                                                          rnn_size, 
                                                          num_layers, 
                                                          vocab_to_int,
                                                          batch_size)
        
        # Create tensors for the training logits and inference logits
        training_logits = tf.identity(training_logits.rnn_output, 'logits')
        inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
        
        # Create the weights for sequence_loss
        masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
    #print("Graph is built.")


    # In[77]:


    # Train the Model
    learning_rate_decay = 0.95
    min_learning_rate = 0.0005
    display_step = 20 # Check training loss after every 20 batches
    stop_early = 0 
    stop = 5 # If the update loss does not decrease in 3 consecutive update checks, stop training
    per_epoch = 3 # Make 3 update checks per epoch
    update_check = (len(sorted_texts)//batch_size//per_epoch)-1

    update_loss = 0 
    batch_loss = 0
    summary_update_loss = [] # Record the update losses for saving improvements in the model

    checkpoint = "Summarization/static/Summarization/TF//./best_model.ckpt"
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        
        # If we want to continue training a previous session
        #loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
        #loader.restore(sess, checkpoint)
        
        for epoch_i in range(1, epochs+1):
            update_loss = 0
            batch_loss = 0
            for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                    get_batches(sorted_summaries, sorted_texts, batch_size)):
                start_time = time.time()
                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: texts_batch,
                     targets: summaries_batch,
                     lr: learning_rate,
                     summary_length: summaries_lengths,
                     text_length: texts_lengths,
                     keep_prob: keep_probability})

                batch_loss += loss
                update_loss += loss
                end_time = time.time()
                batch_time = end_time - start_time

                '''if batch_i % display_step == 0 and batch_i > 0:
                    #print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs, 
                                  batch_i, 
                                  len(sorted_texts) // batch_size, 
                                  batch_loss / display_step, 
                                  batch_time*display_step))
                    batch_loss = 0'''

                if batch_i % update_check == 0 and batch_i > 0:
                    #print("Average loss for this update:", round(update_loss/update_check,3))
                    summary_update_loss.append(update_loss)
                    
                # If the update loss is at a new minimum, save the model
                    if update_loss <= min(summary_update_loss):
                        #print('New Record!') 
                        stop_early = 0
                        saver = tf.train.Saver() 
                        saver.save(sess, checkpoint)

                    else:
                        #print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break
                    update_loss = 0
                
                        
            # Reduce learning rate, but not below its minimum value
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            
            if stop_early == stop:
                #print("Stopping Training.")
                break


    # In[214]:


    def text_to_seq(text):
        text = clean_text(text, remove_stopwords = True)
         
        return [vocab_to_int.get(word, vocab_to_int['<UNK>'])for word in text.split()]
    from pandas import Series, DataFrame
    final = []
    for i in range(0,len(testing.summary)):  
       
            text = text_to_seq(testing.summary[i])
            checkpoint = "Summarization/static/Summarization/TF//./best_model.ckpt"
            loaded_graph = tf.Graph()
            with tf.Session(graph=loaded_graph) as sess:
                loader = tf.train.import_meta_graph(checkpoint + '.meta')
                loader.restore(sess, checkpoint)
                input_data = loaded_graph.get_tensor_by_name('input:0')
                logits = loaded_graph.get_tensor_by_name('predictions:0')
                text_length = loaded_graph.get_tensor_by_name('text_length:0')
                summary_length =  loaded_graph.get_tensor_by_name('summary_length:0')
                keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
                answer_logits = sess.run(logits, {input_data: [text]*batch_size, summary_length: [np.random.randint(40,45)], text_length: [len(text)]*batch_size, keep_prob: 1.0})[0]
                pad = vocab_to_int["<PAD>"]
                aaa = ([int_to_vocab[i] for i in  answer_logits if i != pad])
                aaa += '.'
                final.extend(aaa)
    #print(final)


    # In[215]:


    finals = []
    ww = ' '.join(final)
    ww = ww.split('.')
    for i in ww:
        i=i.strip()
        i+= '.'
        finals.append(i)
    finals = re.sub(r'[_"\;%()|+&=*%!?:#$@\[\]/]', '', str(finals))
    finals = re.sub(r'\'','',finals)
    keyword = finals.split('.')
    advantange = []
    for x in range(0,len(testing.summary)):
        aa = keyword[x]
        kk=set(aa.split())

        ee  = inputtext[x].lower()
        ee = re.sub(r'\d','',ee)
        ee = re.sub(r'[_"\;%()|+&=*%!?:#$@\[\]/]', '', ee)
        ee = ee.split('.')
        jj =[]
        for i in ee:
            ##print (i)
            dd=set(i.split())
            ##print (numpy.float64(len(dd&kk))/numpy.float64(len(dd)))
            jj.append (numpy.float64(len(dd&kk))/numpy.float64(len(dd)))
        
        qq = pd.DataFrame(ee,index = jj)   
        qq = qq.sort_index(axis = 0, ascending=False)
        
        for i in qq.iloc[0] + '. ' + qq.iloc[1]:
            i += '.'
            advantange.append(i)
        
    advantange = ' '.join(advantange)
    #print (advantange)
    advantange = pd.DataFrame([advantange], index = {'Advantage'})
    ##print (advantange)
    finalsummary = pd.concat([novelty,use,advantange,keywords],axis = 0)
    finalsummary.to_csv(path + 'Final summary.csv')

    return True
#--------------------------------------------------------------------------------------
def unzip():
  
    os.chdir(path)
    with zipfile.ZipFile('summarization_input_data.zip', 'r') as zf:
        for file in zf.namelist():
            zf.extract(file)
    os.chdir(os_path)
      
    return True

def home(request):
    return render(request, 'Summarization/index.html')

def Submit(request):
    if request.method == 'POST':

        global current_time
        current_time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

        fileinfo = request.FILES.get('file')

        if fileinfo:
        
            filename = str(request.FILES['file'])

            if filename == 'summarization_input_data.zip':
                handle_uploaded_file(request.FILES['file'], str(request.FILES['file']))
                #return HttpResponse("Successful you upload the " + filename + " and done!")
                return render(request,'Summarization/dataresult.html')

            else:
                #messages.success(request, 'you upload the wrong file')
                return HttpResponse("you upload the wrong file")
        else:
            #messages.success(request, 'you did not upload the file')
            return HttpResponse("you did not upload the file")

 
    return HttpResponse("Failed")

def download(request):
    from django.http import FileResponse

    file=open('Summarization/static/Summarization/upload/example/summarization_input_data.zip','rb')

    response =FileResponse(file)

    response['Content-Type']='application/octet-stream'

    response['Content-Disposition']='attachment;filename="summarization_input_data.zip"'

    return response

def zip_file():
    
    os.chdir(path)
    with zipfile.ZipFile(current_time + '-output.zip', 'w') as zf:
        zf.write('Final summary.csv')
        zf.write('testing sets.csv')
        zf.write('training sets.csv')
    os.chdir(os_path)
    return True

def handle_uploaded_file(file, filename):
    if not os.path.exists('Summarization/static/Summarization/upload/'+ current_time + '/'):
        os.mkdir('Summarization/static/Summarization/upload/'+ current_time + '/')
        global path
        path = 'Summarization/static/Summarization/upload/'+ current_time + '/'
 
    with open( path + filename, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)


    #os.chdir('upload/'+ current_time + '/')
    
    unzip()

    main_jack()

    zip_file()

    return True

def result(request):
    #os.chdir('TF_IDF/static/upload/'+ current_time + '/')
    from django.http import FileResponse

    zip_filename = current_time + '-output.zip'

    file=open(path + zip_filename,'rb')

    response =FileResponse(file)

    response['Content-Type']='application/octet-stream'

    response['Content-Disposition']='attachment;filename="{}"'.format(zip_filename)



    return response