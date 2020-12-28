# Patent-Summarization
The research purposes the NLP=Seq2Seq with attention methdology based patent summarization system to automatically compile the multi-lingual patents which can e-discover the highlights of IP technology and knowledge to provide enterprises focused R&D technology or best defense claims in the timely manner.
# Website Server
http://140.114.53.218/python/programs/Summary/index.php

# Generic Methodology in Four Steps 
1. The raw data pre-processing
A large related domain of raw patent sets are input into the model, which includes Chinese or English language patents. Then, find a set of patents in a given topic (sub-domain) based on some intelligent algorithms such as LDA, K-means, and Hierarchical.

2. Text pre-processing
When pre-processing a set of patents (testing set), the text is converted to lowercase, and stop words and punctuation are removed to retain a set of meaningful words

3. Summarization model training
Sequence-to-sequence models include an encoder step and a decoder step. In the encoder step, a model converts an input sequence into a fixed representation. In the decoder step, a language model is trained on both the output sequence as well as the fixed representation from the encoder.
After the model training, the model can predict the label type of training set 100% accuracy from input type. But these label type are not research summary result.

4. Evaluation
The summary result that precision ratio is 85.7% and recall ratio is 75%.

