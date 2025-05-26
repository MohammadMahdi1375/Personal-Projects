## Table of contents
1. [LSTM Text Classification](#LSTM_Text_Classification)
2. [Sequence To Sequence RNN](#Seq2Seq_RNN)
3. [Sequence To Sequence Attention-based RNN](#Seq2Seq_Attn_RNN)
4. [Sequence To Sequence Transformer_RNN](#Seq2Seq_Trans_RNN)
5. [Sequence To Sequence BERT_RNN](#Seq2Seq_BERT_RNN)
6. [Response Generation with GPT2](#NLG_GPT2)
   - [Introduction](#intro)
   - [Dataset](#dataset)



## <a name='LSTM_Text_Classification'></a> LSTM Text Classification
This project aims to classify SMS messages into two classes Spam and ham (legitimate). The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) which contains one set of SMS messages in English of 5,574 messages. The proposed model is a single-layer bidirectional LSTM model which is followed by a sigmoid activation function (For binary Classification).

## <a name='Seq2Seq_RNN'></a> Sequence To Sequence RNN
The most common sequence-to-sequence (seq2seq) models are encoder-decoder models, which commonly use an LSTM to encode the source (input) sentence into a single vector called context vector (z). We can think of the context vector as being an abstract representation of the entire input sentence}. This vector is then decoded by a second RNN which learns to output the target (output) sentence by generating it one word at a time. The aim of this project is to translate from Dutch to English.
<img src="./Imgs/RNN.png" alt="drawing" width="800" height="250"/>

## <a name='Seq2Seq_Trans_RNN'></a> Sequence To Sequence Attention-based RNN
In this task, we have tried to improve the performance of the Seq2Seq RNN model employing the Attention Model which scores how well the inputs around position j and the output at position i match.

## <a name='Seq2Seq_Trans_RNN'></a> Sequence To Sequence Transformer-based model
In this task, a Transformer-based model has been used in order to translate from Dutch to English.

## <a name='Seq2Seq_BERT_RNN'></a> Sequence To Sequence using BERT
We will utilize a large metadata-rich collection of fictional conversations extracted from raw movie scripts (220,579 conversational exchanges between 10,292 pairs of movie characters in 617 movies). There are two text files in this dataset, movie_conversations.txt and movie_lines.txt. A consecutive conversation between two users in a movie is given in movie_conversations.txt.

## <a name='NLG_GPT2'></a> Response Generation with GPT2
### <a name='intro'></a> Introduction
In this project, we will go through the process of **fine-tuning the pre-trained GPT2 model** that is available in the HuggingFace Transformers library for response generation. It is a **dialogue system** aiming to generate a response given a context. This is also known as **Natural Language Generation (NLG)**. Here we will consider the smallest pre-trained GPT2 model (GPT-2). In this project I am looking for:
- Instantiate a pre-trained GPT2.
- Fine-tuning GPT2 on MultiWOZ with SpeechBrain for response generation task.

### <a name='dataset'></a> Dataset
Now we will discuss how to fine-tune the GPT-2  model for response generation. To achieve this, we will be using a smaller version of the MultiWOZ 2.1 dataset. Multi-Domain Wizard-of-Oz dataset (MultiWOZ), a fully-labeled collection of human-human written conversations spanning over multiple domains and topics. Instead of using all the data, we will set a parameter that identifies the percentage of data to be sampled. For this task, we will just sample 1000, 100, 200 of training, valid, and test.

<img src="./Imgs/wav2vec_asr.png" alt="drawing" width="800" height="250"/>
