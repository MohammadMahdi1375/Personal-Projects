## Table of contents
1. [LSTM Text Classification](#LSTM_Text_Classification)
   - [Wav2Vec](#wav2vec)
   - [Whisper](#whisper)
3. [Sequence To Sequence RNN](#Seq2Seq_RNN)
4. [Sequence To Sequence Attention_RNN](#Seq2Seq_Attn_RNN)
5. [Sequence To Sequence Transformer_RNN](#Seq2Seq_Trans_RNN)
6. [Sequence To Sequence BERT_RNN](#Seq2Seq_BERT_RNN)
7. [Response Generation with GPT2](#NLG_GPT2)
   - [Introduction](#intro)
   - [Whisper](#whisper)
9. [Results](#results)


## <a name='introduction'></a> Introduction
This Project is about end-to-end speech recognizer, also known as automatic speech recognition(ASR) to process human speech into a written format. We feed a speech signal spoken in the Serbian language into the model to generate the transcription for that. This project has been done by means of the Speechbrain library which is an open-source all-in-one speech toolkit based on PyTorch. It is designed to make the research and development of speech technology easier. This [tutorial](https://speechbrain.readthedocs.io/en/latest/index.html) will provide you with all the very basic elements needed to start using SpeechBrain for your projects.

## <a name='dataset'></a> Dataset
The database which is used here is the Common Voice database which offers mp3 audio files at 42Hz. 



## <a name='NLG_GPT2'></a> Response Generation with GPT2
### <a name='intro'></a> Introduction
In this Project we will go through the process of **fine-tuning the pretrained GPT2 model** that is available in the HuggingFace Transformers library for response generation. It is a **dialogue system** aiming to generate a response given a context. This is also known as **Natural Language Generation (NLG)**. Here we will consider the smallest pre-trained GPT2 model (GPT-2). In this project I am looking for:
- Instantiate a pretrained GPT2.
- Fine-tuning GPT2 on MultiWOZ with SpeechBrain for response generation task.



<img src="./Imgs/wav2vec_asr.png" alt="drawing" width="800" height="250"/>
