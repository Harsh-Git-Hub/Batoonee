"""
Author: Harsh Gupta
Created on: 2nd Nov 2020 5:03 PM
"""

import datetime
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from data.data_processing import Data
from model.encoderdecoder import Seq2Seq
from utils.utils import pprint


class Main:
    def __init__(self):
        self.s2s = None
        self.ans = None
        self.ques = None
        self.qtk = None
        self.atk = None
        self.encoder_input_data = None
        self.decoder_input_data = None
        self.decoder_target_data = None
        self.encoder_model = None
        self.decoder_model = None
        self.pred = None
        self.cwd = os.getcwd()
        self.timestamp = "{}_{}".format(datetime.datetime.now().date(),
                                        str(datetime.datetime.now().time()).replace(":", ""))

    def __load_data(self):
        data = Data(dataset_name="coqa", split="train", shuffle_files=True)
        df = data.as_dataframe()

        pprint(title="COQA Data",
               columns=df.columns,
               head=df[["story", "questions", "answers/input_text"]].head(),
               tail=df[["story", "questions", "answers/input_text"]].tail(),
               shape=df.shape)

        self.ques, self.ans = data.make_qa_pair(slicepercent=0.02)
        pprint(qlen=len(self.ques), alen=len(self.ans), ques=self.ques[0], ans=self.ans[0])

        ques_vec, self.qtk = data.texts_to_sequences(self.ques, print=0)
        ans_vec, self.atk = data.texts_to_sequences(self.ans)

        for _ in range(5):
            j = np.random.randint(0, len(ques_vec))
            print(
                " ".join(self.qtk.index_word[x] for x in ques_vec[j]) + " : " + " ".join(
                    self.atk.index_word[x] for x in ans_vec[j]),
                end="\n\n")

        self.s2s = Seq2Seq(ques_vec, ans_vec, self.qtk, self.atk, save_model=False, save_weights=False)
        self.__load_model()

    def __load_model(self):
        self.encoder_input_data, self.decoder_input_data, self.decoder_target_data = self.s2s.get_encodings()
        self.s2s.fill_encodings()
        pprint(title="Comparison",
               enc_inp_shape=self.encoder_input_data.shape,
               dec_inp_shape=self.decoder_input_data.shape,
               dec_tgt_shape=self.decoder_target_data.shape,
               enc_inp_entry=self.encoder_input_data[0],
               dec_inp_entry=self.decoder_input_data[0],
               dec_tgt_entry=self.decoder_target_data[0])

        model, exists = self.s2s.build_model(num_encoder_tokens=len(self.qtk.word_index) + 1,
                                             num_decoder_tokens=len(self.atk.word_index) + 1,
                                             latent_dim=100)
        model.summary()
        keras.utils.plot_model(model, to_file="{}\\plots\\s2s\\s2s_{}.png".format(self.cwd, self.timestamp),
                               show_layer_names=True, show_shapes=True)
        plt.imshow(plt.imread("{}\\plots\\s2s\\s2s_{}.png".format(self.cwd, self.timestamp)))
        plt.show()

        if exists:
            print("Found existing model. Skipping Training. Inferring using saved model")
            self.encoder_model, self.decoder_model = self.s2s.infer(latent_dim=100)
            self.__predict()
        else:
            self.__train_model()

    def __train_model(self):
        adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.987, beta_2=0.999, name="SlowAdam")
        es = keras.callbacks.EarlyStopping(monitor='categorical_crossentropy', min_delta=1e-5, patience=10, verbose=1)
        # logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        self.s2s.train(inputs=[self.encoder_input_data, self.decoder_input_data],
                       output=self.decoder_target_data,
                       optimizer='rmsprop',
                       batch_size=10,
                       epochs=400,
                       val_split=0.20,
                       callbacks=[es],
                       verbose=1,
                       shuffle=True)

        # Commented out IPython magic to ensure Python compatibility.
        # %tensorboard --logdir logs

        self.encoder_model, self.decoder_model = self.s2s.infer(latent_dim=100)
        self.decoder_model.summary()
        keras.utils.plot_model(self.decoder_model,
                               to_file="{}\\plots\\\s2s\\decoder_{}.png".format(self.cwd, self.timestamp),
                               show_layer_names=True, show_shapes=True, expand_nested=True)

        self.encoder_model.summary()
        keras.utils.plot_model(self.encoder_model,
                               to_file="{}\\plots\\s2s\\encoder_{}.png".format(self.cwd, self.timestamp),
                               show_layer_names=True, show_shapes=True, expand_nested=True)

        self.__predict()

    def __predict(self, ):
        self.pred = []
        for seq_idx in range(int(self.encoder_input_data.shape[0])):
            # seq_idx = np.random.randint(0, encoder_input_data.shape[0])
            input_seq = self.encoder_input_data[seq_idx: seq_idx + 1]
            decoded_sequence = self.s2s.decode_sequence(input_seq=input_seq, decoder_input_data=self.decoder_input_data)
            self.pred.append(decoded_sequence)
            per = (seq_idx / self.encoder_input_data.shape[0]) * 100
            if int(per) >= 25 and int(per) % 25 == 0:
                print("{}% completed".format(per))
            # print("Question: {}".format(ques[seq_idx:seq_idx + 1]), end="\n")
            # print("Answer: {}".format(decoded_sequence), end="\n====================\n")
        self.__evaluate()

    def __evaluate(self):
        hypothesis = []
        reference = []
        for i in range(len(self.ans)):
            hypothesis.append([x for x in self.pred[i].split(" ") if x not in ["sos", "eos", " ", ""]])
            reference.append([x for x in self.ans[i].split(" ") if x not in ["sos", "eos", " ", ""]])

        pprint(hyp_len=len(hypothesis), ref_len=len(reference), hyp_data=hypothesis[:5], ref_data=reference[:5])

        bleu_sum = 0.0
        for i in range(len(hypothesis)):
            smoothie = SmoothingFunction().method1
            weights = [1 / len(reference[i])] * len(reference[i])
            # last = bleu_sum
            # noinspection PyTypeChecker
            bleu_sum += sentence_bleu([reference[i]], hypothesis[i], weights=weights, smoothing_function=smoothie)
            # if bleu_sum - last > 0.0:
            #     pprint(title="At %s"%i, weights=weights, bleu_score=bleu_sum - last)
        bleu_score = bleu_sum / len(hypothesis)
        print("BLEU Score: %s" % bleu_score)

    def execute(self):
        self.__load_data()


executor = Main()
executor.execute()
