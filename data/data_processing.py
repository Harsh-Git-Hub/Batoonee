"""
Author: Harsh Gupta
Created on: 2nd Nov 2020 4:30 PM
"""

import keras
import tensorflow_datasets as tfds
from utils.utils import pprint, clean_text


class Data:
    def __init__(self, dataset_name="coqa", split="train", shuffle_files=False):
        self.dataset_name = dataset_name
        self.split = split
        self.shuffle_files = shuffle_files
        self.ds = tfds.load(self.dataset_name, split=self.split, shuffle_files=self.shuffle_files)
        self.df = None
        self.ques = None
        self.ans = None

    def as_dataframe(self):
        self.df = tfds.as_dataframe(self.ds)
        del self.ds
        return self.df

    def make_qa_pair(self, slicepercent=0.02):
        # context = []
        self.ques = []
        self.ans = []
        for i in range(int(self.df.shape[0] * slicepercent)):
            # context.extend([df.iloc[i]["story"].decode("utf-8")] * len(df.iloc[i]["questions"]))
            self.ques.extend(
                [clean_text("SOS " + x.decode("utf-8") + " EOS") for x in self.df.iloc[i]["questions"].tolist()]
            )

            self.ans.extend(
                [clean_text("SOS " + x.decode("utf-8") + " EOS") for x in
                 self.df.iloc[i]["answers/input_text"].tolist()]
            )

        return self.ques, self.ans

    def texts_to_sequences(self, text, print=1):
        def print_summary(tk):
            # summarizing what was learned
            pprint(title="Tokenizer",
                   word_count=tk.word_counts,
                   doc_count=tk.document_count,
                   word_index=tk.word_index,
                   word_docs=tk.word_docs)

        tk = keras.preprocessing.text.Tokenizer()
        tk.fit_on_texts(text)
        if print == 1:
            print_summary(tk)
        return tk.texts_to_sequences(text), tk
