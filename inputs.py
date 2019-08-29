import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer, LancasterStemmer, SnowballStemmer


####################################################################################
######## Lemmatization
####################################################################################
lem = SnowballStemmer('english')
nonce = 'kelilili'


class inputs():

    def __init__(self, lexeme_dictionary, lemmatization=True):
        self.dic=lexeme_dictionary
        self.lemmas = lemmatization
        self.data = None
        super(inputs, self).__init__()

    def bulk_vertical(self, df, columns=['lex']):
        batches=[]
        for sent in df['sent'].unique():
            sent_data=[]
            for col in columns:
                lexemes = []
                if self.lemmas:
                    lexemes=[self.dic[lem.stem(str(w))] for w in df[col].loc[df['sent'].isin([sent])].values]
                else:
                    lexemes = [self.dic[str(w)] for w in df[col].loc[df['sent'].isin([sent])].values]
                sent_data.append(np.array(lexemes).reshape(1, -1))

                batches.append(sent_data)

        return batches

    def individual_vertical(self, loc, columns=['lex']):
        batch=[]
        for col in columns:
            col_data=[]
            if self.lemmas:
                col_data=[self.dic[lem.stem(str(w))] for w in loc[col].values]
            else:
                col_data = [self.dic[w] for w in loc[col].values]

            batch.append(np.array(col_data).reshape(1, -1))

        return batch

    def bulk_horizontal(self, df, columns=['head', 'tref.dep', 'compound', 'tref'], initial_step=nonce):
        batches=[]
        for sent in df['sent'].unique():
            batch = [df[col].loc[df['sent'].isin([sent])].unique()[0] for col in columns[:-1] if
                     df[col].loc[df['sent'].isin([sent])].unique()[0] not in [None, np.nan]]
            batch.insert(0, initial_step)
            if self.lemmas:
                batch = [lem.stem(str(w)) for w in batch]
            batch = [self.dic[str(w)] for w in batch]
            batches.append(np.array(batch).reshape(1, -1))
        return batches

    def individual_horizontal(self, loc, columns=['head', 'tref.dep', 'compound', 'tref'], initial_step=nonce):
        batch = [loc[col].unique()[0] for col in columns[:-1] if
                 loc[col].unique()[0] not in [None, np.nan]]
        batch.insert(0, initial_step)
        if self.lemmas:
            batch = [lem.stem(str(w)) for w in batch]
        batch = [self.dic[str(w)] for w in batch]
        return np.array(batch).reshape(1, -1)

    def bulk_decoderY(self, df, columns=['head', 'tref.dep', 'compound', 'tref']):
        batches = []

        one_hot=[0.0 for _ in range(len(self.dic))]
        for sent in df['sent'].unique():
            batch = [df[col].loc[df['sent'].isin([sent])].unique()[0] for col in columns if
                     df[col].loc[df['sent'].isin([sent])].unique()[0] not in [None, np.nan]]

            if self.lemmas:
                batch = [lem.stem(str(w)) for w in batch]

            batch = [self.dic[str(w)] for w in batch]

            one_hot_batch=[]
            for i in batch:
                hot1=list(one_hot)
                hot1[i] = 1.0
                one_hot_batch.append(hot1)

            batches.append(np.array(one_hot_batch).reshape(1, -1, len(one_hot)))
        return batches

    def individual_decoderY(self, loc, columns=['head', 'tref.dep', 'compound', 'tref'], noncical=True):
        one_hot = [0.0 for _ in range(len(self.dic))]
        batch = [loc[col].unique()[0] for col in columns if
                 loc[col].unique()[0] not in [None, np.nan]]
        if self.lemmas:
            batch = [lem.stem(str(w)) for w in batch]
        batch = [self.dic[str(w)] for w in batch]

        one_hot_batch=[]
        for i in batch:
            hot1 = list(one_hot)
            hot1[i] = 1.0
            one_hot_batch.append(hot1)

        return np.array(one_hot_batch).reshape(1, -1, len(one_hot))
