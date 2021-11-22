# Nomes e RAs:
# Gabriel Penajo Machado,    769712
# Matheus Ramos de Carvalho, 769703
# Matheus Teixeira Matioli,  769783

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# dataset src:
# https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29

def kmeans():
    pass

if __name__ == '__main__':
    df = pd.read_csv('USCensus1990.data.txt')
    # Removendo o atributo caseid, pois não será relevante para o agrupamento.
    df = df.drop(columns=['caseid'])
    print(df)