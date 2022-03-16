from googletrans import Translator
import http.client
import hashlib
import json
import urllib
import random
from sklearn.datasets import fetch_20newsgroups
import re
import time

if __name__ == '__main__':
    IMDB_Text=[]
    with open("/home/UDA_pytorch/data/data_process/train.csv") as f:
        for line in f:
            text, attitude, file = line.split("\t",2)
            IMDB_Text.append(text)
    translator = Translator(service_urls=['translate.google.cn'])
    i=1
    with open("IMDB_google.tsv", 'w') as f:
        for train_data in IMDB_Text:
            if i%25==0:
                print("This is ",i,"data")
            train_data=train_data.strip()
            if len(train_data) == 0 or train_data.strip()=='' or train_data.isspace() == True :
                i+=1
                continue
            else:
                train_data = re.sub('\r|\n|\t|^[^\d]\w+|          |         |        ','',train_data)
                if len(train_data) == 0 or train_data.strip()=='' or train_data.isspace() == True :
                    i+=1
                    continue
                else:
                    train_data=train_data[:512]
                    text = translator.translate(train_data, src='en', dest='fr').text
                    backtranslation1 = translator.translate(text, src='fr', dest='en').text
                    text1 = translator.translate(train_data, src='en', dest='es').text
                    backtranslation2 = translator.translate(text1, src='es', dest='en').text
                    f.write(backtranslation1 + '\t' + backtranslation2)
                    f.write("\n")
            i+=1

    f.close()
