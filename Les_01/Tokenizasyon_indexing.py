# Recruitment with tensorflow and sklearn lesson_01
# This lesson is about how to build a recruitment system with tensorflow and sklearn
# We will use the keras library to build the model
# So create a virtual environment and install the libraries
# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer # Corrected the typo from 'tenserflow' to 'tensorflow'
from tensorflow.keras.preprocessing.sequence import pad_sequences #


def Tokenization_indexing(texts,start_index=0):
    """
    Bu kütüphane ile metinleri tokenize edebiliriz.
    return: tokenize edilmiş metinler
    """
    tokenized_tex=[sentece.lower().split() for sentece in texts]
    return tokenized_tex 

def indexingForTokenizer(tokenized_text,startindex =0):
    """
    Tokenize text for indexing
    """
    index=startindex
    word_index ={}
    for sentence in tokenized_text:
        for word in sentence:
            if word not in word_index:
                word_index[word]=index
                index +=1
    return word_index

def main():
    texts=list()
    print("**************\tTokenization and Indexing Programina Hosgeldiniz\t**************")
    print("Standart Metinlerin Islenmesini istiyorsaniz 1 yaziniz.")
    print("Kullanici Metinlerin Islenmesini istiyorsaniz 2 yaziniz.")
    choice=int(input("Lutfen seciminizi yapiniz: "))
    if choice==1:
        # Add some texts to the list
        texts.append("I love programming in Python")
        texts.append("I love programming in Java")
        texts.append("I love programming in C++")
        texts.append("I love programming in C#")
        texts.append("I love programming in C")
        texts.append("I love programming in C++")
        texts.append("I love programming in C#")
        tokenized_tex=Tokenization_indexing(texts)
        if tokenized_tex is not None:
            word_indeks=indexingForTokenizer(tokenized_tex)
            print("Tokenized Metin: \n",tokenized_tex)
            print("Word Index: \n",word_indeks)
        else:
            print("indexing and tokenization failed")
    elif choice==2:
        text=input("Lutfen bir metin giriniz: ")
        devam=int(input("Daha fazla metın girmek istiyorsanız 3 e basınız lütfen"))
        texts.append(text)
        while devam ==3:
            text=input("Lutfen bir metin giriniz: ")
            texts.append(text)
            devam=int(input("Daha fazla metın girmek istiyorsanız 3 e basınız lütfen,Devam etmek istemiyorsamız herhangi bir şeye basınız."))
        tokenized_tex=Tokenization_indexing(texts)
        if tokenized_tex is not None:
            word_indeks=indexingForTokenizer(tokenized_tex)
            print("Tokenized Metin: \n",tokenized_tex)
            print("Word Index: \n",word_indeks)
        else:
            print("indexing and tokenization failed")
        print("Tokenized Metin: \n",tokenized_tex)
        print("Word Index: \n",word_indeks)
    else:
        print("Gecersiz secim")

if __name__ == "__main__":
    main()

