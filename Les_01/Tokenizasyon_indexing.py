"""
Coding by:
    Name: Eyyüp Toprak
    Date: 06.03.2025
    Version: 1.0
    Description: This code is about how to build a recruitment system with tensorflow and sklearn
    Usage: python Tokenizasyon_indexing.py UTF-8
    For more information: https://github.com/eyyuptoprak
    For more information: eyup.tp@hotmail.com
"""

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


def Tokenization_indexing(texts):
    """Metinleri tokenize eden fonksiyon.

    Bu fonksiyon, verilen metinleri tokenize ederek daha küçük parçalara ayırır.
    Doğal Dil İşleme'de (NLP) temel bir işlem adımıdır.

    Args:
        texts (list): İşlenecek metin listesi.

    Returns:
        list: Tokenize edilmiş metin listesi. Her bir metin küçük harflere
             dönüştürülmüş ve kelimelere ayrılmış şekildedir.

    Example:
        >>> texts = ["Merhaba Dünya", "Python Programlama"]
        >>> Tokenization_indexing(texts)
        [['merhaba', 'dünya'], ['python', 'programlama']]
    """
    tokenized_tex=[sentece.lower().split() for sentece in texts]
    return tokenized_tex

def indexingForTokenizer(tokenized_text,startindex =0):
    """
    Tokenize text for indexing
    Bu fonksiyon, verilen tokenize edilmiş metinleri indekslemek için kullanılır.
    Her kelimeye bir indeks atarak, metinleri daha iyi bir şekilde saklamayı sağlar.

    Args:
        tokenized_text (list): Tokenize edilmiş metin listesi.
        startindex (int, optional): Başlangıç indeksi. Varsayılan değer 0.
    example:
        >>> tokenized_text = [['merhaba', 'dünya'], ['python', 'programlama']]
        >>> indexingForTokenizer(tokenized_text)
        {'merhaba': 0, 'dünya': 1, 'python': 2, 'programlama': 3}
    """
    index=startindex
    word_index ={}
    for sentence in tokenized_text:
        for word in sentence:
            if word not in word_index:
                word_index[word]=index
                index +=1
    return word_index

def textsToSequance(tokenized_text,word_index):
    """
    Tokenize edilmiş metinleri sayısal sıralara çeviren fonksiyon.
    Bu fonksiyon, tokenize edilmiş metinleri sayısal sıralara çevirerek, derin öğrenme modelleri için uygun hale getirir.

    Args:
        tokenized_text (list): Tokenize edilmiş metin listesi.
        word_index (dict): Kelime indeksi sözlüğü.
    
    """
    sequance=[[word_index[word] for word in sentence] for sentence in tokenized_text]

    return sequance

def SequancePadding(sequance,max_length):
    """
    Sequance Padding fonksiyonu, tokenize edilmiş metinleri belirtilen maksimum uzunluğa kadar doldurur.
    Bu, derin öğrenme modelleri için gerekli olan sabit uzunluklu girdileri sağlamak için kullanılır.
    """
    paddet_sequance=pad_sequences(sequance,maxlen=max_length,padding="post",truncating='post',value=0)
    return paddet_sequance

def TF_IDFHesaplama(tokenized_text,word_index):
    """
    TF-IDF hesaplama fonksiyonu, tokenize edilmiş metinleri TF-IDF skorlarına çevirir.
    Bu, metinlerin sınıflandırılmasında veya benzerlik aramasında kullanılır.
    Args:
        tokenized_text (list): Tokenize edilmiş metin listesi.
        word_index (dict): Kelime indeksi sözlüğü.
    Returns:
        numpy.ndarray: TF-IDF matrisi.
    Example:
        >>> tokenized_text = [['merhaba', 'dünya'], ['python', 'programlama']]
        >>> word_index = {'merhaba': 0, 'dünya': 1, 'python': 2, 'programlama': 3}  
    """
    tfidf_vectorizer=TfidfVectorizer(vocabulary=list(word_index.keys()))
    tfidf_matrix=tfidf_vectorizer.fit_transform([" ".join(sentence) for sentence in tokenized_text])
    return tfidf_matrix.toarray()


def main():
    """Metin tokenizasyonu ve indeksleme işlemlerini yöneten ana fonksiyon.

    Bu fonksiyon, kullanıcıya iki seçenek sunar:
    1. Önceden tanımlanmış standart metinlerin işlenmesi
    2. Kullanıcının girdiği metinlerin işlenmesi

    Her iki seçenekte de metinler tokenize edilir, indekslenir ve TF-IDF matrisi oluşturulur.
    Kullanıcı metinleri seçeneğinde, kullanıcı istediği kadar metin girebilir.

    Returns:
        None        
    """
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

            sequance=textsToSequance(tokenized_tex,word_indeks)

            max_length=max(len(seq) for seq in sequance)

            padded_sequance=SequancePadding(sequance,max_length)

            tfidf_matrix=TF_IDFHesaplama(tokenized_tex,word_indeks)
            print("**************\tTokenization Texts\t**************")
            print("Tokenized Metin: \n",tokenized_tex)
            print("******************************************************************************************")
            print("**************\tWord Index Texts\t**************")
            print("Word Index: \n",word_indeks)
            print("******************************************************************************************")
            print("**************\tSequance Texts\t**************")
            print("Sequance: \n",sequance)
            print("******************************************************************************************")
            print("**************\tPadded Sequance Texts\t**************")
            print("Padded Sequance: \n",padded_sequance)
            print("******************************************************************************************")
            print("**************\tTF-IDF Matrix Texts\t**************")
            print("TF-IDF Matrix: \n",tfidf_matrix)
            print("******************************************************************************************")
        else:
            print("**************\tIndexing and Tokenization Failed\t**************")
            print("indexing and tokenization failed")
            print("******************************************************************************************")

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

            sequance=textsToSequance(tokenized_tex,word_indeks)

            max_length=max(len(seq) for seq in sequance)

            padded_sequance=SequancePadding(sequance,max_length)

            tfidf_matrix=TF_IDFHesaplama(tokenized_tex,word_indeks)

            print("**************\tTokenized Texts\t**************")
            print("Tokenized Metin: \n",tokenized_tex)
            print("******************************************************************************************")

            print("**************\tWord Index Texts\t**************")
            print("Word Index: \n",word_indeks)
            print("******************************************************************************************")

            print("**************\tSequance Texts\t**************")
            print("Sequance: \n",sequance)
            print("******************************************************************************************")

            print("**************\tPadded Sequance Texts\t**************")
            print("Padded Sequance: \n",padded_sequance)
            print("******************************************************************************************")

            print("**************\tTF-IDF Matrix Texts\t**************")
            print("TF-IDF Matrix: \n",tfidf_matrix)
            print("******************************************************************************************")
        else:
            print("**********************************************************************\tIndexing and Tokenization Failed\t**********************************************************************")
            print("indexing and tokenization failed")

        print("Tokenized Metin: \n",tokenized_tex)
        print("Word Index: \n",word_indeks)
        print("Sequance: \n",sequance)
        print("Padded Sequance: \n",padded_sequance)
        print("TF-IDF Matrix: \n",tfidf_matrix)
    
    else:
        print("Gecersiz secim")

if __name__ == "__main__":
    main()

