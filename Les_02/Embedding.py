"""
Coding by:
    Name: Eyyüp Toprak
    Date: 06.03.2025
    Version: 1.0
    Description: This code is about how to build a recruitment system with tensorflow and sklearn 
    -This course covers the topics covered in Lesson 1 and embedding processes.
    Usage: python Tokenizasyon_indexing.py UTF-8
    For more information: https://github.com/eyyuptoprak
    For more information: eyup.tp@hotmail.com
"""


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer # Corrected the typo from 'tenserflow' to 'tensorflow'
from tensorflow.keras.preprocessing.sequence import pad_sequences #
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
import tensorflow as tf


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


def Embedding_Vector(word_indeks, padded_sequance, embedding_dim=10):
    """
    Embedding Vector fonksiyonu, modelin gömme vektörlerini oluşturur.
    """
    try:
        # Veri şeklini kontrol et ve düzelt
        if len(padded_sequance.shape) == 1:
            padded_sequance = padded_sequance.reshape(-1, 1)
        
        # Model oluşturma
        model = Sequential([
            Embedding(input_dim=len(word_indeks)+1, 
                     output_dim=embedding_dim, 
                     input_length=1)
        ])
        
        model.compile('adam', 'mse')  # Model derleme
        
        # Tahmin yapma
        embedding = model.predict(padded_sequance)
        return embedding.squeeze()
    
    except Exception as e:
        print(f"Embedding işlemi sırasında hata oluştu: {str(e)}")
        return None


def visulaize_embedding(embedding_vector,texts,pca_components=2,normalize=0.1):
    """
    Embedding Vector fonksiyonu, modelin gömme vektörlerini oluşturur.
    """
    pca=PCA(n_components=2)
    reducet_embedding=pca.fit_transform(embedding_vector)
    plt.figure(figsize=(15,15))
    for i,word in enumerate(texts):
        plt.scatter(reducet_embedding[i,0],reducet_embedding[i,1],c="blue",label=word)
        plt.text(reducet_embedding[i,0]+normalize,reducet_embedding[i,1]+normalize,word,fontsize=12)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Embedding Vector")
    plt.legend()
    plt.show()

def Embedding_main(tokenized_tex, texts, embeddingDim, Normalize, pcaComponents):
    """
    Embedding fonksiyonu, tokenize edilmiş metinleri gömme vektörlerine çevirir. ve gömme vektörlerini PCA ile 2 boyutlu sekilde görselleştirir.
    """
    embedding_dim=embeddingDim
    normalize=Normalize
    pca_components=pcaComponents
    word_indeks=indexingForTokenizer(tokenized_tex)
    sequance=textsToSequance(tokenized_tex,word_indeks)
    max_length=max(len(seq) for seq in sequance)
    padded_sequance=SequancePadding(sequance,max_length)
    print("**************\tTokenization Texts\t**************")
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
    print("Padded Sequence Shape:", padded_sequance.shape)
    print("Word Index Length:", len(word_indeks))
    
    Embedding_vector = Embedding_Vector(word_indeks, padded_sequance, embedding_dim)
    
    if Embedding_vector is None:
        print("Embedding vektörü oluşturulamadı")
        return
    
    print("Embedding Vector Shape:", Embedding_vector.shape)
    print("**************\tEmbedding Vector\t**************")
    print("Embedding Vector:PCA kullanilarak 2 boyutlu sekilde gosteriliyor")
    visulaize_embedding(Embedding_vector,texts,pca_components,normalize)
    print("******************************************************************************************")


def main():
    print("**************\tTokenization indexing and Embedding Programina Hosgeldiniz\t**************")
    print("Standart Metinlerin Islenmesini istiyorsaniz 1 yaziniz.")
    print("Kullanici Metinlerin Islenmesini istiyorsaniz 2 yaziniz.")
    choice=int(input("Lutfen seciminizi yapiniz: "))
    if choice==1:
        texts=list()
        texts.append("cat")
        texts.append("kitten")
        texts.append("dog")
        texts.append("puppy")
        texts.append("mouse")
        texts.append("elephant")
        texts.append("bird")
        texts.append("airplane")
        texts.append("car")
        texts.append("house")
        texts.append("tree")
        texts.append("flower")
        texts.append("fruit")
        tokenized_tex=Tokenization_indexing(texts)
        if tokenized_tex is not None:
           try:
               Embedding_main(tokenized_tex,texts,embeddingDim=10,Normalize=0.01,pcaComponents=2)
           except Exception as e:
                print("**************\tEmbedding Vector Failed\t**************")
                print("Embedding vector failed")
                print("******************************************************************************************")
        else:
            print("**************\tIndexing and Tokenization Failed\t**************")
            print("indexing and tokenization failed")
            print("******************************************************************************************")
    elif choice==2:
        texts=list()
        text=input("Lutfen bir metin giriniz: ")
        texts.append(text)
        devam=int(input("Daha fazla metın girmek istiyorsanız 3 e basınız lütfen,Devam etmek istemiyorsamız herhangi bir şeye basınız."))
        while devam ==3:
            text=input("Lutfen bir metin giriniz: ")
            texts.append(text)
            devam=int(input("Daha fazla metın girmek istiyorsanız 3 e basınız lütfen,Devam etmek istemiyorsamız herhangi bir şeye basınız."))
        tokenized_tex=Tokenization_indexing(texts)
        if tokenized_tex is not None:
           try:
               embeddingDim=int(input("Lutfen gömme vektörünün boyutunu giriniz: "))
               Normalize=float(input("Lutfen normalizasyon degerini giriniz: "))
               if embeddingDim is not None and Normalize is not None:
                   Embedding_main(tokenized_tex,texts,embeddingDim,Normalize,pcaComponents=2)
               else:
                   print("**************\tEmbedding Vector Failed\t**************")
                   print("Embedding vector failed")
                   print("******************************************************************************************")
           except Exception as e:
                print("**************\tEmbedding Vector Failed\t**************")
                print("Embedding vector failed")
                print("******************************************************************************************")

if __name__ == "__main__":
    main()
