# Doğal Dil İşlemede Temel Adımlar: Tokenizasyon ve İndeksleme

## Giriş

Doğal Dil İşleme (NLP), bilgisayarların insan dilini anlama, yorumlama ve üretme yeteneğini geliştirmeyi amaçlayan yapay zeka alanıdır. Bu makalede, NLP'nin en temel adımlarından olan tokenizasyon ve indeksleme işlemlerinin teorik altyapısını ve pratik uygulamalarını inceleyeceğiz.

## 1. Tokenizasyon: Metni Anlamlı Parçalara Ayırma

### Teorik Altyapı

Tokenizasyon, ham metni daha küçük ve anlamlı parçalara (token) ayırma işlemidir. Bu işlem, NLP'nin temel taşlarından biridir çünkü bilgisayarların metinleri işleyebilmesi için öncelikle bu metinleri anlamlı birimlere ayırması gerekir.

Tokenizasyon çeşitli seviyelerde gerçekleştirilebilir:
- **Kelime Tokenizasyonu**: Metni kelimelere ayırma
- **Karakter Tokenizasyonu**: Metni karakterlere ayırma
- **Alt-kelime (Subword) Tokenizasyonu**: Metni anlamlı alt-kelime birimlerine ayırma (örn. BPE, WordPiece)

İncelediğimiz kodda, basit bir kelime tokenizasyonu uygulanmıştır:

```python
def Tokenization_indexing(texts):
    tokenized_tex=[sentece.lower().split() for sentece in texts]
    return tokenized_tex
```

Bu fonksiyon, metinleri küçük harflere dönüştürüp boşluklara göre kelimelere ayırır. Gerçek dünya uygulamalarında, daha karmaşık tokenizasyon algoritmaları kullanılır. Örneğin:

- Noktalama işaretlerinin ele alınması
- Özel karakterlerin işlenmesi
- Dile özgü kuralların uygulanması (Türkçe için ek ayırma gibi)
- Kısaltmaların ve özel durumların tanınması

### Tokenizasyonun Önemi

Tokenizasyon, NLP'nin sonraki tüm adımları için kritik öneme sahiptir. İyi bir tokenizasyon:
- Daha doğru metin temsilleri oluşturur
- Dil modellerinin performansını artırır
- Dile özgü nüansları yakalamaya yardımcı olur

## 2. Kelime İndeksleme: Metinden Sayılara Geçiş

### Teorik Altyapı

Makine öğrenimi modelleri sayılarla çalıştığından, tokenize edilmiş metinleri sayısal temsillere dönüştürmemiz gerekir. Kelime indeksleme, her benzersiz kelimeye bir sayısal indeks atayarak bu dönüşümü sağlar.

```python
def indexingForTokenizer(tokenized_text, startindex=0):
    index=startindex
    word_index ={}
    for sentence in tokenized_text:
        for word in sentence:
            if word not in word_index:
                word_index[word]=index
                index +=1
    return word_index
```

Bu fonksiyon, her yeni kelimeye sıralı bir indeks atayarak bir kelime-indeks sözlüğü oluşturur. Bu sözlük, metin verilerini sayısal verilere dönüştürmek için bir köprü görevi görür.

### Kelime İndekslemenin Önemi

Kelime indeksleme:
- Metinleri makine öğrenimi modellerinin anlayabileceği formata dönüştürür
- Kelime dağarcığını (vocabulary) tanımlar
- Veri yapılarını optimize eder ve bellek kullanımını iyileştirir

## 3. Metin Dizileştirme: Sayısal Dizilere Dönüşüm

### Teorik Altyapı

Kelime indeksleri oluşturulduktan sonra, tokenize edilmiş metinleri bu indeksleri kullanarak sayısal dizilere dönüştürebiliriz:

```python
def textsToSequance(tokenized_text, word_index):
    sequance=[[word_index[word] for word in sentence] for sentence in tokenized_text]
    return sequance
```

Bu işlem, her kelimeyi karşılık gelen sayısal indeksiyle değiştirerek metinleri tamamen sayısal hale getirir. Bu sayısal diziler, makine öğrenimi modellerinin doğrudan işleyebileceği formattadır.

### Dizileştirmenin Önemi

Metin dizileştirme:
- Metinleri vektör uzayında temsil etmeyi sağlar
- Matematiksel işlemlerin uygulanabilmesini mümkün kılar
- Derin öğrenme modellerinin giriş formatına uygun hale getirir

## 4. Dizi Doldurma (Padding): Sabit Uzunluk Sağlama

### Teorik Altyapı

Derin öğrenme modelleri genellikle sabit boyutlu girdiler bekler, ancak metinler farklı uzunluklarda olabilir. Dizi doldurma, tüm dizileri aynı uzunluğa getirerek bu sorunu çözer:

```python
def SequancePadding(sequance, max_length):
    paddet_sequance=pad_sequences(sequance, maxlen=max_length, padding="post", truncating='post', value=0)
    return paddet_sequance
```

Bu fonksiyon, kısa dizileri sıfırlarla doldurarak (padding) veya uzun dizileri keserek (truncating) tüm dizileri aynı uzunluğa getirir.

### Dizi Doldurmanın Önemi

Dizi doldurma:
- Batch işleme imkanı sağlar
- Model eğitimini ve çıkarımını verimli hale getirir
- Farklı uzunluktaki metinlerin aynı modelde işlenmesini mümkün kılar

## 5. TF-IDF Vektörizasyonu: Kelime Önemini Belirleme

### Teorik Altyapı

TF-IDF (Term Frequency-Inverse Document Frequency), bir kelimenin bir dokümandaki önemini belirleyen istatistiksel bir ölçüdür:

```python
def TF_IDFHesaplama(tokenized_text, word_index):
    tfidf_vectorizer=TfidfVectorizer(vocabulary=list(word_index.keys()))
    tfidf_matrix=tfidf_vectorizer.fit_transform([" ".join(sentence) for sentence in tokenized_text])
    return tfidf_matrix.toarray()
```

TF-IDF iki bileşenden oluşur:
- **TF (Term Frequency)**: Bir kelimenin bir dokümanda kaç kez geçtiğini ölçer
- **IDF (Inverse Document Frequency)**: Bir kelimenin tüm doküman koleksiyonundaki yaygınlığının tersini ölçer

Formül olarak: TF-IDF = TF × IDF

### TF-IDF'in Önemi

TF-IDF vektörizasyonu:
- Nadir kelimelere daha yüksek ağırlık verir
- Çok yaygın kelimelerin (stopwords) etkisini azaltır
- Doküman benzerliği hesaplamalarında ve bilgi çıkarımında etkilidir
- Metin sınıflandırma ve kümeleme için güçlü özellikler sağlar

## 6. Pratik Uygulamalar ve Kullanım Alanları

İncelediğimiz kod, NLP'nin temel adımlarını uygulamalı olarak göstermektedir. Bu teknikler çeşitli alanlarda kullanılabilir:

### Metin Sınıflandırma
Tokenizasyon ve TF-IDF vektörizasyonu, metinleri kategorilere ayırmak için kullanılabilir (örn. spam filtreleme, duygu analizi).

### Bilgi Çıkarımı
Tokenize edilmiş metinlerden önemli bilgileri çıkarmak için kullanılabilir (örn. isim varlık tanıma, anahtar kelime çıkarımı).

### Doküman Benzerliği
TF-IDF matrisleri, dokümanlar arasındaki benzerliği hesaplamak için kullanılabilir (örn. arama motorları, öneri sistemleri).

### Dil Modelleri
Tokenizasyon ve indeksleme, dil modellerinin eğitilmesi için temel adımlardır (örn. makine çevirisi, metin üretimi).

## Sonuç

Tokenizasyon ve indeksleme, NLP'nin temel yapı taşlarıdır. Bu işlemler, ham metni makine öğrenimi modellerinin anlayabileceği sayısal temsillere dönüştürerek, bilgisayarların insan dilini işlemesini mümkün kılar.

İncelediğimiz kod, bu temel NLP adımlarını basit ve anlaşılır bir şekilde uygulamaktadır. Gerçek dünya uygulamalarında, daha gelişmiş tokenizasyon teknikleri, daha karmaşık vektör temsilleri (word embeddings gibi) ve daha sofistike modeller kullanılsa da, bu temel adımlar her zaman NLP işlem hattının merkezinde yer alır.

NLP alanındaki ilerlemeler devam ettikçe, tokenizasyon ve indeksleme teknikleri de gelişmeye devam edecektir. Ancak, bu temel kavramları anlamak, daha ileri NLP uygulamalarını geliştirmek için sağlam bir temel oluşturacaktır.
