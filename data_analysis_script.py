import pandas as pd
import spacy
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim import corpora
from gensim.models import LdaModel, Word2Vec, CoherenceModel
from sklearn.cluster import KMeans

def process_text(text):

    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def average_word_vectors(tokens, model, vector_size):

    valid_tokens = [token for token in tokens if token in model.wv]
    if valid_tokens:
        return np.mean(model.wv[valid_tokens], axis=0)
    else:
        return np.zeros(vector_size)

if __name__ == '__main__':

    # Section 1: Preprocess the Text

    df = pd.read_csv('comcast_consumeraffairs_complaints.csv')
    df = df.dropna(subset=['text'])
    
    # Load SpaCy
    nlp = spacy.load('en_core_web_sm')
    
    # Use tqdm for progress
    tqdm.pandas()
    df['c_text'] = df['text'].progress_apply(process_text)
    
    print("Sample preprocessed texts:")
    print(df[['text', 'c_text']].head())
    
    # Tokenize texts for topic modeling
    tokenized_texts = [text.split() for text in df['c_text']]
    

    # Section 2  Build the TF-IDF

    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['c_text'])
    print("\nTF-IDF matrix shape:", tfidf_matrix.shape)
    
    # gensim dictionary & bag-of-words corpus for LDA
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    

    # Section 3  Topic Modeling on TF-IDF

    num_topics = 10  # Fixed no topics
    
    # LDA Topic Extraction on TF-IDF
    
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                         passes=10, random_state=42)
    print("\nLDA Topics on TF-IDF:")
    for topic in lda_model.print_topics(num_words=10):
        print(topic)
    
    # Coherence Score for LDA using TF-IDF
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_texts,
                                         dictionary=dictionary, coherence='c_v')
    lda_coherence = coherence_model_lda.get_coherence()
    print("LDA Coherence Score (TF-IDF):", lda_coherence)
    
    #  NMF Topic Extraction on TF-IDF
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_W = nmf_model.fit_transform(tfidf_matrix)
    nmf_H = nmf_model.components_
    feature_names = tfidf_vectorizer.get_feature_names_out()
    nmf_topics = []
    print("\nNMF Topics on TF-IDF:")
    for topic_idx, topic in enumerate(nmf_H):
        top_words = [feature_names[i] for i in topic.argsort()[-10:]]
        nmf_topics.append(top_words)
        print("Topic {}: {}".format(topic_idx, ", ".join(top_words)))
    
    #  Coherence Score for NMF using TF-IDF
    coherence_model_nmf = CoherenceModel(topics=nmf_topics, texts=tokenized_texts,
                                         dictionary=dictionary, coherence='c_v')
    nmf_coherence = coherence_model_nmf.get_coherence()
    print("NMF Coherence Score (TF-IDF):", nmf_coherence)
    


    # Section 4, Build the Word2Vec Document Embeddings

    w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4)
    doc_vectors = np.array([average_word_vectors(tokens, w2v_model, 100) for tokens in tokenized_texts])
    print("\nWord2Vec document embeddings shape:", doc_vectors.shape)
    

    # Section 5  Clustering on Word2Vec Embeddings

    num_clusters = num_topics
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(doc_vectors)
    df['cluster'] = clusters
    print("\nSample document clusters (Word2Vec based):")
    print(df[['c_text', 'cluster']].head())
    

    # Section 6 Topic Modeling on Word2Vec-based Bag-of-Clusters


    n_clusters_vocab = 100  # clusters for the vocabulary
    vocab_vectors = w2v_model.wv.vectors
    vocab_words = w2v_model.wv.index_to_key
    kmeans_vocab = KMeans(n_clusters=n_clusters_vocab, random_state=42)
    vocab_labels = kmeans_vocab.fit_predict(vocab_vectors)
    # Create mapping
    word_to_cluster = {word: str(label) for word, label in zip(vocab_words, vocab_labels)}
    
    # document's tokens to their corresponding cluster labels
    doc_cluster_texts = []
    for tokens in tokenized_texts:
        cluster_tokens = [word_to_cluster[word] for word in tokens if word in word_to_cluster]
        doc_cluster_texts.append(cluster_tokens)
    
    #  dictionary and corpus for bag-of-clusters
    dictionary_clusters = corpora.Dictionary(doc_cluster_texts)
    corpus_clusters = [dictionary_clusters.doc2bow(doc) for doc in doc_cluster_texts]
    
    # LDA Topic Extraction on Word2Vec-based
    lda_model_w2v = LdaModel(corpus=corpus_clusters, id2word=dictionary_clusters, num_topics=num_topics,
                              passes=10, random_state=42)
    print("\nLDA Topics on Word2Vec-based Bag-of-Clusters:")
    for topic in lda_model_w2v.print_topics(num_words=10):
        print(topic)
    coherence_model_lda_w2v = CoherenceModel(model=lda_model_w2v, texts=doc_cluster_texts,
                                             dictionary=dictionary_clusters, coherence='c_v')
    lda_coherence_w2v = coherence_model_lda_w2v.get_coherence()
    print("LDA Coherence Score (Word2Vec-based):", lda_coherence_w2v)
    
    #   NMF Topic Extraction on Word2Vec-based

    tfidf_vectorizer_clusters = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tfidf_matrix_clusters = tfidf_vectorizer_clusters.fit_transform([' '.join(doc) for doc in doc_cluster_texts])
    nmf_model_w2v = NMF(n_components=num_topics, random_state=42)
    nmf_W_w2v = nmf_model_w2v.fit_transform(tfidf_matrix_clusters)
    nmf_H_w2v = nmf_model_w2v.components_
    feature_names_clusters = tfidf_vectorizer_clusters.get_feature_names_out()
    nmf_topics_w2v = []
    print("\nNMF Topics on Word2Vec-based Bag-of-Clusters:")
    for topic_idx, topic in enumerate(nmf_H_w2v):
        top_words = [feature_names_clusters[i] for i in topic.argsort()[-10:]]
        nmf_topics_w2v.append(top_words)
        print("Topic {}: {}".format(topic_idx, ", ".join(top_words)))
    coherence_model_nmf_w2v = CoherenceModel(topics=nmf_topics_w2v, texts=doc_cluster_texts,
                                              dictionary=dictionary_clusters, coherence='c_v')
    nmf_coherence_w2v = coherence_model_nmf_w2v.get_coherence()
    print("NMF Coherence Score (Word2Vec-based):", nmf_coherence_w2v)



    # Section 7  Compare Results

    print("\n--- Comparison Summary ---")
    print("TF-IDF + LDA Coherence Score:", lda_coherence)
    print("TF-IDF + NMF Coherence Score:", nmf_coherence)
    print("Word2Vec-based (Bag-of-Clusters) + LDA Coherence Score:", lda_coherence_w2v)
    print("Word2Vec-based (Bag-of-Clusters) + NMF Coherence Score:", nmf_coherence_w2v)
    print("Word2Vec Document Embeddings (KMeans clustering) produced", num_clusters, "clusters (see 'cluster' column)")
