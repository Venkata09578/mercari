import numpy as np
import pandas as pd
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#importing Keras Libraries
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.models import Sequential
#from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D,MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec
import pickle

#data_train = pd.read_table('C:/Users/anian/.spyder-py3/Deployment test/train.tsv')

#print(type(data_train))
desc_load = pickle.load(open("C:/Users/anian/.spyder-py3/Deployment test/Word2Vec_desc.pkl",'rb'))
maincat_reload = pickle.load(open("C:/Users/anian/.spyder-py3/Deployment test/maincat.pkl",'rb'))
subcat1_reload = pickle.load(open("C:/Users/anian/.spyder-py3/Deployment test/subcat1.pkl",'rb'))
subcat2_reload = pickle.load(open("C:/Users/anian/.spyder-py3/Deployment test/subcat2.pkl",'rb'))
brandname_reload = pickle.load(open("C:/Users/anian/.spyder-py3/Deployment test/brandname.pkl",'rb'))
name_reload = pickle.load(open("C:/Users/anian/.spyder-py3/Deployment test/word2vec_name.pkl",'rb'))
pcaname_reload = pickle.load(open("C:/Users/anian/.spyder-py3/Deployment test/pca_name.pkl",'rb'))
pca_reload = pickle.load(open("C:/Users/anian/.spyder-py3/Deployment test/pca.pkl",'rb'))
screload= pickle.load(open("C:/Users/anian/.spyder-py3/Deployment test/sc.pkl",'rb'))
model_reload = pickle.load(open("C:/Users/anian/.spyder-py3/Deployment test/model.pkl",'rb'))

#mETHOD

#Method

def price_output(data1):
    nltk.download('stopwords')
    #stopwords = stopwords.words('english')
    stop = set(stopwords.words('english')) 
    
    # https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
    def decontracted(phrase):
       # specific
       phrase = re.sub(r"won't", "will not", phrase)
       phrase = re.sub(r"can\'t", "can not", phrase)
    
       # general
       phrase = re.sub(r"n\'t", " not", phrase)
       phrase = re.sub(r"\'re", " are", phrase)
       phrase = re.sub(r"\'s", " is", phrase)
       phrase = re.sub(r"\'d", " would", phrase)
       phrase = re.sub(r"\'ll", " will", phrase)
       phrase = re.sub(r"\'t", " not", phrase)
       phrase = re.sub(r"\'ve", " have", phrase)
       phrase = re.sub(r"\'m", " am", phrase)
       return phrase
    
    
    def tokenize(text):
        """
        sent_tokenize(): segment text into sentences
        word_tokenize(): break sentences into words
        """
        try: 
            regex = re.compile('[' +re.escape(string.punctuation) + '\\"\\r\\t\\n]') #Removing Quotations,Carraige return,newline char, tab.
            text = regex.sub(" ", text) # remove punctuation
            
            tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
            tokens = []
            for token_by_sent in tokens_:
                tokens += token_by_sent
            tokens = list(filter(lambda t: t.lower() not in stop, tokens)) # Remove stop words
            filtered_tokens = [decontracted(w) for w in tokens if re.search('[a-zA-Z0-9]', w)]
            filtered_tokens = [w.lower().strip() for w in filtered_tokens if len(w)>=3] # Changing to lower case and strpping the word , Removing words with lenght less than 3.
            
            return filtered_tokens
                
        except TypeError as e: print(text,e)
        
    
    def split_category(text):
        try: return text.split("/")
        except: return ("Category Unknown", "Category Unknown", "Category Unknown")
    


    #Cleaning Item Description
    data1['clean'] = data1['item'].map(tokenize)
    corpus1 = list(data1['clean'])
    doc1 = [word for word in corpus1[0] if word in desc_load.wv.vocab]
    X2 =[]
    X2.append(np.mean(desc_load[doc1],axis = 0))
    X1 = np.array(X2)
    desc_vecs = pca_reload .transform(X1)
    df_w_vectors = pd.DataFrame(desc_vecs)
    #Renaming Columns
    for i in range(len(df_w_vectors.columns)):
      df_w_vectors.rename(columns={i:'desc_' + str(i)},inplace=True)
    #Conatenating with original dataframe
    main_w_vectors = pd.concat((data1,df_w_vectors), axis=1)




    #Splitting Main category
    main_w_vectors['main_category'], main_w_vectors['subcategory_1'], main_w_vectors['subcategory_2'] = zip(*main_w_vectors['category_name'].apply(lambda x: split_category(x)))
      
    main_w_vectors['main_category'] = maincat_reload.transform(main_w_vectors['main_category'])
    main_w_vectors['subcategory_1'] = subcat1_reload.transform(main_w_vectors['subcategory_1'])
    main_w_vectors['subcategory_2'] = subcat2_reload.transform(main_w_vectors['subcategory_2'])


    #Brand name word2vec
    main_w_vectors['brand_name'] = brandname_reload.transform(main_w_vectors['brand_name'])

    #Name column cleaning
    main_w_vectors['clean_name'] = main_w_vectors['name'].map(tokenize)
    corpus2 = list(main_w_vectors['clean_name'])
    doc2 = [word for word in corpus2[0] if word in name_reload.wv.vocab]
    X3 =[]
    X3.append(np.mean(name_reload[doc2],axis = 0))
    X4 = np.array(X3) 
    name_vecs = pcaname_reload.transform(X4)
    df_n_vectors = pd.DataFrame(name_vecs)
    #Renaming Columns
    for i in range(len(df_n_vectors.columns)):
      df_n_vectors.rename(columns={i:'name_' + str(i)},inplace=True)
      #Concat Dataframe
    data_final = pd.concat((main_w_vectors,df_n_vectors), axis=1)

    data_final.drop(columns=['item','clean','category_name','name','clean_name'],inplace =True)
    #data_final.shape
    
    data_final = screload.transform(data_final)
    #print(data_final.shape)
    data_final = data_final.reshape((data_final.shape[0], data_final.shape[1], 1))

    y_pred = model_reload.predict(data_final)
    return np.expm1(y_pred)


# =============================================================================
# #Random input to the model
# input = {'item': 'No description yet','category_name': ['Men/Tops/T-shirts'],'brand_name': ['Razer'],'name':'MLB Cincinnati Reds T Shirt Size XL','shipping' : 1,'item_condition_id' : 2}
# #Converting to Df
# data1 = pd.DataFrame(input)
# print(price_output(data1))
# 
# =============================================================================
# =============================================================================
#  #---------------------------------------------------------------------------------------------------------------------------   
# #Filling Null Values  
 #data_train['item_description'].fillna('No description yet',inplace = True) #Replacing Null Values with "No description yet"
# #Apply tokenie and decontracted Methods.
# #from tqdm import tqdm
# 
# =============================================================================

# =============================================================================

# =============================================================================
# data_train['clean_description'] = data_train['item_description'].map(tokenize)
# # 
# # #Storing the list of clean description to corpus
# # 
#  corpus = list(data_train['clean_description'])
#  description_list = [desc for desc in data_train['item_description']]
#  
#  def document_vector(word2vec_model, doc):
#     # remove out-of-vocabulary words
#     doc = [word for word in doc if word in word2vec_model.wv.vocab]
#     return np.mean(word2vec_model[doc], axis=0)
# 
# # Here, we need each document to remain a document 
# def preprocess(text):
#     text = text.lower()
#     doc = word_tokenize(text)
#     doc = [word for word in doc if word not in stop]
#     doc = [word for word in doc if word.isalpha()] 
#     return doc
# 
# # Function that will help us drop documents that have no word vectors in word2vec
# def has_vector_representation(word2vec_model, doc):
#     """check if at least one word of the document is in the
#     word2vec dictionary"""
#     return not all(word not in word2vec_model.wv.vocab for word in doc)
# 
# # Filter out documents
# def filter_docs(corpus, texts, condition_on_doc):
#     """
#     Filter corpus and texts given the function condition_on_doc which takes
#     a doc. The document doc is kept if condition_on_doc(doc) is true.
#     """
#     number_of_docs = len(corpus)
# 
#     if texts is not None:
#         texts = [text for (text, doc) in zip(texts, corpus)
#                  if condition_on_doc(doc)]
# 
#     corpus = [doc for doc in corpus if condition_on_doc(doc)]
# 
#     print("{} docs removed".format(number_of_docs - len(corpus)))
# 
#     return (corpus, texts)
# 
# corpus = list(data_train['clean_description'])
# 
# model = Word2Vec(data_train['clean_description'], min_count=10,size = 350)
# model.save("C:/Users/anian/.spyder-py3/Deployment test/word2vec1.pkl")
# 
# # #print(corpus[0])
# 
# # #Fitting word2vec model
# #model = Word2Vec(data_train['clean_description'], min_count=10,size = 350)
# # 
# # def document_vector(word2vec_model, doc):
# #     # remove out-of-vocabulary words
# #     doc = [word for word in doc if word in word2vec_model.wv.vocab]
# =============================================================================
#     return np.mean(word2vec_model[doc], axis=0)
# 
# description_list = [desc for desc in data_train['item_description']]
# 
# # Here, we need each document to remain a document 
# def preprocess(text):
#     text = text.lower()
#     doc = word_tokenize(text)
#     doc = [word for word in doc if word not in stop]
#     doc = [word for word in doc if word.isalpha()] 
#     return doc
# 
# # Function that will help us drop documents that have no word  in word2vec
# def has_vector_representation(word2vec_model, doc):
#     """check if at least one word of the document is in the
#     word2vec dictionary"""
#     return not all(word not in word2vec_model.wv.vocab for word in doc)
# 
# # Filter out documents
# def filter_docs(corpus, texts, condition_on_doc):
#     """
#     Filter corpus and texts given the function condition_on_doc which takes
#     a doc. The document doc is kept if condition_on_doc(doc) is true.
#     """
#     number_of_docs = len(corpus)
# 
#     if texts is not None:
#         texts = [text for (text, doc) in zip(texts, corpus)
#                  if condition_on_doc(doc)]
# 
#     corpus = [doc for doc in corpus if condition_on_doc(doc)]
# 
#     print("{} docs removed".format(number_of_docs - len(corpus)))
# 
#     return (corpus, texts)
# 
# 
# # Remove docs that don't include any words in W2V's vocab
# corpus, description_list = filter_docs(corpus, description_list, lambda doc: has_vector_representation(model, doc))
# 
# # Filter out any empty docs
# corpus, description_list = filter_docs(corpus, description_list, lambda doc: (len(doc) != 0))
# 
# # Initialize an array for the size of the corpus
# x = []
# for doc in corpus: # append the vector for each document
#     x.append(document_vector(model, doc))
#     
# X = np.array(x) # list to array
# 
# 
# model.save("C:/Users/anian/.spyder-py3/Deployment test/word2vec.pkl")
# 
# 
# 
# from sklearn.decomposition import PCA
# import pickle
# 
# pca = PCA(n_components=120, random_state=10)
# 
# # x is the array with our 350-dimensional vectors
# reduced_vecs_desc = pca.fit(X)
# 
# pickle.dump(pca, open("C:/Users/anian/.spyder-py3/Deployment test/pca.pkl","wb"))
# =============================================================================
# data_train.columns
# data_train.shape
# data_train[data_train['category_name'] == 'Category Unknown']
# 
# data_train['main_category'], data_train['subcategory_1'], data_train['subcategory_2'] = zip(*data_train['category_name'].apply(lambda x: split_category(x)))
# data_train.head()
# data_train.isnull().sum()
# labelencoder_maincat = LabelEncoder()
# labelencoder_subcat1 = LabelEncoder()
# labelencoder_subcat2 = LabelEncoder()
# data_train['main_category'] = labelencoder_maincat.fit_transform(data_train['main_category'])
# data_train['subcategory_1'] = labelencoder_subcat1.fit_transform(data_train['subcategory_1'])
# data_train['subcategory_2'] = labelencoder_subcat2.fit(data_train['subcategory_2'])
# pickle.dump(labelencoder_maincat, open("C:/Users/anian/.spyder-py3/Deployment test/maincat.pkl","wb"))
# pickle.dump(labelencoder_subcat1, open("C:/Users/anian/.spyder-py3/Deployment test/subcat1.pkl","wb"))
# pickle.dump(labelencoder_subcat2, open("C:/Users/anian/.spyder-py3/Deployment test/subcat2.pkl","wb"))
# =============================================================================

# =============================================================================
    
# =============================================================================
# data_train.brand_name.fillna('unk_brand', inplace=True) #Replaceing all the null values with unk_brand
# 
# print('Guessing null Brands from name and category...')
# 
# # Returning Unique List.
# def concat_categories(x):
#     return set(x.values)
# 
# #Getting unique categories for each brand as a dict 
# brand_names_categories = dict(data_train[data_train['brand_name'] != 'unk_brand'][['brand_name','category_name']].astype('str').groupby('brand_name').agg(concat_categories).reset_index().values.tolist())
# 
# #Validating unique categories for each brand as a dict 
# data_train[data_train['brand_name'] == '% Pure']['category_name'].unique()
# 
# #Brands sorted by length (decreasinly), so that longer brand names have precedence in the null brand search
# brands_sorted_by_size = list(sorted(filter(lambda y: len(y) >= 3, list(brand_names_categories.keys())), key = lambda x: -len(x)))
# 
# #Count of unknow brand in the dataset.
# brand_name_null_count = len(data_train.loc[data_train['brand_name'] == 'unk_brand'])
# 
# #Try to guess the Brand based on Name and Category. Returning brand name if brand is 'name' and category in 'brand_names_categories'.
# def brandfinder(name, category):    
#     for brand in brands_sorted_by_size:
#         if brand in name and category in brand_names_categories[brand]:
#           
#             return brand
#         
#     return 'unk_brand'
# 
# from joblib import Parallel, delayed
# train_names_unknown_brands = data_train[data_train['brand_name'] == 'unk_brand'][['name','category_name']].astype('str').values
# train_estimated_brands = Parallel(n_jobs=1)(delayed(brandfinder)(name, category) for name, category in train_names_unknown_brands) #Returns generator object.
# #print(train_estimated_brands)
# data_train.loc[data_train['brand_name'] == 'unk_brand', 'brand_name'] = train_estimated_brands
# 
# found = brand_name_null_count-len(data_train.loc[data_train['brand_name'] == 'unk_brand'])
# print("Null brands found: %d from %d" % (found, brand_name_null_count))
# 
# labelencoder_brand_name = LabelEncoder()
# data_train['brand_name'] = labelencoder_brand_name.fit_transform(data_train['brand_name'])
# pickle.dump(labelencoder_brand_name, open("C:/Users/anian/.spyder-py3/Deployment test/brandname.pkl","wb"))
# =============================================================================
# =============================================================================
# 
# data_train['clean_name'] = data_train['name'].map(tokenize)
# name_list = [name for name in data_train['name']]
# corpus_name = list(data_train['clean_name'])
# model_name = Word2Vec(data_train['clean_name'], min_count=5,size = 250)
# words_name = list(model_name.wv.vocab)
# 
# def document_vector(word2vec_model, doc):
#     # remove out-of-vocabulary words
#     doc = [word for word in doc if word in word2vec_model.wv.vocab]
#     return np.mean(word2vec_model[doc], axis=0)
# 
# # Here, we need each document to remain a document 
# def preprocess(text):
#     text = text.lower()
#     doc = word_tokenize(text)
#     doc = [word for word in doc if word not in stop]
#     doc = [word for word in doc if word.isalpha()] 
#     return doc
# 
# # Function that will help us drop documents that have no word vectors in word2vec
# def has_vector_representation(word2vec_model, doc):
#     """check if at least one word of the document is in the
#     word2vec dictionary"""
#     return not all(word not in word2vec_model.wv.vocab for word in doc)
# 
# # Filter out documents
# def filter_docs(corpus, texts, condition_on_doc):
#     """
#     Filter corpus and texts given the function condition_on_doc which takes
#     a doc. The document doc is kept if condition_on_doc(doc) is true.
#     """
#     number_of_docs = len(corpus)
# 
#     if texts is not None:
#         texts = [text for (text, doc) in zip(texts, corpus)
#                  if condition_on_doc(doc)]
# 
#     corpus = [doc for doc in corpus if condition_on_doc(doc)]
# 
#     print("{} docs removed".format(number_of_docs - len(corpus)))
# 
#     return (corpus, texts)
# 
# corpus_name, name_list = filter_docs(corpus_name, name_list, lambda doc: has_vector_representation(model_name, doc))
# 
# corpus_name, name_list = filter_docs(corpus_name, name_list, lambda doc: (len(doc) != 0))
# 
# x = []
# for doc in corpus_name: # append the vector for each document
#     x.append(document_vector(model_name, doc))
#     
# X = np.array(x) # list to array
# 
# model_name.save("C:/Users/anian/.spyder-py3/Deployment test/word2vec_name.pkl")
# 
# from sklearn.decomposition import PCA
# 
# pca2 = PCA(n_components=80, random_state=10)
# 
# # x is the array with our 350-dimensional vectors
# reduced_vecs = pca2.fit_transform(X)
# 
#reduced_vecs.isnull().sum()
# pickle.dump(pca2, open("C:/Users/anian/.spyder-py3/Deployment test/pca_name.pkl","wb"))
# =============================================================================
#--------------------------------------------------------------------------------------------------------------------------


