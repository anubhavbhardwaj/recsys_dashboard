import os
import sys
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics.pairwise import cosine_similarity
import operator 


sys.path.append(os.path.realpath(".."))
import settings


def load_dataset(file_path):
    """Load and preprocess the file."""
    
    data = pd.read_csv(file_path, header = None )
    data.columns = ["ProductID" , "UserID" , "Rating" , "Time" ] 
    data['Rating'] = data['Rating'].astype('int8')
#     data.drop('Time' , axis = 1 , inplace = True )
    data = data.sort_values(by = "UserID") 
    
    return data


#Recommendor system using Matrix Factorisation using neural networks

def recsys_matfactnn(data):
    
    number_of_samples = 5000
    df  = data.sample(number_of_samples)
    
    user_ids = df["UserID"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}

    product_ids = df["ProductID"].unique().tolist()
    product2product_encoded = {x: i for i, x in enumerate(product_ids)}
    product_encoded2product = {i: x for i, x in enumerate(product_ids)}

    df["user"] = df["UserID"].map(user2user_encoded)
    df["product"] = df["ProductID"].map(product2product_encoded)

    num_users = len(user2user_encoded)
    num_product = len(product_encoded2product)
    df['Rating'] = df['Rating'].values.astype(np.float32)

    min_rating = min(df['Rating'])
    max_rating = max(df['Rating'])
    
    df = df.sample(frac=1, random_state=42)
    x = df[["user", "product"]].values

    y = df["Rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

    train_indices = int(0.7 * df.shape[0])
    val_indices = int(0.9 * df.shape[0]) 

    x_train, x_val, x_test , y_train, y_val , y_test = (
        x[:train_indices],
        x[train_indices:val_indices],
        x[val_indices : ] , 
        y[:train_indices],
        y[train_indices:val_indices], 
        y[val_indices : ])
    
    EMBEDDING_SIZE = 40

    class Recommender(keras.Model):
        def __init__(self, num_users, num_product, embedding_size):
            super(Recommender, self).__init__()
            self.num_users = num_users
            self.num_product = num_product
            self.embedding_size = embedding_size
            self.user_embedding = layers.Embedding(
                num_users,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
            )
            self.user_bias = layers.Embedding(num_users, 1)
            self.product_embedding = layers.Embedding(
                num_product,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6)
            )
            self.product_bias = layers.Embedding(num_product, 1)

        def call(self, inputs):

            user_vector = self.user_embedding(inputs[:, 0])
            product_vector = self.product_embedding(inputs[:, 1])

            user_bias = self.user_bias(inputs[:, 0])
            product_bias = self.product_bias(inputs[:, 1])

            dot_prod = tf.tensordot(user_vector, product_vector, 2)

            x = dot_prod + user_bias + product_bias

            return tf.nn.sigmoid(x)

        def getRecomendation(self , df , user , k )  : 

            encoded_user = user2user_encoded[user]

            all_prods = df['product'].unique() 
            prods = df[df.user == encoded_user]['product'].values
            remainder = list(set(all_prods) - set(prods))
            n = len(remainder) 
            out = np.empty((n, 2),dtype=int)
            out[: ,  0 ] = encoded_user
            out[ : , 1 ] = remainder[:None]
            output = self.predict(out)

            ndx = map(lambda x : product_encoded2product[x] , remainder )
            vals = output[: , 0 ]

            return pd.Series(index = ndx , data = vals).sort_values(ascending = False )[ :k ].index

    model = Recommender(num_users, num_product, EMBEDDING_SIZE)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )

    history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=32,
    epochs=5,
    #     verbose=1,
    validation_data=(x_val, y_val))

    return df, model    
    
#Recommendor system based on user based cosine similarity

def recsys_usercossim(data):
    
    df = data.sample(5000) 
    matrix = df.pivot_table(index = 'UserID' , columns = 'ProductID' , values = 'Rating').fillna(0)
    
    class UserSimilarityRS: 
    
        def __init__(self ,df ,  matrix) :
            self.df = df  
            self.matrix = matrix

        def find_similar_users(self ,userId  , k ) : 

            user = matrix[matrix.index == userId]
            other_users = matrix[matrix.index != userId]

            similarities = cosine_similarity(user,other_users)[0].tolist()

            indices = other_users.index.tolist()

            index_similarity = dict(zip(indices, similarities))

            index_similarity_sorted = sorted(index_similarity.items(), key=operator.itemgetter(1))
            index_similarity_sorted.reverse()

            users = [u[0] for u in index_similarity_sorted[:k] ]

            return users

        def getRecommendations(self , user , k ) : 

            users = self.find_similar_users(user , k )
            products = df[df['UserID'] == user ].ProductID 

            similar_products = df[df['UserID'].isin(users)][['ProductID' , 'Rating']]
            sorted_products = similar_products.groupby('ProductID').agg('median').sort_values(by = "Rating" , ascending = False )

            l = len(sorted_products ) 

            return sorted_products[ : min( l , k )]
    
    model = UserSimilarityRS(df , matrix ) 
    
    return model, df

#Recommendor system based on item based cosine similarity

def recsys_itemcossim(data):
    
    df = data.sample(5000) 
    matrix = df.pivot_table(index= 'ProductID' , columns = 'UserID' , values= 'Rating').fillna(0) 
    
    class ProductSimilarityRS : 
    
        def __init__(self ,df ,  matrix ) : 
            self.df = df 
            self.matrix = matrix 

        def find_similar_product(self , last, k  ) : 

            curr_product = matrix[matrix.index == last]
            other_products = matrix[matrix.index != last]
            similarities = cosine_similarity(curr_product,other_products)[0].tolist()

            indices = other_products.index.tolist()

            index_similarity = dict(zip(indices, similarities))

            index_similarity_sorted = sorted(index_similarity.items(), key=operator.itemgetter(1))
            index_similarity_sorted.reverse()


            l = len(index_similarity_sorted)  

            products = [u for u in index_similarity_sorted[:min(l , k )] ]

            return products 

        def getRecommendations(self, user  , k ) : 

            last_product = df[df['UserID'] == user ].sort_values(by = 'Time' , ascending = False ).ProductID.values[0]
            return self.find_similar_product(last_product ,  k  )  

    model = ProductSimilarityRS(df , matrix )
    
    return model, df


def recsys_popularitybased(data):
    
    df = data.groupby("ProductID").filter(lambda x:x['Rating'].count() >=50).sample(5000) 
    rating_count = df.groupby('ProductID').count()['Rating']
    avg_rating = df.groupby('ProductID')['Rating'].mean()
    df = df.set_index('ProductID')
    df = df.merge(rating_count.rename('rating_count') , left_index = True , right_index= True )
    df = df.merge(avg_rating.rename('avg_rating') , left_index = True , right_index= True)
    df.drop(['UserID' , 'Rating'] , axis = 1  , inplace = True)
    df  = df.sort_values(by =['avg_rating' , 'rating_count'] , ascending  = False )
    df = df[~df.index.duplicated(keep='first')]
    

    return df

def getTopKRecommendations(df , k ) : 
    l  = len(df)
    topkrecs = list(df.index[:min(l , k) ])
    return topkrecs
    