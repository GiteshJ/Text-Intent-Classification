
# coding: utf-8

# In[1]:


import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


# In[2]:


def plot_confusion_matrix(y_actu, y_pred, title='Confusion matrix', cmap=plt.cm.gray_r):
    
    df_confusion = pd.crosstab(y_actu, y_pred.reshape(y_pred.shape[0],), rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
def predict(X, Y, W, b, word_to_vec_map):
    
    m = X.shape[0]
    pred = np.zeros((m, 1))
    
    for j in range(m):                      
        words = X[j].lower().split()
        
       
        avg = np.zeros((50,))
        for w in words:
            avg += word_to_vec_map[w]
        avg = avg/len(words)

        
        Z = np.dot(W, avg) + b
        A = softmax(Z)
        pred[j] = np.argmax(A)
        
    print("Accuracy: "  + str(np.mean((pred[:] == Y.reshape(Y.shape[0],1)[:]))))
    
    return pred
def print_predictions(X, pred):
    print()
    for i in range(X.shape[0]):
        print(X[i], label_to_emoji(int(pred[i])))


# In[29]:


X_train, Y_train = read_csv('dataset.csv')
X_test, Y_test = read_csv('dataset.csv')


# In[23]:


maxLen = len(max(X_train, key=len).split())
print(X_train.shape[0])


# In[8]:


index = 5
print(X_train[index], Y_train[index])


# In[9]:


Y_oh_train = convert_to_one_hot(Y_train, C = 3)
Y_oh_test = convert_to_one_hot(Y_test, C = 3)


# In[10]:


word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')


# In[11]:


word = "cucumber"
index = 289846
print("the index of", word, "in the vocabulary is", word_to_index[word])
print("the", str(index) + "th word in the vocabulary is", index_to_word[index])
print(word_to_vec_map['cucumber'])


# In[12]:




def sentence_to_avg(sentence, word_to_vec_map):
    
    words = [i.lower() for i in sentence.split()]

    avg = np.zeros((50,))
    
    for w in words:
        avg += word_to_vec_map[w]
    avg = avg / len(words)
    
    
    return avg


# In[18]:


avg = sentence_to_avg("I APPLIED FOR A CREDIT CARD LAST MONTH BUT I DID NOT GET THAT ONE TILL NOW ALTHOUGH I FULLFILL ALL THE CRITERIA REQUIRED FOR APPLYING CREDIT CARD .", word_to_vec_map)
print("avg = ", avg)


# In[62]:


X_train, Y_train = read_csv('dataset.csv')
X_test, Y_test = read_csv('dataset.csv')
for i in range(X_train.shape[0]):
    
    avg = sentence_to_avg(X_train[i], word_to_vec_map)
    


# In[63]:




def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 4):
   
    
    np.random.seed(1)

    
    m = Y.shape[0]                        
    n_y = 5                                 
    n_h = 50                              
    
    
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    
    
    Y_oh = convert_to_one_hot(Y, C = n_y) 
    
   
    for t in range(num_iterations):                       
        for i in range(m):                               
            
          
            avg = sentence_to_avg(X[i], word_to_vec_map)

         
            z = np.dot(W, avg) + b
            a = softmax(z)

           
            cost = -np.sum(np.multiply(Y_oh[i], np.log(a)))
           
            
          
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz

            
            W = W - learning_rate * dW
            b = b - learning_rate * db
        
        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map)

    return pred, W, b


# In[64]:


print(X_train.shape)
print(Y_train.shape)
print(np.eye(3)[Y_train.reshape(-1)].shape)
print(X_train[0])
print(type(X_train))
print(np.eye(3)[Y_train.reshape(-1)].shape)
print(type(X_train))


# In[65]:


pred, W, b = model(X_train, Y_train, word_to_vec_map)
print(pred)


# In[66]:


print("Training set:")
pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
print('Test set:')
pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)




print(Y_test.shape)
print('           '+ ' 0 '+ '    ' + ' 1 ' + '    ' +  ' 2 '+ '    ')
print(pd.crosstab(Y_test, pred_test.reshape(27,), rownames=['Actual'], colnames=['Predicted'], margins=True))
plot_confusion_matrix(Y_test, pred_test)

