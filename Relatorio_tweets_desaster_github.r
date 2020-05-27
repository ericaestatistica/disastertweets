---
title: "Disaster Tweets"
author: "Erica"
date: "21 de maio de 2020"
output: rmarkdown::github_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

![](disasters.jpg)

## Introduction/Objective
**Purpose**: classify tweets in disasters or not. 
In this project we will different types of classifiers in order to 
classify a set of tweets. The algorithm should be able to figure out if a tweet is reporting
a disaster or not. 

# Methods:
Naive Bayes\
SVM\
Deep Leaning

# Technologies:
R\
Keras\
e1071\
caret

## Analysis

First we need to load the required packages.

```{r packages, message=FALSE, warning=FALSE}
require(data.table)
require(tidytext)
require(dplyr)
require(tm)
require(SnowballC)
require(wordcloud)
require(e1071)
library(keras) 
require(gmodels)
require(kernlab)
require(caret)
```

Now we will read and process the dataset. We will split the data in two parts
in order to test our algorithms, 80% will be used for training and, 20% for tasting. 
After that we need to create our corpus of texts, which will be composed of our set of tweets.

```{r cars}

dados_treinamento<- fread("train.csv")
# Transform label in factor
dados_treinamento$target<-factor(dados_treinamento$target)

# Separate data in training (80%) and test(20%)
set.seed(1) #fix seed
train_index<-sample(c(0,1),replace=T,prob=c(0.2,0.8),nrow(dados_treinamento)) #index training


# Create corpus
tweets_corpus_train <- VCorpus(VectorSource(dados_treinamento$text[train_index==1]))
tweets_corpus_test <- VCorpus(VectorSource(dados_treinamento$text[train_index==0]))

# Separate labels

y_train=dados_treinamento$target[train_index==1]
y_test=dados_treinamento$target[train_index==0]

```


We need to process the texts. For this, will transform all words to lower case, remove numbers, stop words, white spaces and punctioations. We will also apply stem to all the words, which means that we will extract its root. 

```{r tweests}

# Text processing
tweets_corpus_clean_train <- tweets_corpus_train %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords()) %>%
  tm_map(removePunctuation) %>%
  tm_map(stemDocument) %>%
  tm_map(stripWhitespace)

tweets_corpus_clean_test <- tweets_corpus_test %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords()) %>%
  tm_map(removePunctuation) %>%
  tm_map(stemDocument) %>%
  tm_map(stripWhitespace)

# Print some tweets after processing
for(i in 1:3){
  print(as.character(tweets_corpus_clean_train [[i]]))
}

```

## Naive Bayes
In order to use the Naive Bayes classifier, we need to transform the word data
into  numeric data. Since this method use the bag of words approach, we will use the
document frequency matrix.

```{r documentmatrix}

# Create document term matrix
tweets_dtm_train<- DocumentTermMatrix(tweets_corpus_clean_train)
tweets_dtm_test<- DocumentTermMatrix(tweets_corpus_clean_test)


```

It always is important to visualize the data. When we are dealing with texts, a good
tool is the wordcloud, which shows the most frequent words with their sizers proportional to the frequencies. We will display the different types of wordclouds, one for the whole training data and two separated for tweets which are disaster and not.

```{r wordcloud, message=FALSE, warning=FALSE}

# Wordcloud general
wordcloud(tweets_corpus_clean_train, min.freq = 50, random.order = FALSE)

# wordcloud desasters 
wordcloud(tweets_corpus_clean_train[y_train==1], min.freq = 50, random.order = FALSE)

# wordcloud not desasters 
wordcloud(tweets_corpus_clean_train[y_train==0], min.freq = 50, random.order = FALSE)


```


Now is time to select the features that will be used in the algorithm. In this case the features are the words. Will elminate some of them, all that have frequency smaller than 5 in the corpus. After that we will transform the frequency matrix in a binary matrix, it will receive
value "yes" it the word appears in the document and "no", otherwise. This second step in necessary because we will use a boolean version of the Naive Bayes classifier. This means that the frequency of one word is replaced by "yes/no" if it appears or not in the text. The idea behind this is that word occurrence matters more than word frequency.

```{r processmodel, message=FALSE, warning=FALSE}

# Select words with frequency greater than 5
tweets_dtm_freq_train <- tweets_dtm_train %>%
  findFreqTerms(5) %>%
  tweets_dtm_train[ , .]

tweets_dtm_freq_test <- tweets_dtm_test %>%
  findFreqTerms(5) %>%
  tweets_dtm_test[ , .]

# Function to convert numeric in yes or no
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

## Convert the matrices in yes or no
tweets_train <- tweets_dtm_freq_train %>%
  apply(MARGIN = 2, convert_counts)

tweets_test <- tweets_dtm_freq_test %>%
  apply(MARGIN = 2, convert_counts)


```
To train  the model naiveBayes function from the 'e1071' package. We will apply the laplace correction in order to deal with the cases in which the words do not appear in the corpus. After
that we will use this model to predict the labels of tweets in the test data. 

```{r model, message=FALSE, warning=FALSE}
# Trainning the model
tweets_classifier <- naiveBayes(tweets_train, dados_treinamento$target[train_index==1],
                                 laplace = 1)

# Predicting the tweets for the test data. 
tweets_pred <- predict(tweets_classifier,tweets_test)

```

Now we need to compare the real labels with the ones predicted by the model. We will use a confusion matrix to do that.

```{r confusion, message=FALSE, warning=FALSE}

confusion_matrix<-CrossTable(tweets_pred, dados_treinamento$target[train_index==0], prop.chisq = FALSE, chisq = FALSE, 
           prop.t = FALSE,
           dnn = c("Predicted", "Actual"))

conf_mtx <- table(tweets_pred, dados_treinamento$target[train_index==0])

print(conf_mtx)
print(paste0('Correctly predicted ',
             sum(diag(conf_mtx)), ' of ',
             sum(conf_mtx), ' tweets, which means ', 
             round(100 * sum(diag(conf_mtx))/sum(conf_mtx), 2), 
             '% of the total.'))


```

The accuracy for this model turns out to be `r   round(100 * sum(diag(conf_mtx))/sum(conf_mtx), 2)`%.


# SVM
We can use the document frequency terms to fit another model. We will use now the Suport Vector Machine. We need to garantee that the train and test matrices have the same columns. Therefore we will discard of the data all words that do not apeears in both. After that we will fit the model using the caret package and the svmLinear3 method. Finally we will predict the labels and compute the confusion matrix and print the acuracy of the model. 


```{r ksvm, message=FALSE, warning=FALSE}

# Covert the document matrix in matrix object

train.dtm<-  as.matrix(DocumentTermMatrix(tweets_corpus_clean_train))
test.dtm<-  as.matrix(DocumentTermMatrix(tweets_corpus_clean_test))

# Select only the columns of the common words in the test and training data
train.df <- data.frame(train.dtm[,intersect(colnames(train.dtm), colnames(test.dtm))])
test.df <- data.frame(test.dtm[,intersect(colnames(test.dtm), colnames(train.dtm))])

# Create the vector of labels in the training set
train.df$target<-dados_treinamento$target[train_index==1]

# Fit the model
df.model<-train(target~., data = train.df, method = 'svmLinear3')

# Predict labels for the test data
df.pred<-predict(df.model, test.df)


confusion_matrix_svm<-CrossTable(df.pred, dados_treinamento$target[train_index==0], prop.chisq = FALSE, chisq = FALSE, 
           prop.t = FALSE,
           dnn = c("Predicted", "Actual"))



conf_mtx <- table(df.pred, dados_treinamento$target[train_index==0])

print(conf_mtx)
print(paste0('Correctly predicted ',
             sum(diag(conf_mtx)), ' of ',
             sum(conf_mtx), ' tweets, which means ', 
             round(100 * sum(diag(conf_mtx))/sum(conf_mtx), 2), 
             '% of the total.'))


```
The accuracy for this model turns out to be `r   round(100 * sum(diag(conf_mtx))/sum(conf_mtx), 2)` %.

# Deep Learning

Both algorithms were applied before, use the bag of words approach. Since the results were not very good, we will try to train a model that considers the dependence structure of the tex. This can be done using Deep Learning models. This model will be fitted using Keras package. The text processing in this case is a little bit different.
First we need to tokenize the sentences. This means that we need to divide them into words and associate each one with an integer. 
To do this we will use the function text_tokenizer from Keras package. This function receives as one parameter the maximum number of words in the corpus. We will fix this in 1000 words, this means that we will consider only the 1000 most frequent words. 

```{r tokenizer, message=FALSE, warning=FALSE}
# Tokenization
max_features <- 1000
tokenizer <- text_tokenizer(num_words = max_features)


tokenizer %>% 
  fit_text_tokenizer(dados_treinamento$text)

```

We can now see the number of documents and see some examples of integer associated with each word.
So we have a dictionary of integers and the number associated with each word. The next step is to associate this sequence of integers with each tweet of our data. This will be done using the function texts_to_sequences.

```{r examples, message=FALSE, warning=FALSE}
# Number of documents
tokenizer$document_count

# Some examples of numbers associate o
tokenizer$word_index %>%
  head()
# Associate intergers to tweets
text_seqs<- texts_to_sequences(tokenizer, dados_treinamento$text)

# Some examples
text_seqs %>%
  head()
```

Next we need to define the paremeters of the model. 

```{r parameters, message=FALSE, warning=FALSE}

# Set parameters:
maxlen <- 100
batch_size <- 32
embedding_dims <- 50
filters <- 64
kernel_size <- 3
hidden_dims <- 50
epochs <- 5
```

In order to fit the model, the documents should have the same length.
To fix this we need to pad the sentences (add zeros) to force them to have the same length. This will
be done with the function pad_sequences. At the same time, we will
split the data in training and test data.

```{r split, message=FALSE, warning=FALSE}

# Padding setences
x_train <- text_seqs[train_index==1] %>%
  pad_sequences(maxlen = maxlen)
dim(x_train)

x_test <- text_seqs[train_index==0] %>%
  pad_sequences(maxlen = maxlen)
dim(x_test)



y_train <- as.numeric(dados_treinamento$target[train_index==1])-1
length(y_train)

y_test <- as.numeric(dados_treinamento$target[train_index==0])-1
length(y_test)

```

Now we will define and fit the model. The first step is the word embedding. We need
to associate each word with a numeric vector that will be used as input of the model.
This vector is estimated in a way tha the considers the context that each word appears in the documents. Words with symilar meanings will have vectors that are close to each other.
In the next step wi will define some regularization method. We will use the dropout tecnice with a probability of 0.2. This means that 20% of the neurons will randomly chosen and omitted
of the network during during a particular forward or backward pass.
Then we define the next layer, which is a convolution with one dimension. The next layer is a max pooling operator. Than 2 dense layers are added, one relu activation function and finally 
a sigmoid activation function is applied to generate the output. We will define the loss function as a binary cross entropy and use the adam maximizer. 


```{r modelkeras, message=FALSE, warning=FALSE}

model_keras <- keras_model_sequential() %>% 
  layer_embedding(max_features, embedding_dims, input_length = maxlen) %>%
  layer_dropout(0.2) %>%
  layer_conv_1d(
    filters, kernel_size, 
    padding = "valid", activation = "relu", strides = 1
  ) %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(hidden_dims) %>%
  layer_dropout(0.2) %>%
  layer_activation("relu") %>%
  layer_dense(1) %>%
  layer_activation("sigmoid") %>% compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )


hist <- model_keras %>%
  fit(
    x_train,
    y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.3
  )
```

Now we will see how the model performs in the test data. We will consider the threshold probability as 0.5. Then we will print the confusion matrix.

```{r keraspredict, message=FALSE, warning=FALSE}

pred <- model_keras %>% # let keras sweat...
  predict_proba(x_test)

label_pred<-ifelse(pred>0.5,1,0)

conf_mtx <- table(label_pred, y_test)

confusion_matrix<-CrossTable(label_pred, y_test, prop.chisq = FALSE, chisq = FALSE, 
           prop.t = FALSE,
           dnn = c("Predicted", "Actual"))

print(paste0('Correctly predicted ',
             sum(diag(conf_mtx)), ' of ',
             sum(conf_mtx), ' tweets, which means ', 
             round(100 * sum(diag(conf_mtx))/sum(conf_mtx), 2), 
             '% of the total.'))
```

The accuracy for this model turns out to be `r   round(100 * sum(diag(conf_mtx))/sum(conf_mtx), 2)` %. We see that in this case the most complex model did not lead tho the best results. 