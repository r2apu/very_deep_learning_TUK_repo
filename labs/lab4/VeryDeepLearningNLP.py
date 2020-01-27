#!/usr/bin/env python
# coding: utf-8

# # Exercise 5 (NLP): Very Deep Learning
# 
# **Natural language processing (NLP)** is the ability of a computer program to understand human language as it is spoken. It involves a pipeline of steps and by the end of the exercise, we would be able to classify the sentiment of a given review as POSITIVE or NEGATIVE.
# 
# 
# Before starting, it is important to understand the need for RNNs and the lecture from Stanford is a must to see before starting the exercise:
# 
# https://www.youtube.com/watch?v=iX5V1WpxxkY
# 
# When done, let's begin. 

# In[1]:


# In this exercise, we will import libraries when needed so that we understand the need for it. 
# However, this is a bad practice and don't get used to it.
import numpy as np

# read data from reviews and labels file.
with open('data/reviews.txt', 'r') as f:
    reviews_ = f.readlines()
with open('data/labels.txt', 'r') as f:    
    labels = f.readlines()


# In[3]:


# One of the most important task is to visualize data before starting with any ML task. 
for i in range(5):
    print(labels[i] + "\t: " + reviews_[i][:100] + "...")


# 
# 
# We can see there are a lot of punctuation marks like fullstop(.), comma(,), new line (\n) and so on and we need to remove it. 
# 
# Here is a list of all the punctuation marks that needs to be removed 
# ```
# (!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~)
# ```
# 

# ## Task 1: Remove all the punctuation marks from the reviews.
# Many ways of doing it: Regex, Spacy, import punctuation from string.

# In[4]:


# Make everything lower case to make the whole dataset even. 
reviews = ''.join(reviews_).lower()


# In[5]:


# complete the function below to remove punctuations and save it in no_punct_text

def text_without_punct(reviews):
    for ichar in '''!"#$%&'()*+,-./:;<=>?@[]^_`{|}~''':
        reviews = reviews.replace(ichar, '')
    return reviews

no_punct_text = text_without_punct(reviews)
reviews_split = no_punct_text.split('\n')
#print(reviews_split[1000:2000])


# In[6]:


# split the formatted no_punct_text into words
def split_in_words(no_punct_text):
    return no_punct_text.split()

words = split_in_words(no_punct_text)
#print(words[:1000])


# In[7]:


# once you are done print the ten words that should yield the following output
words[:10] #['bromwell', 'high', 'is', 'a', 'cartoon', 'comedy', 'it', 'ran', 'at', 'the']


# In[8]:


# print the total length of the words
len(words) #6020196


# In[9]:


# Total number of unique words
len(set(words)) #74072


# 
# Next step is to create a vocabulary. This way every word is mapped to an integer number.
# ```
# Example: 1: hello, 2: I, 3: am, 4: Robo and so on...
# ```
# 

# In[10]:


# Lets create a vocab out of it

# feel free to use this import 
from collections import Counter

## Let's keep a count of all the words and let's see how many words are there. 
def word_count(words):
    return Counter(words)

counts = word_count(words)


# In[11]:


# If you did everything correct, this is what you should get as output. 
print (counts['wonderful']) #1658
print (counts['bad']) #9308


# ## Task 2: Word to Integer and Integer to word
# The task is to map every word to an integer value and then vice-versa. 
# 

# In[12]:


# define a vocabulary for the words
def vocabulary(counts):
    #counts is a dict
    return sorted([key for key in counts])

vocab = vocabulary(counts)
print(len(vocab))
vocab[1] #and -> Ahm, no


# In[13]:


# map each vocab word to an integer. Also, start the indexing with 1 as we will use 
# '0' for padding and we dont want to mix the two.
def vocabulary_to_integer(vocab):
    return {k:(i+1) for i,k in enumerate(vocab)}

vocab_to_int = vocabulary_to_integer(vocab)


# In[14]:


# verify if the length is same and if 'and' is mapped to the correct integer value.
print(len(vocab_to_int)) #74072
print(vocab_to_int['and']) #2 -> Ahm, no
print(vocab_to_int['high']) #2


# Let's see what positve words in positive reviews we have and what we have in negative reviews. 

# In[15]:


positive_counts = Counter()
negative_counts = Counter()


# In[16]:


for i in range(len(reviews_)):
    if(labels[i] == 'positive\n'):
        for word in reviews_[i].split(" "):
            positive_counts[word] += 1
    else:
        for word in reviews_[i].split(" "):
            negative_counts[word] += 1


# In[17]:


labels


# In[18]:


positive_counts.most_common()


# In[19]:


negative_counts.most_common()


# The above is just to show the most common words in the positive and negative sentences. However, there are a lot of unnecessary words like `the`, `a`, `was`, and so on. Can you find a way to show the relevant words and not these words? 
# 
# ```
# Hint: Stop Words removal or normalizing each term.
# ```

# In[20]:


words[:30]


# In[21]:


[vocab_to_int[word] for word in words[:30]]


# In[22]:


vocab_to_int['bromwell'] #21025


# ## One hot encoding
# 
# We need one hot encoding for the labels. Think of a reason why we need one hot encoded labels for classes?
# 
# ## Task 3: Create one hot encoding for the labels. 
# 
# * Write the one hot encoding logic in the `one_hot` function.
# * Use 1 for positive label and 0 for negative label.
# * Save all the values in the `encoded_labels` function.

# In[23]:


# 1 for positive label and 0 for negative label
def one_hot(labels):
    return [1 if label == 'positive\n' else 0 for label in labels]
encoded_labels = one_hot(labels)

#print(encoded_labels[0:500])


# In[24]:


#print the length of your label and uncomment next line only if the encoded_labels size is 25001.
# If you dont get the intuition behind this step, print encoded_labels to see it.
#encoded_labels = encoded_labels[:25000]


# In[25]:


len(encoded_labels)


# In[26]:


reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])


# In[27]:


# This step is to see if any review is empty and we remove it. Otherwise the input will be all zeroes.
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0])) #1
print("Maximum review length: {}".format(max(review_lens))) #2514


# In[28]:


print('Number of reviews before removing outliers: ', len(reviews_ints)) #25001

## remove any reviews/labels with zero length from the reviews_ints list.

# get indices of any reviews with length 0
non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

# remove 0-length reviews and their labels
reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

print('Number of reviews after removing outliers: ', len(reviews_ints)) #25000


# In[29]:


len(encoded_labels) #25000


# ## Task 4: Padding the data
# 
# > Define a function that returns an array `features` that contains the padded data, of a standard size, that we'll pass to the network. 
# * The data should come from `review_ints`, since we want to feed integers to the network. 
# * Each row should be `seq_length` elements long. 
# * For reviews shorter than `seq_length` words, **left pad** with 0s. That is, if the review is `['best', 'movie', 'ever']`, `[117, 18, 128]` as integers, the row will look like `[0, 0, 0, ..., 0, 117, 18, 128]`. 
# * For reviews longer than `seq_length`, use only the first `seq_length` words as the feature vector.
# 
# As a small example, if the `seq_length=10` and an input review is: 
# ```
# [117, 18, 128]
# ```
# The resultant, padded sequence should be: 
# 
# ```
# [0, 0, 0, 0, 0, 0, 0, 117, 18, 128]
# ```
# 
# **Your final `features` array should be a 2D array, with as many rows as there are reviews, and as many columns as the specified `seq_length`.**

# In[30]:


# Write the logic for padding the data
def pad_features(reviews_ints, seq_length):
    features = np.zeros((len(reviews_ints), seq_length), int)
    for i,rv in enumerate(reviews_ints):
        if len(rv) < seq_length:
            dif = seq_length-len(rv)
            features[i,0:dif] = np.zeros((1,dif),int) # zeros from beggining until difference, padding
            features[i,dif:] = rv[:seq_length] # acutal features the rest of the row
        else:
            features[i,0:] = rv[:seq_length]
    
    return features


# In[31]:


# Verify if everything till now is correct. 

seq_length = 200

features = pad_features(reviews_ints, seq_length=seq_length)

## test statements - do not change - ##
assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches 
print(features[:30,:10])


# Now we have everything ready. It's time to split our dataset into `Train`, `Test` and `Validate`. 
# 
# Read more about the train-test-split here : https://cs230-stanford.github.io/train-dev-test-split.html
# 
# ## Task 5: Lets create train, test and val split in the ratio of 8:1:1.  
# 
# Hint: Either use shuffle and slicing in Python or use train-test-val split in Sklearn. 

# In[32]:


from sklearn.model_selection import train_test_split

train_frac = 0.8
val_frac = 0.1
test_frac = 0.1


def train_test_val_split(features):
    _train, _tmp = train_test_split(features, train_size=train_frac)
    
    _val, _test = train_test_split(_tmp, train_size=val_frac/(val_frac+test_frac))
    return _train, _val, _test

def train_test_val_labels(encoded_labels):
    return train_test_val_split(encoded_labels)

train_x, val_x, test_x = train_test_val_split(features)
train_y, val_y, test_y = train_test_val_labels(encoded_labels)


# In[33]:


## print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),     #(20000, 200) 
      "\nValidation set: \t{}".format(val_x.shape),  #(2500, 200) 
      "\nTest set: \t\t{}".format(test_x.shape))     #(2500, 200)


# ## DataLoaders and Batching
# 
# After creating training, test, and validation data, we can create DataLoaders for this data by following two steps:
# 1. Create a known format for accessing our data, using [TensorDataset](https://pytorch.org/docs/stable/data.html#) which takes in an input set of data and a target set of data with the same first dimension, and creates a dataset.
# 2. Create DataLoaders and batch our training, validation, and test Tensor datasets.
# 
# ```
# train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
# train_loader = DataLoader(train_data, batch_size=batch_size)
# ```
# 
# This is an alternative to creating a generator function for batching our data into full batches.
# 
# ### Task 6: Create a generator function for the dataset. 
# See the above link for more info.

# In[34]:


import torch
from torch.utils.data import TensorDataset, DataLoader

# create Tensor datasets for train, test and val
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 50

# make sure to SHUFFLE your training data. Keep Shuffle=True.
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# In[35]:


# obtain one batch of training data and label. 
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)


# In[36]:


# Check if GPU is available.
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')


# ## Creating the Model 
# 
# Here we are creating a simple RNN in PyTorch and pass the output to the a Linear layer and Sigmoid at the end to get the probability score and prediction as POSITIVE or NEGATIVE. 
# 
# The network is very similar to the CNN network created in Exercise 2. 
# 
# More info available at: https://pytorch.org/docs/0.3.1/nn.html?highlight=rnn#torch.nn.RNN
# 
# Read about the parameters that the RNN takes and see what will happen when `batch_first` is set as `True`.

# In[37]:


import torch.nn as nn

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # RNN layer
        self.rnn = nn.RNN(vocab_size, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # RNN out layer
        rnn_out, hidden = self.rnn(x, hidden)
    
        # stack up lstm outputs
        rnn_out = rnn_out.view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        out = self.dropout(rnn_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

    


# 
# ## Task 7 : Know the shape
# 
# Given a batch of 64 and input size as 1 and a sequence length of 200 to a RNN with 2 stacked layers and 512 hidden layers, find the shape of input data (x) and the hidden dimension (hidden) specified in the forward pass of the network. Note, the batch_first is kept to be True. 
# 
# 

# In[38]:


# Instantiate the model w/ hyperparamsrnn_out
vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
output_size = 1
hidden_dim = 256
n_layers = 1

net = SentimentRNN(vocab_size, output_size, hidden_dim, n_layers)

print(net)


# 
# ## Task 8: LSTM 
# 
# Before we start creating the LSTM, it is important to understand LSTM and to know why we prefer LSTM over a Vanilla RNN for this task. 
# > Here are some good links to know about LSTM:
# * [Colah Blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
# * [Understanding LSTM](http://blog.echen.me/2017/05/30/exploring-lstms/)
# * [RNN effectiveness](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
# 
# 
# Now create a class named SentimentLSTM with `n_layers=2`, and rest all hyperparameters same as before. Also, create an embedding layer and feed the output of the embedding layer as input to the LSTM model. Dont forget to add a regularizer (dropout) layer after the LSTM layer with p=0.4 to prevent overfitting. 

# In[39]:


import torch.nn as nn

class SentimentLSTM(nn.Module):
    """
    The LSTM model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers=2, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentLSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # define embedding, LSTM, dropout and Linear layers here:
        
        #embedding layer
        self.emb_layer = nn.Embedding(vocab_size, embedding_dim)
        
        #LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        
        #Dropout layer to prevent overfitting
        self.regularizer = nn.Dropout(p=0.4)
        
        # linear layers
        self.fc = nn.Linear(hidden_dim, output_size)
        
        #Sigmoid
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        
        batch_size = x.size(0)
        
        #Embedding output:
        emb_out = self.emb_layer(x)
        
        #Output of embedding to lstm
        lstm_out, hidden = self.lstm(emb_out)
        
        #Dropout
        drop_out = self.regularizer(lstm_out)
        
        #fully connected linear layer
        out = self.fc(drop_out)
        
        #sigmoid
        out = self.sig(out)
        
         # reshape to be batch_size first
        out = out.view(batch_size, -1)
        out = out[:, -1] # get last batch of labels
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden
        


# ## Instantiate the network
# 
# Here, we'll instantiate the network. First up, defining the hyperparameters.
# 
# * `vocab_size`: Size of our vocabulary or the range of values for our input, word tokens.
# * `output_size`: Size of our desired output; the number of class scores we want to output (pos/neg).
# * `embedding_dim`: Number of columns in the embedding lookup table; size of our embeddings.
# * `hidden_dim`: Number of units in the hidden layers of our LSTM cells. Usually larger is better performance wise. Common values are 128, 256, 512, etc.
# * `n_layers`: Number of LSTM layers in the network. Typically between 1-3

# In[40]:


# Instantiate the model with these hyperparameters
vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 300
hidden_dim = 256
n_layers = 2

net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)


# In[41]:


# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


# ### Task 9: Loss Functions
# We are using `BCELoss (Binary Cross Entropy Loss)` since we have two output classes. 
# 
# Can Cross Entropy Loss be used instead of BCELoss? 
# 
# If no, why not? If yes, how?
# 
# Is `NLLLoss()` and last layer as `LogSoftmax()` is same as using `CrossEntropyLoss()` with a Softmax final layer? Can you get the mathematical intuition behind it?

# In[43]:


#Training and Validation

epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    net.cuda()

net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        assert len(inputs) == len(labels)

        counter += 1

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        #print('output',output.squeeze(0))
        #print('labels',labels.float())
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))


# ## Inference
# Once we are done with training and validating, we can improve training loss and validation loss by playing around with the hyperparameters. Can you find a better set of hyperparams? Play around with it. 
# 
# ### Task 10: Prediction Function
# Now write a prediction function to predict the output for the test set created. Save the results in a CSV file with one column as the reviews and the prediction in the next column. Calculate the accuracy of the test set.

# In[ ]:


def predict():
    pass


# ## Bonus Question: Create an app using Flask
# 
# > Extra bonus points if someone attempts this question:
# * Save the trained model checkpoints.
# * Create a Flask app and load the model. A similar work in the field of CNN has been done here : https://github.com/kumar-shridhar/Business-Card-Detector (Check `app.py`)
# * You can use hosting services like Heroku and/or with Docker to host your app and show it to everyone. 
# Example here: https://github.com/selimrbd/sentiment_analysis/blob/master/Dockerfile
# 
