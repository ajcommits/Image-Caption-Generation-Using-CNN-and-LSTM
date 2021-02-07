1 Motivation
Image caption generation is a task that requires proper understanding of both Computer Vision(CV) techniques and Natural Language Processing(NLP) methods. It uses techniques from both domains and puts them together hand in hand to generate captions that are apt for any given image. The following areas make use of this technique as a part of their pipelines -

1. Self-Driving cars:
This task renders itself useful to such a use case as we can take an image of the surroundings and if it can automatically generate a useful caption from the image, it can then be converted into a voice command or information that is useful to the driver.

2. Aid to help the blind
I wanted to understand and learn about the model that can do the image caption generation and train one ourselves to understand the various nuances involved in it. We have implemented the use case by trying to make use of \Web Scraping. Thus this project implements the Image caption generation using CNN and LSTM two deep learning models for the computer vision and natural language processing parts respectively.

2 Introduction
The entire project was worked using Google Colab's python environment using Colab's TPU processor. The models's layers and training, loss functions have all made use of Keras library's functions. The rest of the document is ordered as follows - Problem Statement, Datasets, Implementation, Results, Conclusion, References.

• Photo Feature Extractor. This is a 16-layer VGG model pre-trained on the ImageNet dataset.
I have pre-processed the photos with the VGG model (without the output layer) and will use the extracted features predicted by this model as input. Sequence Processor. This is a word embedding layer for handling the text input, followed by a Long Short-Term Memory (LSTM) recurrent neural network layer. Decoder (for lack of a better name). Both the feature extractor and sequence processor output a vector. These are merged together and processed by a Dense layer to make a prediction. The Photo Feature Extractor model expects input photo features to be a vector of 4,096 elements. These are processed by a Dense layer to produce a 256 element representation of the photo.

• The Sequence Processor model expects input sequences with a length (34 words)
which are fed into an Embedding layer that uses a mask to ignore padded values. This is followed by an LSTM layer with 256 memory units. Both the input models produce a 256 element vector. Further, both input models use regularization in the form of 50

• The Decoder model merges the vectors from both input models using an addition operation.
This is then fed to a Dense 256 neuron layer and then to a output Dense layer that makes a softmax prediction over the entire output vocabulary for the next word in the sequence.

3 Datasets
3.1 Training Data
The dataset used in this project is the Flickr8K dataset. The Flickr8K dataset is a free and readily available dataset

• Flickr8K Dataset:
contains a total of 8092 images in JPEG format with di_erent shapes and sizes. Of which 6000 are used for training, 1000 for test and 1000 for development. This covers the number of images that are there in this.

• Flickr8K text:
Contains text _les describing train set, test set.

3.2 Testing data
For testing we have implemented Web Scraping. Web Scraping is a technique employed to extract large amounts of data from websites whereby the data is extracted and saved to a local file in your computer. We have implemented a web scraping function in our code where we enter a query term that describes a component in the image we want to search for on the internet. This image is then scraped the internet and directly fed to the testing part of the model to obtain the captions.

4 Implementation
4.1 Photo Preparation
The idea in this project is to extract the features of the image and directly correlate the higher level features with the appropriate text words that get generated in the NLP network. Thus in order to be able to extract features from the image we use a transfer learning model with VGG as the trained transfer model. Instead of running our images through the entire architecture, we shall download the model weights and feed them to the model as an interpretation of the photo in the dataset. Keras provides us with VGG class which enables us to do just the same. Here we freeze all the layers and their weights except the last two layers and pass our images through them to learn the features thoroughly. We load each photo and extract features from the VGG16 model as a 1x4096 vector. The function will return a dictionary of image identifying tags and the corresponding feature vectors

4.2 Text Preparation
The model makes use of a LSTM as mentioned before which is a Recurrent Neural Network(RNN) for the NLP activites. A RNN is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This makes them ideal to be able to generate words iteratively to form a sentence/caption in our case for our image. Thus in order to prepare the given text in our dataset to meet the RNN standards, we have to perform two main pre-processing steps:

1. Tokenization:
Tokenization is a way of separating a piece of text into smaller units called tokens. Here, tokens can be either words, characters, or subwords. Hence, tokenization can be broadly classifed into 3 types { word, character, and subword (n-gram characters) tokenization • Separate token Id and Image descriptions word by word and put them into two separate variables. • Remove the extension from the image ID. • Now concatenate all the word of a single caption into a string again. • For every image ID store all 5 captions. • Return as a dictionary consisting of lists of image IDs mapped to their corresponding captions.

2. Vocabulary:
The main aim of Tokenization in any NLP task is to ultimately end up with the suitable vocabulary to train the model. Likewise, in order to create meaningful vocabulary for our model to learn from we further do the following pre-processing steps to our tokens: • convert all the words to Lower case. • Remove all the punctuation. • Remove " 's " and 'a' • Remove all word with numbers in them. Finally, our code returns the cleaned words as a set named so that we have unique items in our vocabulary list which was extracted from the annotations document. Now, we make a dictionary of image identifiers and descriptions to a new file and save the mappings to a file. This is later directly called for any further usage. At the end of this step of text preparation we have a vocabulary size of 8793 from the 8092 images' corresponding captions.

Having done this model, the training was then done for 10 epochs. Each epoch took half an hour for training. After training on the 6000 images from the dataset,we move onto testing.

4.3 Testing
The metric used to test this project is the Bilingual Evaluation Understudy Score(BLEU). This score is a metric for evaluating a generated sentence to a reference sentence. A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0. Thus we first tested it using 1000 images from the dataset Flickr8K and obtained the BLEU values. The "query" function paramater bears the string we want to search and scrap of the web.
