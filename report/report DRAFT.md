#  Neural Network Weights Debugger 
# Report
### Authors: Lindsey Sawatzky, Volodymyr Kozyr

## Introduction, Motivation

Nowadays neural networks took one of the leading places in computing science research. While you are developing a neural network, you will most probably encounter a step when you have to evaluate your neural network. Imagine a situation that you are pretty sure that the structure of your neural network perfectly fits your task and data processing is correct but somehow you are getting wrong results on your testing data. Now you need something that's called "debugging" tools.

There are some existing tools like tensorboard which allow you to see the learning process, weights changing of your neural network etc.
However, if you are trying to get deeper, you will see just bunch of numbers, which is impossible to follow step by step especially if your neural network has big amount of nodes and/or layers.

Neural Network Weights Debugger was mostly designed for people who are developing and/or doing research about neural networks to help them to increase speed and productivity of debugging.

## Tools
For your visualization task, we decided to use pure d3.js and simple python server on the backend to keep project simple enough to deliver a working prototype in such a short-term.

## Visualization task
As a visualization task, we took a neural network that for the given N words, it will output the next word that fits the sequence the most.
The main idea is to present word vectors not as numbers, but as colorful widgets with operations between them.
(pic. 1.png)

This allows not to overwhelm users with unnecessary information that is not helpful for debugging.

Each word has a vector of numbers (which is a word representation for neural network input) and assigned color which is chosen according to the dimensionality reduction algorithm. 

Similar words will have similar colors (pic. 2.png)

## UI Description
On each step, user can input a word into the neural network and tool will show the output of the list of most probable words which will be next with all vectors in layers and operations which led to result, then user can enter the next word and so on.
(pic. 3.png)

To see each operation in more details, user can zoom in by clicking on the operation sign. (pic. 4.png)

But the most useful information user can get by clicking on a particular element of the vector and it will be shown (gray highlight) which elements of other vectors made the most influence of forming the value of this particular element (pic. 5.png)

## Results, conclusions, further work
As a result, we have a fully working interactive demo of the Neural Network Weights Visualization.
The tough part is to choose a correct algorithm for dimensionality reduction to get word coloring because sometimes you can not really see the difference between colors of different words. But this highly depends on the training dataset, so the algorithm should be tuned for language models with different vocabularies.
This demo represents a structure for a particular neural network with a fixed structure. A great addition would be implementing support for generic neural networks.

