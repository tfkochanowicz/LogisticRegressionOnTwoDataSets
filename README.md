**Overview:** This project uses the logistic regression model and trains it on two different data sets. A logisitic regression is trained
on a labeled data set(which is usually true or false, black or white, or even cpu or gpu). For example if you wanted to know the difference between a cpu, gpu or embedded processor this would
likely not work or have low precision is a logistic regression was used(this could be a use for a larger neural network with more layers and nodes).


**How to run the app?:** This be ran with python 3.7 (this is what I used when creating this program), and I would recommend using your favorite IDE(I used pycharm). Simply just download the data sets off the links
commented in the program and ensure you are accessing the correct directory.
for exmaple in the code lines 6-8:

# dataset obtained from: https://www.kaggle.com/datasets/michaelbryantds/cpu-and-gpu-product-data

**data = pd.read_csv("--the directory the csv file is contained in--")**

**How does this program work, and what does it do?: **

The first data set is a dataset that contains various unique cpus and gpus that have been in the public 
market since the 90's. During the training of the model the amount of transistors(millions), 
Frequancy of the processor(MHz), Process size (nm) are all used to differentiate the unique cpus and gpus. I chose these
three values for this dataset because upon looking at the dataset I saw the differences very clearly between these 
values when looking at the different cpu and gpus. Once the model was trained it was able to predict
within 98% whether the data was from a gpu or cpu by looking at the values mentioned earlier(frequancy, # of transistors, and Process size).


The second data set is the Wisconsin breast cancer data set. Conceptually this model works almost identically to the first model since they are both logisitic regressions.
This dataset has different measurements of tumors (benign and malignant), and the program will tell you if a particular tumor is benign or malignant
with high precision. This use of the logistic regression likely seems to be much more useful and transferable to a real life application. However the issue with this is the collection of the data, and ensuring
there is high precision in that data.

Once each model's training and testing is done there will be a classification report output about each data set.

**Any questions?:** Email me at tfkochanowicz@gmail.com
