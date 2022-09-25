---------- Decision Tree and Preprocessing --------------
First Compile Preprocessing_and_DT.py

This file contains the pre-processing of data for the KNN and NN algorthms as well as the decision tree algorithm and training. 

This file should compile straight away, without any change necessary. It should print out the results for the decision tree (accuracy) 
as well as training and testing dataset sizes, and decsion tree created. *Please ignore all warnings. 

This will then generate new csv files with the pre-processed data for knn.m algorithm. This should also compile with no necessary changes. 


 --------- KNN --------------
If training or testing data sets have been changed/updated, the following data should be changed in the .m file

n_att = 27   -> Number of attributes in training and testing Data sets
n_ex = 519;  -> Number of instances in Training set
class = n_att + 1; -> Collumn # of classifier (keep this if all the attributes are being used)
n_ex_test = 130; -> Number of instances in Testing set


---------- Neural Network -----------
The NN.py file or NN.ipynb should contain the neural network. 

This document needs the updating of the location of the training and testing sets (google drive location is listed since this was done using google colab)

Please update the following variables for the code to work:
save_path -> location to save the NN localy
train_data_path -> location of the "bdf_train.csv" file (included in .zip file uploaded)
test_data_path -> location of the "bdf_test.csv" file (included in .zip file uploaded)

*For creation of new training and testing sets with different attributes (as outputed from Preprocessing_and_DT.py), the header row must be deleted.
**If number of attributes is different (from 13), it must be changed under line 5 of the model set up
