Logistic Function: 
	1. Please ensure you have sklearn, numpy install in your system.
	2. Please put train_x.csv train_y.csv test_x.csv in the same folder (you can use .pkl if you have one)
	3. Run the code by python Logistic.py, it will produce logistic.csv file in the same folder.
	Warning: It may take long time to get the result.
Feedforward Neural Network:
	1. Please ensure you have numpy, pandas and math installed
	2. Please put train_x.csv train_y.csv test_x.csv in the same folder
	3. The code in test.py can be run as a standalone script. Parameters can be changed in the initialization of the network, so long as the number of layers does not exceed 2. The code will run validation at the end of each fold and print the validation loss and accuracy.
Convolutional Neural Network:
	1. Please ensure you have tensorflow, keras, numpy, sklearn, matplotlib installed in your system
	2. Please put train_x.csv train_y.csv test_x.csv in the same folder (you can use .pkl if you have one)
	3. Run the code by python cnn_final.py, it will produce cnn.csv in the same folder, you can see the model building progress in this way.
	Or
	You can just run python cnn_direct_predict.py to get the prediction for test_x fast. It use the existing model cnn.h5 to predict.
 	Notice that the results obtained by cnn.h5 may be slightly different from that on Keras due to unstable model building. We didn't save the model for the kaggle. 
	The accuracy should be around 96 for this cnn.h5， the cnn.h5 can be found in the https://drive.google.com/open?id=1k5yt6xJfLverqsqA5R5jMhYDgUrk2NHy
	