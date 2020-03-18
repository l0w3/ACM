# ACM
This project is for classify music in different genres using the spectogram of the song
I used GTZAN dataset, that can be found here: https://www.kaggle.com/carlthome/gtzan-genre-collection

First step is to run "save-img.py". This will go through all the music archives and save the respective images of each music archive on a folder called img_data.

Second is to run "save-img-arrays.py". This will save two pickle files. X.pickle and y.pickle. In X.pickle is stored all the arrays of the images, previously shufeled. In y.pickle we have the categorical lables of the images in X.pickle.

Third and last, we run "CNN-model.py" or "CNN-model_cc.py". The first one creates the model wiht 4 convolutional layers and 2 dense layers of 8 and 10 units. The loss function is sparse categorical crossentropy and the optimizer I used is Adam.
The second one is the same but with the lables changed to be coompatible with categorical crossentropy (the lables were encoded in one-hot encoding). Other small change in the network architecture is that the first dense layer of 8 units was deleted, so there were 4 convolutional layers and 1 dense layer of 10 units.
