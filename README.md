# tlu
A program that generates threshold logic units (TLUs). This was a hastily finished school project. As such, the code isn't very pretty to look at, but it works. The program 
will spit out an array of weights trained against a dataset. The weights can be used to evaluate the progress of learning for the TLUs thereafter.

The program expects a path to a training set, a validation set, and a file to export the weights. NumPy does most of the heavy lifting with the matrices.
