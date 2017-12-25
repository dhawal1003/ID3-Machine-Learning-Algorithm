# ID3-Machine-Learning-Algorithm

- The program builds a binary decision tree classifier using the ID3 algorithm. The attribute(feature) having the highest information gain is selected as the splitting attribute for that particular node. This process is repeated for every node until all the training examples are perfectly classified.

- ID3 helps in selecting the shortest i.e. most compact tree.

- The program reads four arguments from the command line – complete path of the training dataset, complete path of the validation dataset, complete path of the test dataset, and the pruning factor.
- The datasets can contain any number of Boolean attributes and one Boolean class label. The class label will always be the last column.
- The first row will define column names and every subsequent non-blank line will contain a data instance. If there is a blank line, the program skips it.
- After building the classifier on the training dataset, pruning technique is applied on the validation dataset to improve the accuracy of the classifier.
- The final model is then tested with the test datasets.