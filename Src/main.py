import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

train_file_name = 'train.csv'
test_file_name = 'test.csv'
input_columns = range(1, 29)
output_column = 29

# read testing and training data from csv files
df_train = pd.read_csv(train_file_name)
df_test = pd.read_csv(test_file_name)

# get training columns
training_inputs = df_train.iloc[:, input_columns]
training_output = df_train.iloc[:, output_column]

# get testing columns
testing_inputs = df_test.iloc[:, input_columns]
testing_output = df_test.iloc[:, output_column]

# for every number of inputs (1 to 28)
for i in range(len(input_columns)):
	
	# find the most accurate (based on high r^2 value) model with i number of features
	best_linear_model = RFE(LinearRegression(), i + 1).fit(training_inputs, training_output)
	
	# test this model with test data and see how it compares to the real testing outputs
	r_squared = best_linear_model.score(testing_inputs, testing_output)
	
	# print results
	accuracy = round(r_squared * 100, 2)
	print(i + 1, 'features:\t', accuracy, '%')