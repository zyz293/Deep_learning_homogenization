import numpy as np
import pickle
from keras.models import load_model
from sklearn.metrics import mean_absolute_error

print ('load original data')
with open('data.pkl', 'r') as f:
	final_test_data = np.array(pickle.load(f))

final_test_data = final_test_data - 0.5
print 'testing data shape: ', final_test_data.shape

print 'create model'
model = load_model('best_model.h5')

final_pred_y = np.array(model.predict(final_test_data))
with open('predict_result.pkl', 'w') as f:
	pickle.dump(final_pred_y, f)




