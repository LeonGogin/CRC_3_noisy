import numpy as np
import tensorflow as tf

from models import generate_model, get_epistemic_uncertainty
from utils import load_data, real_class_dictionary

batch_size = 32
epochs = 20
num_classes = 3
metrics = ['accuracy']
loss_func = tf.keras.losses.categorical_crossentropy



real_labels = ["AC", "AD", "H"]
noise_labels = ["blood", "fat", "glass", "stroma"]

# base_path = "results/"
# model_back_up = base_path + "model.h5"
# history_back_up = base_path + "history.csv"
# uncertaincy_back_up = base_path + "uncertancy.csv"



num_of_models = 5

if __name__ == "__main__":
	x_data, y_data, real_train_class = load_data("train", base_path = "data/")
	x_test_data, y_test_data, real_test_class = load_data("test", base_path = "data/")



	index_real_class = real_class_dictionary(real_train_class)
	index_test_real_class = real_class_dictionary(real_test_class)

	for i in range(num_of_models):
		model = generate_model(num_classes, input_data_size, layer_num = 3, drop_rate = 0.25)
		model.compile(loss = loss_func,
	                      optimizer = tf.keras.optimizers.Adam(),
	                      metrics = metrics)
		mod_history = model.fit(x_data[0], y_data[0],
	                                    batch_size = batch_size,
	                                    epochs=epochs,verbose=1,
	                                    validation_data=(x_data[1], y_data[1]))

		prediction, epistemic = get_epistemic_uncertainty(model, x_test_data, T = 15)
		for i in range(test_dim[0]):

			if len(prediction[i][0]) != 0:
				if real_test_class[i] == "AC":
					dic_predictions["AC"] = np.append(dic_predictions["AC"], np.squeeze(prediction[i]))
					epistemic_lable_uncertainty_AC.append(epistemic[i])

				elif real_test_class[i] == "AD":
					dic_predictions["AD"] = np.append(dic_predictions["AD"], np.squeeze(prediction[i]))
					epistemic_lable_uncertainty_AD.append(epistemic[i])

				elif real_test_class[i] == "H":
					dic_predictions["H"] = np.append(dic_predictions["H"], np.squeeze(prediction[i]))
					epistemic_lable_uncertainty_H.append(epistemic[i])

				elif real_test_class[i] == "blood":
					epistemic_lable_uncertainty_blood.append(epistemic[i])

				elif real_test_class[i] == "fat":
					epistemic_lable_uncertainty_fat.append(epistemic[i])

				elif real_test_class[i] == "glass":
					epistemic_lable_uncertainty_glass.append(epistemic[i])

				elif real_test_class[i] == "stroma":
					epistemic_lable_uncertainty_stroma.append(epistemic[i])
