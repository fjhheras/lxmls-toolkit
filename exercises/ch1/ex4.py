import matplotlib.pyplot as plt
import lxmls.classifiers.max_ent_batch as mebc
import lxmls.readers.simple_data_set as sds

sd = sds.SimpleDataSet(
    nr_examples=100,
    g1=[[-0.7, -1], 1.5],
    g2=[[1, 1.2], 1],
    balance=0.3,
    split=[0.5, 0, 0.5],
)

# Perceptron
import lxmls.classifiers.perceptron as percc

perc = percc.Perceptron()
params_perc_sd = perc.train(sd.train_X, sd.train_y)
y_pred_train = perc.test(sd.train_X, params_perc_sd)
acc_train = perc.evaluate(sd.train_y, y_pred_train)
y_pred_test = perc.test(sd.test_X, params_perc_sd)
acc_test = perc.evaluate(sd.test_y, y_pred_test)
print(
    "Perceptron Simple Dataset Accuracy train: %f test: %f"
    % (acc_train, acc_test)
)

fig, axis = sd.plot_data()
fig, axis = sd.add_line(fig, axis, params_perc_sd, "Perceptron", "blue")

# Mira
import lxmls.classifiers.mira as mirac

mira = mirac.Mira()
mira.regularizer = 1.0  # This is lambda
params_mira_sd = mira.train(sd.train_X, sd.train_y)
y_pred_train = mira.test(sd.train_X, params_mira_sd)
acc_train = mira.evaluate(sd.train_y, y_pred_train)
y_pred_test = mira.test(sd.test_X, params_mira_sd)
acc_test = mira.evaluate(sd.test_y, y_pred_test)
print(
    "Mira Simple Dataset Accuracy train: %f test: %f" % (acc_train, acc_test)
)
fig, axis = sd.add_line(fig, axis, params_mira_sd, "MIRA", "red")

# Maxent batch
me_lbfgs = mebc.MaxEntBatch(regularizer=1.0)
params_meb_sd = me_lbfgs.train(sd.train_X, sd.train_y)
y_pred_train = me_lbfgs.test(sd.train_X, params_meb_sd)
acc_train = me_lbfgs.evaluate(sd.train_y, y_pred_train)
y_pred_test = me_lbfgs.test(sd.test_X, params_meb_sd)
acc_test = me_lbfgs.evaluate(sd.test_y, y_pred_test)
print(
    "Max-Ent batch Simple Dataset Accuracy train: %f test: %f"
    % (acc_train, acc_test)
)
fig, axis = sd.add_line(fig, axis, params_meb_sd, "Max-Ent-Batch", "orange")

# Support Vector Machine
import lxmls.classifiers.svm as svmc

svm = svmc.SVM()
svm.regularizer = 1.0  # This is lambda
params_svm_sd = svm.train(sd.train_X, sd.train_y)
y_pred_train = svm.test(sd.train_X, params_svm_sd)
acc_train = svm.evaluate(sd.train_y, y_pred_train)
y_pred_test = svm.test(sd.test_X, params_svm_sd)
acc_test = svm.evaluate(sd.test_y, y_pred_test)
print(
    "SVM Online Simple Dataset Accuracy train: %f test: %f"
    % (acc_train, acc_test)
)
fig, axis = sd.add_line(fig, axis, params_svm_sd, "SVM", "green")

plt.show()
