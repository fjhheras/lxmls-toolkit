import matplotlib.pyplot as plt
import lxmls.readers.simple_data_set as sds
import lxmls.readers.sentiment_reader as srs
import lxmls.classifiers.perceptron as percc


def run_on_simple_dataset():
    sd = sds.SimpleDataSet(
        nr_examples=100,
        g1=[[-2, -1], 1.2],
        g2=[[0, 1], 1],
        balance=0.5,
        split=[0.5, 0, 0.5],
    )

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
    colors = ["b", "g", "r", "y"]
    for i, w in enumerate(perc.params_per_round[::3]):
        fig, axis = sd.add_line(
            fig, axis, w, f"Perceptron step {i}", colors[i]
        )

    plt.show()


def run_on_amazon_dataset():
    scr = srs.SentimentCorpus("books")

    perc = percc.Perceptron()
    params_perc_scr = perc.train(scr.train_X, scr.train_y)
    y_pred_train = perc.test(scr.train_X, params_perc_scr)
    acc_train = perc.evaluate(scr.train_y, y_pred_train)
    y_pred_test = perc.test(scr.test_X, params_perc_scr)
    acc_test = perc.evaluate(scr.test_y, y_pred_test)
    print(
        "Perceptron Amazon Dataset Accuracy train: %f test: %f"
        % (acc_train, acc_test)
    )


if __name__ == "__main__":
    run_on_amazon_dataset()
    run_on_simple_dataset()
