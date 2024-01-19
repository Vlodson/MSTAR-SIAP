import matplotlib.pyplot as plt
from keras.callbacks import History


def plot_model_history(model_history: History) -> None:
    epochs = range(1, model_history.params["epochs"] + 1)
    metrics = [metric for metric in model_history.history if "val" not in metric]

    for ctr, metric in enumerate(metrics):
        # this maybe needs to sit after plt title
        plt.figure(ctr)  # ctr is here to make each plot window have a unique id
        plt.plot(
            epochs, model_history.history[metric], "r-", label=f"Training {metric}"
        )
        plt.plot(
            epochs,
            model_history.history[f"val_{metric}"],
            "b-",
            label=f"Validation {metric}",
        )
        plt.legend()
        plt.title(f"Training and validation {metric}")

    plt.show()
