import keras


def make_model(out_units: int) -> keras.models.Sequential:
    return keras.models.Sequential(
        [
            keras.layers.Dense(units=4096, activation="relu"),
            keras.layers.Dense(units=1024, activation="relu"),
            keras.layers.Dense(units=out_units, activation=keras.activations.softmax),
        ]
    )


def compile_model(model: keras.models.Sequential) -> keras.models.Sequential:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-2),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[
            "accuracy",
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.F1Score(average="macro"),
        ],
    )

    return model
