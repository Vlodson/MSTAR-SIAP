import keras


def make_model(out_units: int) -> keras.models.Sequential:
    return keras.models.Sequential(
        [
            keras.layers.Dense(units=2048, activation="relu"),
            keras.layers.Dense(units=512, activation="relu"),
            keras.layers.Dense(units=out_units, activation="softmax"),
        ]
    )


def compile_model(model: keras.models.Sequential) -> keras.models.Sequential:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-1),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            "categorical_accuracy",
            keras.metrics.F1Score(average="macro"),
        ],
    )

    return model
