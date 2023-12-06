import keras


def make_model(out_units: int) -> keras.models.Sequential:
    return keras.models.Sequential(
        [
            keras.layers.Conv2D(filters=8, kernel_size=(5, 5), activation="relu"),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu"),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu"),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(units=32, activation="relu"),
            keras.layers.Dense(units=out_units, activation="softmax"),
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
