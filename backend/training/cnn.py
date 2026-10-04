from tensorflow.keras import Sequential, layers


def build_cnn():
    model = Sequential([
        layers.Input(shape=(254, 254, 3)),

        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),

        layers.Conv2D(16, 3, padding="same", activation="relu"),

        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    build_cnn().summary()