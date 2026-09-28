import io

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


st.set_page_config(page_title="MNIST Digit Recognition", page_icon="🔢", layout="wide")


@st.cache_data
def load_mnist():
    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0
    train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes=10)
    return (
        train_images,
        train_labels,
        train_labels_one_hot,
        test_images,
        test_labels,
        test_labels_one_hot,
    )


@st.cache_resource(show_spinner="Training the neural network on MNIST...")
def train_model(epochs, batch_size, learning_rate):
    (
        train_images,
        train_labels,
        train_labels_one_hot,
        test_images,
        test_labels,
        test_labels_one_hot,
    ) = load_mnist()

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_images,
        train_labels_one_hot,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    validation_size = int(len(train_images) * 0.1)
    validation_probabilities = model.predict(
        train_images[-validation_size:], verbose=0
    )
    temperature = fit_temperature(
        validation_probabilities, train_labels[-validation_size:]
    )
    test_loss, test_accuracy = model.evaluate(
        test_images, test_labels_one_hot, verbose=0
    )
    test_probabilities = model.predict(test_images, verbose=0)
    calibrated_test_probabilities = apply_temperature(
        test_probabilities, temperature
    )
    test_predictions = calibrated_test_probabilities.argmax(axis=1)
    return (
        model,
        history.history,
        float(test_loss),
        float(test_accuracy),
        test_images,
        test_labels,
        test_predictions,
        train_labels,
        temperature,
        expected_calibration_error(test_probabilities, test_labels),
        expected_calibration_error(calibrated_test_probabilities, test_labels),
    )


def apply_temperature(probabilities, temperature):
    logits = np.log(np.clip(probabilities, 1e-7, 1.0)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=-1, keepdims=True)


def fit_temperature(validation_probabilities, validation_labels):
    temperatures = np.linspace(0.5, 3.0, 101)
    best_temperature = 1.0
    best_loss = float("inf")
    row_indices = np.arange(len(validation_labels))

    for temperature in temperatures:
        probabilities = apply_temperature(validation_probabilities, temperature)
        loss = -np.log(
            np.clip(probabilities[row_indices, validation_labels], 1e-7, 1.0)
        ).mean()
        if loss < best_loss:
            best_temperature = float(temperature)
            best_loss = float(loss)

    return best_temperature


def expected_calibration_error(probabilities, labels, bin_count=10):
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    bin_edges = np.linspace(0.0, 1.0, bin_count + 1)
    error = 0.0

    for bin_index in range(bin_count):
        in_bin = (confidences >= bin_edges[bin_index]) & (
            confidences <= bin_edges[bin_index + 1]
            if bin_index == bin_count - 1
            else confidences < bin_edges[bin_index + 1]
        )
        if in_bin.any():
            error += in_bin.mean() * abs(
                correct[in_bin].mean() - confidences[in_bin].mean()
            )

    return float(error)


def prepare_uploaded_image(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    pixels = np.asarray(image)

    border = np.concatenate(
        (pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1])
    )
    if border.mean() > 127:
        image = ImageOps.invert(image)
        pixels = np.asarray(image)

    foreground = pixels > 32
    if not foreground.any():
        return np.zeros((28, 28), dtype="float32")

    rows, columns = np.where(foreground)
    bounds = (columns.min(), rows.min(), columns.max() + 1, rows.max() + 1)
    digit = image.crop(bounds)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    canvas.paste(
        digit,
        ((28 - digit.width) // 2, (28 - digit.height) // 2),
    )
    return np.asarray(canvas, dtype="float32") / 255.0


def show_prediction(probabilities):
    predicted_digit = int(probabilities.argmax())

    st.subheader(f"Prediction: {predicted_digit}")
    st.metric("Confidence", f"{probabilities[predicted_digit]:.1%}")
    probability_frame = pd.DataFrame(
        {"Probability": probabilities}, index=[str(digit) for digit in range(10)]
    )
    st.bar_chart(probability_frame, horizontal=True, height=300)


st.title("MNIST Digit Recognition")
st.write(
    "Train a neural network on handwritten digits, inspect its test performance, "
    "then try your own image."
)

with st.sidebar:
    st.header("Training")
    with st.form("training_settings"):
        epochs = st.slider("Epochs", min_value=1, max_value=20, value=5)
        batch_size = st.select_slider(
            "Batch size", options=[32, 64, 128, 256], value=128
        )
        learning_rate = st.select_slider(
            "Learning rate", options=[0.01, 0.03, 0.05, 0.1], value=0.05
        )
        train_clicked = st.form_submit_button("Train model", type="primary")

if train_clicked:
    st.session_state.training_result = train_model(
        epochs, batch_size, learning_rate
    )

training_result = st.session_state.get("training_result")
if training_result is not None and len(training_result) != 11:
    del st.session_state.training_result
    training_result = None

recognize_tab, evaluation_tab, method_tab = st.tabs(
    ["Recognize", "Evaluation", "Method"]
)

with recognize_tab:
    training_result = st.session_state.get("training_result")
    if training_result is None:
        st.info("Choose training settings in the sidebar and train the model to begin.")
    else:
        model = training_result[0]
        input_type = st.radio(
            "Image source",
            ["Draw a digit", "Upload an image", "MNIST test image"],
            horizontal=True,
        )

        if input_type == "Draw a digit":
            canvas = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=st.slider("Brush size", 8, 24, 16),
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                display_toolbar=True,
                key="digit_canvas",
            )
            if canvas.image_data is not None:
                drawn_image = Image.fromarray(canvas.image_data.astype("uint8"))
                prepared_image = prepare_uploaded_image(drawn_image)
                if prepared_image.any():
                    preview, result = st.columns([1, 2])
                    with preview:
                        st.image(
                            (prepared_image * 255).astype("uint8"),
                            caption="28 x 28 model input",
                            width=180,
                        )
                    with result:
                        probabilities = apply_temperature(
                            model.predict(prepared_image[np.newaxis, ...], verbose=0)[0],
                            training_result[8],
                        )
                        show_prediction(probabilities)
                else:
                    st.info("Draw a digit on the canvas to get a prediction.")
        elif input_type == "Upload an image":
            uploaded_files = st.file_uploader(
                "Choose digit images",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                prepared_images = [
                    prepare_uploaded_image(
                        Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
                    )
                    for uploaded_file in uploaded_files
                ]
                probabilities = apply_temperature(
                    model.predict(np.stack(prepared_images), verbose=0),
                    training_result[8],
                )
                if len(uploaded_files) > 1:
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "File": [file.name for file in uploaded_files],
                                "Predicted digit": probabilities.argmax(axis=1),
                                "Confidence": [
                                    f"{confidence:.1%}"
                                    for confidence in probabilities.max(axis=1)
                                ],
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    selected_index = st.selectbox(
                        "Inspect image",
                        options=range(len(uploaded_files)),
                        format_func=lambda index: uploaded_files[index].name,
                    )
                else:
                    selected_index = 0

                preview, result = st.columns([1, 2])
                with preview:
                    st.image(
                        (prepared_images[selected_index] * 255).astype("uint8"),
                        caption="28 x 28 model input",
                        width=180,
                    )
                with result:
                    show_prediction(probabilities[selected_index])
        else:
            test_images, test_labels = training_result[4], training_result[5]
            sample_index = st.slider(
                "Test image", min_value=0, max_value=len(test_images) - 1, value=0
            )
            sample_image = test_images[sample_index]
            preview, result = st.columns([1, 2])
            with preview:
                st.image(
                    (sample_image * 255).astype("uint8"),
                    caption=f"True label: {test_labels[sample_index]}",
                    width=180,
                )
            with result:
                probabilities = apply_temperature(
                    model.predict(sample_image[np.newaxis, ...], verbose=0)[0],
                    training_result[8],
                )
                show_prediction(probabilities)

with evaluation_tab:
    training_result = st.session_state.get("training_result")
    if training_result is None:
        st.info("Train the model to see its evaluation on the held-out test set.")
    else:
        (
            _,
            history,
            test_loss,
            test_accuracy,
            _,
            test_labels,
            test_predictions,
            train_labels,
            temperature,
            raw_calibration_error,
            calibrated_calibration_error,
        ) = training_result
        accuracy_column, loss_column, temp_column, raw_ece_column, calibrated_ece_column = (
            st.columns(5)
        )
        accuracy_column.metric("Test accuracy", f"{test_accuracy:.2%}")
        loss_column.metric("Test cross-entropy", f"{test_loss:.4f}")
        temp_column.metric("Calibration temperature", f"{temperature:.2f}")
        raw_ece_column.metric("Raw test ECE", f"{raw_calibration_error:.2%}")
        calibrated_ece_column.metric(
            "Calibrated test ECE", f"{calibrated_calibration_error:.2%}"
        )

        chart_column, class_column = st.columns(2)
        with chart_column:
            st.subheader("Training history")
            history_frame = pd.DataFrame(
                {
                    "Training accuracy": history["accuracy"],
                    "Validation accuracy": history["val_accuracy"],
                    "Training loss": history["loss"],
                    "Validation loss": history["val_loss"],
                },
                index=range(1, len(history["accuracy"]) + 1),
            )
            st.line_chart(history_frame)

        with class_column:
            st.subheader("Training-set class balance")
            class_counts = np.bincount(train_labels, minlength=10)
            st.bar_chart(
                pd.DataFrame(
                    {"Images": class_counts},
                    index=[str(digit) for digit in range(10)],
                )
            )

        confusion = np.zeros((10, 10), dtype=int)
        np.add.at(confusion, (test_labels, test_predictions), 1)
        st.subheader("Confusion matrix")
        st.caption("Rows are true digits; columns are predicted digits.")
        st.dataframe(
            pd.DataFrame(
                confusion,
                index=[f"True {digit}" for digit in range(10)],
                columns=[f"Pred {digit}" for digit in range(10)],
            ),
            use_container_width=True,
        )

with method_tab:
    st.subheader("End-to-end workflow")
    st.markdown(
        """
        ### 1. Data Preprocessing
        MNIST supplies 60,000 training images and 10,000 test images. Each 28 x 28
        grayscale image is converted to `float32` and normalized from pixel values
        in $[0, 255]$ to $[0, 1]$. Digit labels are converted to 10-class one-hot
        vectors. Ten percent of the training data is reserved for validation.

        ### 2. Model Development
        The classifier accepts a 28 x 28 image, flattens it, then applies a 256-unit
        ReLU layer, dropout, and a 128-unit ReLU layer. A 10-unit softmax output
        layer produces probabilities for digits 0 through 9.

        ### 3. Training
        Choose the epoch count, batch size, and learning rate in the Training
        sidebar, then select **Train model**. The network is trained with
        categorical cross-entropy and stochastic gradient descent (SGD); accuracy
        is tracked for both training and validation data.

        ### 4. Evaluation
        After training, the model is evaluated against the separate 10,000-image
        test set. The Evaluation tab reports test loss and accuracy, training
        history, class balance, and a confusion matrix. Temperature scaling is
        fitted on validation predictions, and expected calibration error is
        reported for raw and calibrated test probabilities. These held-out results
        measure how well the model performs on unseen examples.

        ### 5. Prediction
        In the Recognize tab, draw a digit, upload one or more images, or inspect a
        test example. Inputs are converted to grayscale, normalized, and centered
        on a 28 x 28 canvas. The model scores all ten classes; the class with the
        highest probability is shown with its confidence. Uploaded batches return
        a prediction and confidence for every image.
        """
    )