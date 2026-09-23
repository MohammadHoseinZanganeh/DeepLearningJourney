import random
import numpy as np
import matplotlib.pyplot as plt


def read_csv_dataset(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a labeled dataset from a CSV file.

    Each row is expected to have the label in the first column,
    followed by feature values in the remaining columns.

    Args:
        file_path: Path to the CSV file containing the dataset.

    Returns:
        A tuple of (features, labels) where:
            - features: np.ndarray of shape (N, D) with dtype float32
            - labels: np.ndarray of shape (N,) with dtype float32
    """
    records = []  # placeholder for data

    with open(file_path, "r") as csv_file:
        for row in csv_file:
            # remove trailing newline, split by comma, convert to float32 array
            # .strip() removes \n at the end: "5,0,0,12,…,0\n" → "5,0,0,12,…,0"
            values = np.array(row.strip().split(","), dtype=np.float32)
            records.append(values)

    # stack all rows into a single 2D array of shape (N, 1+D)
    # where column 0 is the label and columns 1: are the features
    dataset = np.asarray(records)

    features = dataset[:, 1:]  # pixel values
    labels   = dataset[:, 0].astype(np.int32)   # digit class (0–9)

    return features, labels


def plot_random_samples(
    X: np.ndarray,
    Y: np.ndarray,
    samples_per_class: int = 5,
    num_classes: int = 10,
    image_shape: tuple[int, int] = (28, 28),
) -> None:
    """Display random samples from each class in a grid layout.

    Creates a grid of (num_classes x samples_per_class) where each row
    belongs to one digit class and each column is a random sample.

    Args:
        X: Feature matrix of shape (N, D) containing flattened images.
        Y: Label array of shape (N,) containing digit classes (0-9).
        samples_per_class: Number of random samples to show per class.
        num_classes: Total number of digit classes.
        image_shape: The (height, width) to reshape each sample into.
    """
    _, axes = plt.subplots(num_classes, samples_per_class, figsize=(10, 20))

    for digit in range(num_classes):

        # get all the indexes where the label equals this digit
        class_indices = np.where(Y == digit)[0]

        # pick 5 random ones
        random_indices = np.random.choice(class_indices, size=samples_per_class, replace=False)

        for col, idx in enumerate(random_indices):
            ax = axes[digit, col]

            # each image is stored as a flat array, need to reshape it to 28x28
            image = X[idx].reshape(image_shape)

            ax.imshow(image, cmap="gray")
            ax.axis("off")

            if col == 0:
                ax.set_title(f"Digit: {digit}", fontsize=10)

    plt.suptitle("5 Random Samples per Class", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_class_distribution(Y: np.ndarray, num_classes: int = 10) -> None:
    """Plot a histogram of class distribution and report imbalance.

    Visualizes how many samples exist per digit class and prints
    a simple imbalance ratio to detect dataset bias.

    Args:
        Y: Label array of shape (N,) containing digit classes (0-9).
        num_classes: Total number of digit classes.
    """
    _, ax = plt.subplots(figsize=(10, 5))

    # count how many samples we have for each digit
    unique, counts = np.unique(Y, return_counts=True)

    ax.bar(unique, counts, color="steelblue", edgecolor="black")
    ax.set_xticks(range(num_classes))
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution in Training Set")

    # show the exact number on top of each bar
    for x, count in zip(unique, counts):
        ax.text(x, count + 50, str(count), ha="center", fontsize=9)

    plt.tight_layout()
    plt.show()

    # check if the dataset is balanced or not
    most_common  = unique[counts.argmax()]
    least_common = unique[counts.argmin()]

    print(f"Max samples : {counts.max()} (digit {most_common})")
    print(f"Min samples : {counts.min()} (digit {least_common})")
    print(f"Imbalance ratio: {counts.max() / counts.min():.2f}x")


def split_validation(
    features: np.ndarray, labels: np.ndarray, val_ratio: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits dataset into train and validation sets in a stratified way.

    Args:
        features: np.ndarray of input samples (pixel values).
        labels: np.ndarray of class labels corresponding to each sample.
        val_ratio: Fraction of data to use for validation. Defaults to 0.1.

    Returns:
        A tuple of (train_features, val_features, train_labels, val_labels).
    """
    # find all unique classes (0 to 9)
    classes = np.unique(labels)

    val_features,   val_labels   = [], []
    train_features, train_labels = [], []

    # for each class, take val_ratio% for validation
    for cls in classes:

        # get all indexes that belong to this class
        cls_indexes = np.where(labels == cls)[0].tolist()

        # shuffle the indexes randomly
        random.shuffle(cls_indexes)

        # calculate how many samples go to validation
        val_count = int(len(cls_indexes) * val_ratio)

        # split indexes into val and train
        val_idx   = cls_indexes[:val_count]
        train_idx = cls_indexes[val_count:]

        # add samples to their respective lists
        val_features.extend(features[val_idx])
        val_labels.extend(labels[val_idx])
        train_features.extend(features[train_idx])
        train_labels.extend(labels[train_idx])

    return (
        np.array(train_features),
        np.array(val_features),
        np.array(train_labels),
        np.array(val_labels),
    )


# This block only runs when the file is executed directly
# Nothing here is triggered during import
if __name__ == "__main__":

    X_train, Y_train = read_csv_dataset("Part2/data/dataset/mnist_train.csv")
    X_test,  Y_test  = read_csv_dataset("Part2/data/dataset/mnist_test.csv")

    print(f"The shape of the training set: {X_train.shape}")
    print(f"The shape of the test set: {X_test.shape}")
    print(f"The shape of the label training set: {Y_train.shape}")
    print(f"The shape of the label test set: {Y_test.shape}")

    plot_random_samples(X_train, Y_train)
    plot_class_distribution(Y_train)

    X_train_final, X_val, Y_train_final, Y_val = split_validation(X_train, Y_train)

    print(f"Train size:      {len(X_train_final)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size:       {len(X_test)}")
