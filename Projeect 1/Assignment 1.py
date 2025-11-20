import numpy as np

# -------------------------- part 1 ---------------------

def initialize_parameters(layer_dims):
    initialized = {}
    for layer in range(1, len(layer_dims)):
        shape = (layer_dims[layer], layer_dims[layer - 1])
        w_rand = np.random.randn(*shape)
        b_rand = np.zeros((layer_dims[layer], 1))
        initialized['W' + str(layer)] = w_rand
        initialized['b' + str(layer)] = b_rand
    return initialized


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)
    return Z, linear_cache


def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    activation_cache = Z
    return A, activation_cache

def relu(Z):
    A = np.maximum(0, Z)
    activation_cache = Z
    return A, activation_cache

def linear_activation_forward(A_prev, W, B, activation):
    Z, linear_cache = linear_forward(A_prev, W, B)
    if activation == 'softmax':
        A, activation_cache = softmax(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)
    cache = {
        "linear_cache": linear_cache,
        "activation_cache": activation_cache
    }
    return A, cache


def apply_batchnorm(A):
    return None

def l_model_forward(X,parameters,use_batch_norm):
    caches = []
    A = X
    length = len(parameters)//2

    for layer in range(1,length):
        W = parameters[f"W{layer}"]
        b = parameters[f"b{layer}"]

        A , cache = linear_activation_forward(A,W,b,"relu")
        if use_batch_norm:
            A = apply_batchnorm(A)
        caches.append(cache)
    
    WL = parameters[f"W{length}"]
    bL = parameters[f"b{length}"]

    AL , cache = linear_activation_forward(A,WL,bL,"softmax")
    caches.append(cache)

    return AL, caches

def apply_batchnorm(A):
    epsilon = 1e-12
    mean = np.mean(A, axis=1, keepdims=True)
    var = np.var(A, axis=1, keepdims=True)
    NA = (A - mean) / np.sqrt(var + epsilon)
    return NA


def compute_cost(AL, Y):
    m = Y.shape[1]
    AL = np.clip(AL, 1e-12, 1.0)
    cost = - (1 / m) * np.sum(Y * np.log(AL))
    return cost


# -------------------------- part 2 ---------------------

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax_backward(dA, activation_cache):
    return dA  # dZ = AL - Y is computed directly in l_model_backward

def linear_activation_backward(dA, cache, activation):
    linear_cache = cache["linear_cache"]
    activation_cache = cache["activation_cache"]
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def l_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dZL = AL - Y
    linear_cache = caches[L - 1]["linear_cache"]
    dA_prev, dW, db = linear_backward(dZL, linear_cache)
    grads["dA" + str(L - 1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return parameters

# -------------------------- part 3A ---------------------

def get_batches(X, Y, batch_size):
    m = X.shape[1] 
    for i in range(0, m, batch_size):
        X_batch = X[:, i:i+batch_size]
        Y_batch = Y[:, i:i+batch_size]
        yield X_batch, Y_batch


def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    use_batch_norm = False
    parameters = initialize_parameters(layers_dims)
    costs = []
    m = X.shape[1]
    iteration_count = 0
    
    for epoch in range(num_iterations):
        epoch_cost = 0
        num_batches = 0
        
        # Process each batch
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            
            X_batch = X[:, start:end]
            Y_batch = Y[:, start:end]
            
            # Forward propagation
            AL, caches = l_model_forward(X_batch, parameters, use_batch_norm)
            
            # Compute cost
            cost = compute_cost(AL, Y_batch)
            epoch_cost += cost
            num_batches += 1
            
            # Backward propagation
            grads = l_model_backward(AL, Y_batch, caches)
            
            # Update parameters
            parameters = update_parameters(parameters, grads, learning_rate)
            
            iteration_count += 1
            
            # Save cost every 100 iterations
            if iteration_count % 100 == 0:
                costs.append(cost)
                print(f"Iteration {iteration_count} | Cost: {cost:.4f}")
        
        # Average cost for the epoch
        avg_epoch_cost = epoch_cost / num_batches
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Average Cost: {avg_epoch_cost:.4f}")
    
    return parameters, costs

# -------------------------- part 3B ---------------------

def predict(X, Y, parameters):
    use_batch_norm=False
    AL, _ = l_model_forward(X, parameters, use_batch_norm)   
    y_pred = np.argmax(AL, axis=0)
    if Y.ndim > 1:
        y_true = np.argmax(Y, axis=0)
    else:
        y_true = Y
    
    accuracy = np.mean(y_pred == y_true) * 100
    
    return accuracy


# -------------------------- part 4A ---------------------
import os
import urllib.request
import gzip
import numpy as np

def download_mnist():
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    for filename in files:
        if not os.path.exists(filename):
            print(f"Downloading {filename} ...")
            urllib.request.urlretrieve(base_url + filename, filename)
            print("Done")
        else:
            print(f"{filename} already exists")

def load_idx(path):
    """Loads MNIST IDX format file."""
    with gzip.open(path, 'rb') as f:
        data = f.read()

    magic = int.from_bytes(data[0:4], "big")

    if magic == 2051:  # images
        num_images = int.from_bytes(data[4:8], "big")
        rows = int.from_bytes(data[8:12], "big")
        cols = int.from_bytes(data[12:16], "big")
        images = np.frombuffer(data, dtype=np.uint8, offset=16)
        images = images.reshape(num_images, rows * cols).T
        return images / 255.0

    elif magic == 2049:  # labels
        labels = np.frombuffer(data, dtype=np.uint8, offset=8)
        return labels

    else:
        raise ValueError("Unknown MNIST IDX file format")

def one_hot(y, num_classes=10):
    m = y.shape[0]
    Y = np.zeros((num_classes, m))
    Y[y, np.arange(m)] = 1
    return Y

def load_and_check_mnist():
    print("Downloading MNIST (if missing)...")
    download_mnist()

    print("\nLoading MNIST from local files...")
    X_train = load_idx("train-images-idx3-ubyte.gz")
    Y_train_raw = load_idx("train-labels-idx1-ubyte.gz")
    X_test = load_idx("t10k-images-idx3-ubyte.gz")
    Y_test_raw = load_idx("t10k-labels-idx1-ubyte.gz")

    Y_train = one_hot(Y_train_raw)
    Y_test = one_hot(Y_test_raw)

    print("\n=== MNIST Loaded Successfully ===")
    print("X_train shape:", X_train.shape)  # (784, 60000)
    print("Y_train shape:", Y_train.shape)  # (10, 60000)
    print("X_test shape:", X_test.shape)    # (784, 10000)
    print("Y_test shape:", Y_test.shape)    # (10, 10000)

    return X_train, Y_train, X_test, Y_test


# -----------------4B -----------------------

def train_val_split(X, Y, val_ratio=0.2):
    m = X.shape[1]
    idx = np.random.permutation(m)
    split = int(m * (1 - val_ratio))

    train_idx = idx[:split]
    val_idx = idx[split:]

    X_train = X[:, train_idx]
    Y_train = Y[:, train_idx]

    X_val = X[:, val_idx]
    Y_val = Y[:, val_idx]

    return X_train, Y_train, X_val, Y_val


# -----------------4B -----------------------

def train_mnist_early_stopping(
    X_train, Y_train, X_val, Y_val,
    layers_dims,
    learning_rate=0.009,
    batch_size=64,
    patience_steps=100,
    use_batchnorm=False
):
    """
    MNIST training with early stopping.
    Stops when validation accuracy does not improve for <patience_steps>.
    """

    parameters = initialize_parameters(layers_dims)

    best_val_acc = 0
    steps_without_improvement = 0
    costs = []
    step = 0
    epoch = 0

    m = X_train.shape[1]

    print("\n=== Training Started ===")

    while True:  # loop until early stop
        # shuffle dataset each epoch
        perm = np.random.permutation(m)
        X_train = X_train[:, perm]
        Y_train = Y_train[:, perm]

        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            X_batch = X_train[:, start:end]
            Y_batch = Y_train[:, start:end]

            # Forward
            AL, caches = l_model_forward(X_batch, parameters, use_batchnorm)

            # Cost
            cost = compute_cost(AL, Y_batch)

            # Backward
            grads = l_model_backward(AL, Y_batch, caches)

            # Update
            parameters = update_parameters(parameters, grads, learning_rate)

            # Count training steps
            step += 1

            # Save cost every 100 steps
            if step % 100 == 0:
                costs.append((step, cost))
                print(f"Step {step} | Cost: {cost:.4f}")

            # Check validation accuracy
            val_acc = predict(X_val, Y_val, parameters)

            # Improvement?
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            # Early stopping condition
            if steps_without_improvement >= patience_steps:
                print("\n=== Early Stopping Triggered ===")
                print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
                print(f"Total Steps: {step}")
                print(f"Total Epochs: {epoch}")
                print("==============================\n")
                return parameters, costs, step, epoch, best_val_acc

        epoch += 1



# -------------------------- part 4C ---------------------

def run_mnist_experiment(use_batchnorm):
    print(f"\n=== MNIST Experiment (BatchNorm {use_batchnorm}) ===")

    # Load MNIST from your working loader
    X_train_full, Y_train_full, X_test, Y_test = load_and_check_mnist()

    # Split train → train + validation
    X_train, Y_train, X_val, Y_val = train_val_split(X_train_full, Y_train_full)

    layers_dims = [784, 20, 7, 5, 10]

    parameters, costs, steps, epochs, best_val_acc = train_mnist_early_stopping(
        X_train, Y_train,
        X_val, Y_val,
        layers_dims,
        learning_rate=0.009,
        batch_size=64,
        patience_steps=100,
        use_batchnorm= use_batchnorm
    )

    # Final accuracy scores
    train_acc = predict(X_train, Y_train, parameters)
    val_acc = predict(X_val, Y_val, parameters)
    test_acc = predict(X_test, Y_test, parameters)

    print("=== FINAL ACCURACIES ===")
    print(f"Train Accuracy:      {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Accuracy:       {test_acc:.2f}%")

    return parameters, costs





run_mnist_experiment(False)
run_mnist_experiment(True)

