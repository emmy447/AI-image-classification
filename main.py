import torch
import torchvision
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import torch.nn as nn
import torch.optim as optim
import pickle
import joblib

# Define transformations for the data (resize, normalize)
transformation = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing images to 224x224 for ResNet-18
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet standards
])

# Load CIFAR-10 dataset
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transformation)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transformation)


# Function to get subset of the first 500 training images and 100 test images per class
def get_subset(dataset, num_train=500, num_test=100):
    class_index = {i: [] for i in range(10)}

    # Collect indexes for each class in the dataset
    for index, (_, label) in enumerate(dataset):
        class_index[label].append(index)

    train_index = []
    test_index = []

    for label in range(10):
        train_index.extend(class_index[label][:num_train])  # First 500 images for each class in train
        test_index.extend(class_index[label][:num_test])  # First 100 images for each class in test

    return Subset(dataset, train_index), Subset(dataset, test_index)


# Create subsets for each of the training images and testing images
train_subset, test_subset = get_subset(train_set, num_train=500, num_test=100)

# Create dataloader for batching
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

# Sample output of one batch (checking the first batch of images and their labels)
data_iterator = iter(train_loader)
images, labels = next(data_iterator)

# Initialize ResNet-18 model pre-trained on ImageNet
resnet = models.resnet18(pretrained=True)
resnet = resnet.eval()

# Remove the final layer
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])


# Method to extract features using resnet
def extract_features(dataloader, model, device):
    features = []
    labels = []
    with torch.no_grad():  # removes gradient calculation to save memory
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)  # flats out the output to 512x1
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    features = np.concatenate(features, axis=0)  # Combine all batches
    labels = np.concatenate(labels, axis=0)  # Combine all labels
    return features, labels


# Assuming you're using GPU, or else changes cuda to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

# Extract features from training set and test set
train_features, train_labels = extract_features(train_loader, resnet, device)
test_features, test_labels = extract_features(test_loader, resnet, device)

# Standardize the feature vectors for PCA
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)  # Fit on training data
test_features = scaler.transform(test_features)  # Only transforms test data

# Apply PCA to reduce from 512 dimensions to 50
pca = PCA(n_components=50)
train_features_pca = pca.fit_transform(train_features)
test_features_pca = pca.transform(test_features)


# Naive Bayes implementation
class GaussianNaiveBayes:
    def __init__(self):
        self.mean = None
        self.var = None
        self.classes = None
        self.class_prior = None

    def fit(self, x, y):
        self.classes = np.unique(y)
        num_features = x.shape[1]
        self.mean = np.zeros((len(self.classes), num_features))
        self.var = np.zeros((len(self.classes), num_features))
        self.class_prior = np.zeros(len(self.classes))

        # Calculate mean, variance and prior probabilities for each class
        for c in self.classes:
            x_c = x[y == c]
            if x_c.ndim == 1:
                x_c = x_c.reshape(1, -1)  # Convert to 2D array if 1D array

            # Ensure the dimensions are consistent (all 2d arrays)
            if x_c.ndim == 2:
                self.mean[c] = x_c.mean(axis=0)  # Mean of each feature per class
                self.var[c] = x_c.var(axis=0)  # Variance of each feature per class
                self.class_prior[c] = x_c.shape[0] / float(x.shape[0])  # Prior probability of each class
            else:
                print(f"Unexpected shape for x_c: {x_c.shape}") # Handles if the array is not 2D

    def predict(self, x):
        posteriors = []
        # go through all the classes
        for c in self.classes:
            # finding the prior probability of the class
            prior = np.log(self.class_prior[c])

            # finding the likelihood of every sample in x
            likelihood = np.sum(np.log(self._gaussian_pdf(x, c)), axis=1)  # Sum along features (axis=1)

            # calculate the posterior (log of prior + likelihood)
            posterior = prior + likelihood

            # Store the posterior for each class
            posteriors.append(posterior)

        # posteriors to a numpy array
        posteriors = np.array(posteriors).T  # Transpose so samples are rows and classes are columns

        return self.classes[np.argmax(posteriors, axis=1)]

    def _gaussian_pdf(self, x, c):
        mean = self.mean[c]
        var = self.var[c]
        eps = 1e-6  # To handle division by 0
        return (1 / np.sqrt(2 * np.pi * var + eps)) * np.exp(-0.5 * ((x - mean) ** 2) / (var + eps))


# test the basic naive bayes model
nb_basic = GaussianNaiveBayes()
nb_basic.fit(train_features_pca, train_labels)

# saving basic model
with open('naive_bayes_custom_model.pkl', 'wb') as f:
    pickle.dump(nb_basic, f)

# test the scikit-learn naive bayes model
gnb = GaussianNB()
gnb.fit(train_features_pca, train_labels)

# saving scikit-learn naive bayes model
joblib.dump(gnb, 'naive_bayes_sklearn_model.pkl')

# loading the models
with open('naive_bayes_custom_model.pkl', 'rb') as f:
    nb_basic_loaded = pickle.load(f)

nb_scikit_loaded = joblib.load('naive_bayes_sklearn_model.pkl')

# comparing both models
# basic naive bayes predictions
prediction_basic = nb_basic_loaded.predict(test_features_pca)

# scikit-Learn Naive bayes predictions
prediction_sk = nb_scikit_loaded.predict(test_features_pca)

# evaluation basic Naive Bayes
accuracy_basic = accuracy_score(test_labels, prediction_basic)
precision_basic, recall_basic, f1_basic, _ = precision_recall_fscore_support(test_labels, prediction_basic, average='weighted')

# evaluation Scikit-Learn Naive Bayes
accuracy_sk = accuracy_score(test_labels, prediction_sk)
precision_sk, recall_sk, f1_sk, _ = precision_recall_fscore_support(test_labels, prediction_sk, average='weighted')

# confusion matrix
matrix_basic = confusion_matrix(test_labels, prediction_basic)
matrix_sk = confusion_matrix(test_labels, prediction_sk)


def make_confusion_matrix(cm, name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


make_confusion_matrix(matrix_basic, "Basic Naive Bayes")
make_confusion_matrix(matrix_sk, "Scikit-Learn Naive Bayes")

# print the evaluation metrics
print("Basic Naive Bayes Evaluation:")
print(f"Accuracy: {accuracy_basic:.4f}")
print(f"Precision: {precision_basic:.4f}")
print(f"Recall: {recall_basic:.4f}")
print(f"F1-Score: {f1_basic:.4f}")

print("\nScikit-Learn Naive Bayes Evaluation:")
print(f"Accuracy: {accuracy_sk:.4f}")
print(f"Precision: {precision_sk:.4f}")
print(f"Recall: {recall_sk:.4f}")
print(f"F1-Score: {f1_sk:.4f}")


# Decision tree (gini coefficient)
class DecisionTree:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, x, y):
        self.tree = self.build_tree(x, y)

    def build_tree(self, x, y, depth=0):
        # stop if max depth is max
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return self.create_leaf(y)

        # recursion case-- split data and build tree
        best_split = self.find_best_split(x, y)
        left_index = x[:, best_split['feature']] <= best_split['threshold']
        right_index = ~left_index

        left_tree = self.build_tree(x[left_index], y[left_index], depth + 1)
        right_tree = self.build_tree(x[right_index], y[right_index], depth + 1)

        return {'feature': best_split['feature'], 'threshold': best_split['threshold'], 'left': left_tree,
                'right': right_tree}

    def create_leaf(self, y):
        # returns most common class
        return np.bincount(y).argmax()

    def find_best_split(self, x, y):
        # goes through all the features and finds the best to split (limits the gini impurity)
        best_split = {}
        min_gini = float('inf')
        num_features = x.shape[1]
        for feature in range(num_features):
            thresholds = np.unique(x[:, feature])
            for threshold in thresholds:
                gini = self.calculate_gini(x, y, feature, threshold)
                if gini < min_gini:
                    min_gini = gini
                    best_split = {'feature': feature, 'threshold': threshold}
        return best_split

    def calculate_gini(self, x, y, feature, threshold):
        # Calculate Gini impurity for a specific feature and threshold
        left_index = x[:, feature] <= threshold
        right_index = ~left_index
        left_classes = y[left_index]
        right_classes = y[right_index]
        left_gini = 1 - sum([(np.sum(left_classes == c) / len(left_classes)) ** 2 for c in np.unique(y)])
        right_gini = 1 - sum([(np.sum(right_classes == c) / len(right_classes)) ** 2 for c in np.unique(y)])
        left_weight = len(left_classes) / len(y)
        right_weight = len(right_classes) / len(y)
        return left_weight * left_gini + right_weight * right_gini

    def predict(self, x):
        return np.array([self._predict_single(sample, self.tree) for sample in x])

    def _predict_single(self, sample, tree):
        if isinstance(tree, dict):
            if sample[tree['feature']] <= tree['threshold']:
                return self._predict_single(sample, tree['left'])
            else:
                return self._predict_single(sample, tree['right'])
        else:
            return tree  # return the leaf class


def evaluation_model(y_true, y_pred, model_name):
    # Calculate the evaluation aspects
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    # create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # create the confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Evaluation metrics
    print(f"Metrics for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # gives the evaluation metrics
    return accuracy, precision, recall, f1


def change_dept(depths, train_features_pca, train_labels, test_features_pca, test_labels):
    results_custom_tree = []
    results_sklearn_tree = []

    for depth in depths:
        # Train basic tree
        custom_tree = DecisionTree(max_depth=depth)
        custom_tree.fit(train_features_pca, train_labels)

        # Save basic model
        with open(f'custom_decision_tree_max_depth_{depth}.pkl', 'wb') as f:
            pickle.dump(custom_tree, f)

        # Loading the basic decision tree model
        with open(f'custom_decision_tree_max_depth_{depth}.pkl', 'rb') as f:
            basic_tree_loaded = pickle.load(f)

        # Predict basic model
        y_pred_basic = basic_tree_loaded.predict(test_features_pca)

        # Train scikit-learn tree
        sklearn_tree = DecisionTree(max_depth=depth)
        sklearn_tree.fit(train_features_pca, train_labels)

        # Save scikit model
        joblib.dump(sklearn_tree, f'sklearn_decision_tree_max_depth_{depth}.pkl')

        # Loading the scikit-learn decision tree model
        sklearn_tree_loaded = joblib.load(f'sklearn_decision_tree_max_depth_{depth}.pkl')

        # Predict scikit model
        y_pred_sklearn = sklearn_tree_loaded.predict(test_features_pca)

        # Test the models
        evaluation_metrics_basic = evaluation_model(test_labels, y_pred_basic, f"Custom Decision Tree (max_depth={depth})")
        evaluation_metrics_sklearn = evaluation_model(test_labels, y_pred_sklearn, f"Scikit-learn Decision Tree (max_depth={depth})")

        # returns result
        results_custom_tree.append(evaluation_metrics_basic)
        results_sklearn_tree.append(evaluation_metrics_sklearn)

    return results_custom_tree, results_sklearn_tree


# different depths
depths_to_test = [5, 50]

# Experiment with different depths
results_custom_tree, results_sklearn_tree = change_dept(depths_to_test, train_features_pca,
                                                        train_labels, test_features_pca, test_labels)
# Multi-Layer Perceptron
# Define the Three-Layer MLP Model


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Linear(50, 512)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Linear(512, 512)
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # BatchNorm(512)
        self.fc3 = nn.Linear(hidden_size, output_size)  # Linear(512, 10)
        self.relu = nn.ReLU()  # ReLU activation

    def forward(self, x):
        x = self.relu(self.fc1(x))  # apply reLU after first layer
        x = self.relu(self.batch_norm(self.fc2(x)))  # apply batchnorm and relu after second layer
        x = self.fc3(x)  # Output layer
        return x


# function to train the MLP
def training_mlp(train_features, train_labels, model, criterion, optimizer, batch_size=64, epochs=10):
    training_dataset = torch.utils.data.TensorDataset(torch.tensor(train_features, dtype=torch.float32),
                                                      torch.tensor(train_labels, dtype=torch.long))
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for inputs, labels in training_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(training_loader):.4f}")


# function to Evaluate the MLP Model
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    # Plotting confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return accuracy, precision, recall, f1


# function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# function to load the model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")
    return model


# Prepare Data (CIFAR-10 PCA extracted features)
train_features_pca = train_features_pca
test_features_pca = test_features_pca

# Initialize and Train MLP
input_size = 50  # features after PCA
hidden_size = 512  # hidden layer size
output_size = 10  # 10 classes

mlp_model = MLP(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp_model.parameters(), lr=0.001, momentum=0.9)

# Training the MLP model
training_mlp(train_features_pca, train_labels, mlp_model, criterion, optimizer, epochs=10)

# Saving the model after training
save_model(mlp_model, 'mlp_model.pth')

# Loading the model for evaluation
mlp_model_loaded = MLP(input_size, hidden_size, output_size).to(device)
load_model(mlp_model_loaded, 'mlp_model.pth')

# Test the MLP Model
with torch.no_grad():
    outputs = mlp_model_loaded(torch.tensor(test_features_pca, dtype=torch.float32).to(device))
    _, predicted = torch.max(outputs, 1)

# metrics
accuracy, precision, recall, f1 = evaluate_model(test_labels, predicted.cpu().numpy(), "MLP Model")
print(f"\nMLP Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

'''
# CNN
class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512 * 1 * 1, 4096)
        self.relu7 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(4096, 4096)
        self.relu8 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool3(self.relu4(self.bn4(self.conv4(x))))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.pool4(self.relu6(self.bn6(self.conv6(x))))

        x = x.view(x.size(0), -1)  # flatten for fully connected layers
        x = self.drop1(self.relu7(self.fc1(x)))
        x = self.drop2(self.relu8(self.fc2(x)))
        x = self.fc3(x)
        return x


# Define transformations for training data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

training_loader = DataLoader(trainset, batch_size=64, shuffle=True)
testing_loader = DataLoader(testset, batch_size=64, shuffle=False)

# Initialize the model
model = VGG11()
criterion = nn.CrossEntropyLoss()  # initialize the loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # initialize the optimizer

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in training_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(training_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# test the model
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in testing_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

# Calculate accuracy, precision, recall, F1-score
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix: VGG11 Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the model
torch.save(model.state_dict(), 'vgg11_cifar10.pth')
print("Model saved as vgg11_cifar10.pth")

'''
