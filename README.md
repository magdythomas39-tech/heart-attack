import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder # Added OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random

df = pd.read_csv("/content/heart.csv")
target_col = df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

le = LabelEncoder()
y = le.fit_transform(y)

# Separate numerical and categorical columns
numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(exclude=np.number).columns

# Impute numerical columns
imputer_numeric = SimpleImputer(strategy='median')
X_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(X[numeric_cols]), columns=numeric_cols)

# Impute categorical columns
imputer_categorical = SimpleImputer(strategy='most_frequent')
X_categorical_imputed = pd.DataFrame(imputer_categorical.fit_transform(X[categorical_cols]), columns=categorical_cols)

# One-hot encode categorical columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_categorical_ohe = pd.DataFrame(ohe.fit_transform(X_categorical_imputed), columns=ohe.get_feature_names_out(categorical_cols))

# Concatenate processed numerical and categorical features
X_processed = pd.concat([X_numeric_imputed, X_categorical_ohe], axis=1)

# Scale the features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_processed), columns=X_processed.columns)

# Identify and remove classes with only one member in y
# Convert y to a pandas Series to use value_counts()
y_series = pd.Series(y)
class_counts = y_series.value_counts()
singleton_classes = class_counts[class_counts == 1].index

# Get indices of rows to keep (where y is not a singleton class)
rows_to_keep = y_series[~y_series.isin(singleton_classes)].index

# Filter X_scaled and y
X_filtered = X_scaled.iloc[rows_to_keep]
y_filtered = y_series.iloc[rows_to_keep].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
)



svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

orig_acc  = accuracy_score(y_test, y_pred)
orig_prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
orig_rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
orig_f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("=== ORIGINAL SVM RESULTS ===")
print(f"Accuracy:  {orig_acc:.4f}")
print(f"Precision: {orig_prec:.4f}")
print(f"Recall:    {orig_rec:.4f}")
print(f"F1 Score:  {orig_f1:.4f}")
print(confusion_matrix(y_test, y_pred))

n_features = X_train.shape[1]
POP_SIZE = 20
GENERATIONS = 10
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.05

def fitness(individual):
    selected = [i for i in range(n_features) if individual[i] == 1]
    if len(selected) == 0:
        return 0
    X_tr = X_train.iloc[:, selected]
    X_te = X_test.iloc[:, selected]
    model = SVC(kernel="rbf", C=1.0, gamma="scale")
    model.fit(X_tr, y_train)
    pred = model.predict(X_te)
    return accuracy_score(y_test, pred)

population = [
    [random.randint(0, 1) for _ in range(n_features)]
    for _ in range(POP_SIZE)
]

for gen in range(GENERATIONS):
    fits = [fitness(ind) for ind in population]
    print(f"Gen {gen+1}: best={max(fits):.4f}")

    new_pop = []
    for _ in range(POP_SIZE):
        a, b = random.sample(range(POP_SIZE), 2)
        winner = population[a] if fits[a] > fits[b] else population[b]
        new_pop.append(winner.copy())

    for i in range(0, POP_SIZE, 2):
        if random.random() < CROSSOVER_RATE:
            point = random.randint(1, n_features - 1)
            new_pop[i][:point], new_pop[i+1][:point] = \
                new_pop[i+1][:point], new_pop[i][:point]

    for ind in new_pop:
        for j in range(n_features):
            if random.random() < MUTATION_RATE:
                ind[j] = 1 - ind[j]

    population = new_pop

fitnesses = [fitness(ind) for ind in population]
best_individual = population[np.argmax(fitnesses)]
selected_features = [
    X_train.columns[i] for i in range(n_features) if best_individual[i] == 1
]

print("\n=== SELECTED FEATURES BY GA ===")
print(f"Number of selected features: {len(selected_features)}")
print(selected_features)

if len(selected_features) > 0:
    X_train_red = X_train[selected_features]
    X_test_red  = X_test[selected_features]

    svm2 = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm2.fit(X_train_red, y_train)
    y_pred2 = svm2.predict(X_test_red)

    red_acc  = accuracy_score(y_test, y_pred2)
    red_prec = precision_score(y_test, y_pred2, average='weighted', zero_division=0)
    red_rec  = recall_score(y_test, y_pred2, average='weighted', zero_division=0)
    red_f1   = f1_score(y_test, y_pred2, average='weighted', zero_division=0)

    print("\n=== GA-REDUCED SVM RESULTS ===")
    print(f"Accuracy:  {red_acc:.4f}")
    print(f"Precision: {red_prec:.4f}")
    print(f"Recall:    {red_rec:.4f}")
    print(f"F1 Score:  {red_f1:.4f}")
    print(confusion_matrix(y_test, y_pred2))

else:
    print("\nNo features were selected by GA.")


    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if len(selected_features) > 0:
    X_train_red = X_train[selected_features]
    X_test_red  = X_test[selected_features]

    svm2 = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm2.fit(X_train_red, y_train)
    y_pred2 = svm2.predict(X_test_red)

    red_acc  = accuracy_score(y_test, y_pred2)
    red_prec = precision_score(y_test, y_pred2, average='weighted', zero_division=0)
    red_rec  = recall_score(y_test, y_pred2, average='weighted', zero_division=0)
    red_f1   = f1_score(y_test, y_pred2, average='weighted', zero_division=0)

    print("\n=== GA-REDUCED SVM RESULTS ===")
    print(f"Accuracy:  {red_acc:.4f}")
    print(f"Precision: {red_prec:.4f}")
    print(f"Recall:    {red_rec:.4f}")
    print(f"F1 Score:  {red_f1:.4f}")

    cm = confusion_matrix(y_test, y_pred2)
    print(cm)

    # Pretty matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix (GA-Reduced Features)")
    plt.show()

else:
    print("\nNo features were selected by GA.")
