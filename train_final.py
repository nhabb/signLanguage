"""
ASL - Final Training on Combined Normalized Data
with Multiple Models (Random Forest, MLP, Voting Classifier)
"""

# import libraries (tools we need)
import os, pickle, json              # os for folders, pickle to save model, json (not used much here)
import numpy as np                  # math operations / arrays
import pandas as pd                # handling CSV (data)

# machine learning tools
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  # models
from sklearn.neural_network import MLPClassifier  # Neural Network
from sklearn.model_selection import train_test_split  # split train/test
from sklearn.preprocessing import LabelEncoder, StandardScaler  # encode labels + normalize data
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix  # evaluation
from sklearn.utils import resample  # used for oversampling

# plotting
import matplotlib
matplotlib.use("Agg")   # important: run without GUI (for saving images)
import matplotlib.pyplot as plt
import seaborn as sns   # nicer plots

# paths and config
DATA_CSV    = "landmarks/landmarks_combined.csv"   # dataset location
LABEL_COL   = "label"                             # column that contains labels
MODELS_DIR  = "output/models"                     # where to save models
PLOTS_DIR   = "output/plots"                      # where to save plots
RANDOM_STATE = 42                                 # for reproducibility (same results every run)

# create folders if not exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# Oversampling function
# -----------------------------
def oversample_minority(X, y, min_samples=2000):
    # idea: if a class has few samples → we duplicate it
    classes, counts = np.unique(y, return_counts=True)  # get each class and how many samples it has
    
    X_list, y_list = [X], [y]   # start with original data
    
    for cls, cnt in zip(classes, counts):
        if cnt < min_samples:   # if class is small
            idx = np.where(y == cls)[0]   # get indexes of this class
            
            n_needed = min_samples - cnt   # how many we need to reach 2000
            
            # resample (duplicate randomly with replacement)
            X_res = resample(X[idx], n_samples=n_needed,
                             random_state=RANDOM_STATE, replace=True)
            
            X_list.append(X_res)                    # add new samples
            y_list.append(np.full(n_needed, cls))   # add labels
    
    return np.vstack(X_list), np.concatenate(y_list)   # merge all data

# -----------------------------
# Data augmentation (add noise)
# -----------------------------
def augment(X, y, noise=0.008, copies=2):
    rng = np.random.default_rng(RANDOM_STATE)  # random generator
    
    Xl, yl = [X], [y]   # start with original data
    
    for _ in range(copies):
        # add small noise → makes model more robust
        Xl.append(X + rng.normal(0, noise, X.shape).astype(np.float32))
        yl.append(y)
    
    return np.vstack(Xl), np.concatenate(yl)

# -----------------------------
# Load data
# -----------------------------
print("Loading data...")
df = pd.read_csv(DATA_CSV)   # read CSV

print(f"  {len(df)} samples, {df[LABEL_COL].nunique()} classes")

# separate features and labels
X = df.drop(columns=[LABEL_COL]).values.astype(np.float32)  # all columns except label
le = LabelEncoder()                                         # convert labels (A,B,C → 0,1,2)
y = le.fit_transform(df[LABEL_COL].values)

# -----------------------------
# Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,              # 20% test
    random_state=RANDOM_STATE,
    stratify=y                 # keep same class distribution
)

# -----------------------------
# Balance + augment
# -----------------------------
print("Oversampling...")
X_train, y_train = oversample_minority(X_train, y_train, min_samples=2000)

print("Augmenting...")
X_train, y_train = augment(X_train, y_train)

# -----------------------------
# Normalization
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)   # learn mean/std and normalize
X_test  = scaler.transform(X_test)        # apply same normalization

print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

# -----------------------------
# Train models
# -----------------------------
print("\n" + "="*50)
print("Training Models...")
print("="*50)

# 1. Random Forest
print("\n1. Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,      # number of trees
    max_features="sqrt",   # how many features each tree sees
    random_state=RANDOM_STATE,
    n_jobs=-1              # use all CPU cores
)
rf.fit(X_train, y_train)

# 2. MLP (Neural Network)
print("\n2. Training MLP Classifier...")
mlp = MLPClassifier(
    #making mlp faster a bit by reducing the number of neurons and layers, but still keeping it powerful enough to learn complex patterns
    #3 layers took so long to train that I had to reduce it to 2 layers, but I kept the number of neurons high enough to capture complexity
    hidden_layer_sizes=(128,64),  # three hidden layers
    activation='relu',                   # activation function
    solver='adam',                       # optimizer
    batch_size=32,                       # batch size
    learning_rate='adaptive',            # adaptive learning rate
    learning_rate_init=0.001,            # initial learning rate
    max_iter=200,                        # maximum iterations
    early_stopping=True,                 # stop if no improvement
    validation_fraction=0.1,             # 10% for validation
    random_state=RANDOM_STATE,
    verbose=False
)
mlp.fit(X_train, y_train)

# 3. Voting Classifier (Ensemble)
print("\n3. Training Voting Classifier...")
voting_clf = VotingClassifier(
    estimators=[
        ('random_forest', rf),
        ('mlp', mlp)
    ],
    voting='soft',  # use predicted probabilities for voting
    # soft here means it will average the probabilities from both models and pick the class with highest average probability
    weights=[1, 1]  # equal weights for both models
    #we gave the models equal weights, but we could give more weight to the one that performs better on validation data
)
voting_clf.fit(X_train, y_train)

# -----------------------------
# Evaluation function
# -----------------------------
def evaluate_model(model, name, X_test, y_test, le):
   
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    
    print(f"\n{name} Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    return y_pred, acc, f1

# -----------------------------
# Evaluate all models
# -----------------------------
print("\n" + "="*50)
print("Model Evaluation")
print("="*50)

models = {
    'Random Forest': rf,
    'MLP': mlp,
    'Voting Classifier': voting_clf
}

results = {}
predictions = {}

for name, model in models.items():
    y_pred, acc, f1 = evaluate_model(model, name, X_test, y_test, le)
    results[name] = {'accuracy': acc, 'f1_score': f1}
    predictions[name] = y_pred

# -----------------------------
# Save models and preprocessing objects
# -----------------------------
print("\n" + "="*50)
print("Saving Models...")
print("="*50)

# Save Random Forest
with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "wb") as f:
    pickle.dump(rf, f)
print("  ✓ Random Forest saved")

# Save MLP
with open(os.path.join(MODELS_DIR, "mlp_classifier.pkl"), "wb") as f:
    pickle.dump(mlp, f)
print("  ✓ MLP saved")

# Save Voting Classifier
with open(os.path.join(MODELS_DIR, "voting_classifier.pkl"), "wb") as f:
    pickle.dump(voting_clf, f)
print("  ✓ Voting Classifier saved")

# Save preprocessing tools
with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)
print("  ✓ Label Encoder saved")

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("  ✓ Scaler saved")

# Save results as JSON
with open(os.path.join(MODELS_DIR, "model_results.json"), "w") as f:
    json.dump(results, f, indent=4)

# -----------------------------
# Plot confusion matrices
# -----------------------------
print("\n" + "="*50)
print("Generating Plots...")
print("="*50)

for name, y_pred in predictions.items():
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(
        cm,
        annot=True, fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax,
        annot_kws={'size': 10}
    )
    
    ax.set_title(f"Confusion Matrix - {name}", fontsize=16, fontweight='bold')
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    fig.tight_layout()
    
    # Save figure
    filename = f"cm_{name.lower().replace(' ', '_')}.png"
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Confusion matrix for {name} saved")

# -----------------------------
# Plot comparison bar chart
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

models_list = list(results.keys())
accuracies = [results[m]['accuracy'] for m in models_list]
f1_scores = [results[m]['f1_score'] for m in models_list]

x = np.arange(len(models_list))
width = 0.35

bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='coral', alpha=0.8)

ax.set_xlabel('Models', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=15, ha='right')
ax.legend(loc='lower right', fontsize=11)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  ✓ Model comparison chart saved")

# -----------------------------
# Print detailed classification reports
# -----------------------------
print("\n" + "="*50)
print("Detailed Classification Reports")
print("="*50)

for name, y_pred in predictions.items():
    print(f"\n{name}:")
    print("-" * 40)
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_
    ))

# -----------------------------
# Find and save best model
# -----------------------------
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = models[best_model_name]

print("\n" + "="*50)
print(f"🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"   F1 Score: {results[best_model_name]['f1_score']:.4f}")
print("="*50)

# Save best model separately
with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

with open(os.path.join(MODELS_DIR, "best_model_info.json"), "w") as f:
    json.dump({
        'model_name': best_model_name,
        'accuracy': results[best_model_name]['accuracy'],
        'f1_score': results[best_model_name]['f1_score']
    }, f, indent=4)

print("\n✅ All done! Models, plots, and results saved successfully.")
print(f"   📁 Models saved in: {MODELS_DIR}")
print(f"   📊 Plots saved in: {PLOTS_DIR}")