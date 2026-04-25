# signLanguage
# signLanguage
PS C:\Users\rajha\ai_project> $env:PYTHONIOENCODING="utf-8"
PS C:\Users\rajha\ai_project> python train_models.py
=======================================================
  ASL P2 — Model Training & Comparison Pipeline
=======================================================

[1/5] Loading data...
  → 95889 samples, 36 classes
  → Train: 76711  |  Test: 19178

[2/5] Training Random Forest...
  → Best params: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}
  → Accuracy: 0.9277  |  F1: 0.8977
  → Saved → output/models\random_forest.pkl
  → Confusion matrix → output/plots\cm_randomforest.png

[3/5] Training MLP (scikit-learn)...
  → Stopped at iter: 51
  → Accuracy: 0.9237  |  F1: 0.8707
  → Saved → output/models\sklearn_mlp.pkl
  → Confusion matrix → output/plots\cm_mlp.png
  → Training curve → output/plots\mlp_training_curve.png

[4/5] Comparing...
  RandomForest → Acc: 0.9277  F1: 0.8977
  Sklearn MLP  → Acc: 0.9237  F1: 0.8707
  Winner: RandomForest
  → Report saved → output/comparison_report.json
  → Comparison chart → output/plots\model_comparison.png

[5/5] Done! Files ready for Person 3:
  best_model      → output/models\random_forest.pkl
  label_encoder   → output/models\label_encoder.pkl
  scaler          → output/models\scaler.pkl

  Note: Person 3: apply scaler.transform() on landmarks before inference.

All done!

