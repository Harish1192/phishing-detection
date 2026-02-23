import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score
from xgboost import XGBClassifier
from feature_extractor import extract_features

print("Loading Dataset...")

df=pd.read_csv("data/phishing_site_urls.csv")

print(f"Total records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nClass Distribution:\n{df['Label'].value_counts()}")
print(f"\nNull values:\n{df.isnull().sum()}")

print("\n Cleaning Data....")

df.dropna(subset=['URL','Label'],inplace=True)
df['URL']=df['URL'].str.strip()
df.drop_duplicates(subset=['URL'],inplace=True)
print(f"Total records after cleaning: {len(df)}")

print("\nEncoding labels....")
df['label_encoded']=df['Label'].apply(lambda x:1 if str(x).strip().lower()=='bad' else 0)

print("\nExtracting features...")
feature_list=[]

for i, url in enumerate(df['URL']):
    features=extract_features(url)
    feature_list.append(features)

print("\nFeature extraction completed....")

X=pd.DataFrame(feature_list)
y=df['label_encoded'].values

print("\nSample features: {X.iloc[0]}")

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("\n Training model.....")

neg=(y_train==0).sum()
pos=(y_train==1).sum()
ratio=neg/pos 


model=XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train,y_train)

print("\nEvaluating model....")


y_pred_proba=model.predict_proba(X_test)[:,1]
precisions,recalls,thresholds=precision_recall_curve(y_test,y_pred_proba)

f1_scores=2*(precisions*recalls)/(precisions+recalls+1e-8)
best_idx=f1_scores.argmax()
best_threshold=thresholds[best_idx]

joblib.dump(best_threshold,"models/threshold.pkl")


threshold=best_threshold
y_pred=(y_pred_proba>=threshold).astype(int)

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)    
f1=f1_score(y_test,y_pred)
roc_auc=roc_auc_score(y_test,y_pred_proba)
conf_matrix=confusion_matrix(y_test,y_pred)

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
print(f"ROC AUC: {roc_auc*100:.2f}%")

print("\nConfusion Matrix:")
print(f"Valid predicted as  (TN): {conf_matrix[0][0]}")
print(f"Valid predicted as Phishing (FP): {conf_matrix[0][1]}")
print(f"Phishing predicted as Valid (FN) : {conf_matrix[1][0]}")
print(f"Phishing predicted as Phishing (TP): {conf_matrix[1][1]}")

print("\n Saving model")

os.makedirs("models",exist_ok=True)
joblib.dump(model,"models/phishing_ml_model.pkl")

print("Model saved")


