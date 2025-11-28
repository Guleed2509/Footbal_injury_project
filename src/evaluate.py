from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns, matplotlib.pyplot as plt
import joblib, pandas as pd

def evaluate(model_path:str, test_csv:str):

    model = joblib.load(model_path)
    df = pd.read_csv(test_csv)

    X = df.drop(columns=["injured"])
    y = df["injured"]

    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1]

    print("\n📊 Evaluation Report:\n", classification_report(y,preds))
    print("\nROC-AUC:", roc_auc_score(y,probs))

    sns.heatmap(confusion_matrix(y,preds),annot=True)
    plt.savefig("plots/confusion_matrix.png")
    plt.show()
