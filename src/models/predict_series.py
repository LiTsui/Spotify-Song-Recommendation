from src.utils import h
from src.models.neural_net import NeuralNet
import pandas as pd
import torch
import joblib

"""
Accurately predict playoff series outcomes based on testing data.
"""


def main():

    STATS = [
        "PointsPG_DIFF",
        "OppPointsPG_DIFF",
        "DiffPointsPG_DIFF",
        "E_OFF_RATING_DIFF",
        "OFF_RATING_DIFF",
        "E_DEF_RATING_DIFF",
        "DEF_RATING_DIFF",
        "E_NET_RATING_DIFF",
        "NET_RATING_DIFF",
        "AST_PCT_DIFF",
        "AST_TO_DIFF",
        "AST_RATIO_DIFF",
        "OREB_PCT_DIFF",
        "DREB_PCT_DIFF",
        "REB_PCT_DIFF",
        "TM_TOV_PCT_DIFF",
        "EFG_PCT_DIFF",
        "TS_PCT_DIFF",
        "E_PACE_DIFF",
        "PACE_DIFF",
        "PIE_DIFF",
        "H2H_WIN_RATIO",
    ]

    test_df = pd.read_csv("../data/testing.csv")

    # load scaler to transform testing data
    scaler = joblib.load("../data/model/scaler.pkl")
    test_df[STATS] = scaler.transform(test_df[STATS])

    X_test = h.convert(test_df, STATS)
    y_test = h.convert(test_df, "LABEL").unsqueeze(1)

    # load model data based on validation accuracy
    model = NeuralNet()
    model.load_state_dict(torch.load("../data/model/best_model.pt"))

    model.eval()
    with torch.no_grad():
        preds = (model(X_test) >= 0.5).float()
        acc = (preds == y_test).float().mean().item()

        h.actual_vs_pred(model, X_test, y_test, preds)

        print(f"Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
