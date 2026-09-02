import os
import sys
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from src.logger import logging
from src.exception import CustomException


class ModelTrainer:

    def __init__(self):

        self.artifacts_dir = os.path.join(
            "artifacts"
        )

        os.makedirs(
            self.artifacts_dir,
            exist_ok=True
        )

    def evaluate_model(
        self,
        model_name,
        model,
        X_test,
        y_test
    ):
        """
        Evaluates a trained classification model.
        """

        try:

            logging.info(
                f"Evaluating model: {model_name}"
            )

            y_pred = model.predict(
                X_test
            )

            y_proba = model.predict_proba(
                X_test
            )[:, 1]

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            auc = roc_auc_score(
                y_test,
                y_proba
            )

            precision = precision_score(
                y_test,
                y_pred
            )

            recall = recall_score(
                y_test,
                y_pred
            )

            f1 = f1_score(
                y_test,
                y_pred
            )

            results = {
                "model": model_name,
                "accuracy": accuracy,
                "roc_auc": auc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

            logging.info(
                f"{model_name} results: {results}"
            )

            return results

        except Exception as e:

            logging.error(
                f"Error evaluating {model_name}"
            )

            raise CustomException(
                e,
                sys
            )

    def initiate_model_trainer(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ):
        """
        Trains Logistic Regression and Random Forest,
        evaluates both models, selects the best model,
        and saves it along with the preprocessor.
        """

        try:

            logging.info(
                "Starting model training"
            )

            # ==========================================
            # MODEL 1: LOGISTIC REGRESSION
            # ==========================================

            logging.info(
                "Training Logistic Regression"
            )

            logistic_model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )

            logistic_model.fit(
                X_train,
                y_train
            )

            logistic_results = self.evaluate_model(
                "Logistic Regression",
                logistic_model,
                X_test,
                y_test
            )

            # ==========================================
            # MODEL 2: RANDOM FOREST
            # ==========================================

            logging.info(
                "Training Random Forest"
            )

            random_forest_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

            random_forest_model.fit(
                X_train,
                y_train
            )

            random_forest_results = self.evaluate_model(
                "Random Forest",
                random_forest_model,
                X_test,
                y_test
            )

            # ==========================================
            # MODEL COMPARISON
            # ==========================================

            results = [
                logistic_results,
                random_forest_results
            ]

            best_model_results = max(
                results,
                key=lambda x: x["roc_auc"]
            )

            if best_model_results["model"] == "Logistic Regression":

                best_model = logistic_model

            else:

                best_model = random_forest_model

            logging.info(
                f"Best model selected: "
                f"{best_model_results['model']}"
            )

            # ==========================================
            # SAVE BEST MODEL
            # ==========================================

            model_path = os.path.join(
                self.artifacts_dir,
                "model.pkl"
            )

            with open(
                model_path,
                "wb"
            ) as file:

                pickle.dump(
                    best_model,
                    file
                )

            # ==========================================
            # SAVE PREPROCESSOR
            # ==========================================

            preprocessor_path = os.path.join(
                self.artifacts_dir,
                "preprocessor.pkl"
            )

            with open(
                preprocessor_path,
                "wb"
            ) as file:

                pickle.dump(
                    preprocessor,
                    file
                )

            logging.info(
                f"Best model saved to: {model_path}"
            )

            logging.info(
                f"Preprocessor saved to: "
                f"{preprocessor_path}"
            )

            return (
                results,
                best_model_results,
                model_path,
                preprocessor_path
            )

        except Exception as e:

            logging.error(
                "Exception occurred during model training"
            )

            raise CustomException(
                e,
                sys
            )