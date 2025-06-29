import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, PrecisionRecallDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelEvaluator:
    def __init__(self):
        self.results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Cross-Val Accuracy'])
        self.last_predictions = {} 
        self.feature_names = []

    def set_feature_names(self, names):
        self.feature_names = names
        logging.info(f"ModelEvaluator: Feature names set. Count: {len(self.feature_names)}")

    def evaluate_model(self, model_name, model, X_val, y_val, X_test, y_test, X_train_val, y_train_val):
        logging.info(f"Starting evaluation of model: {model_name}")

        y_val_pred = model.predict(X_val)
        y_val_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_val_prob = model.predict_proba(X_val)[:, 1]
            except AttributeError:
                logging.warning(f"Model {model_name} has predict_proba but failed to produce probabilities on validation set.")
        
        y_test_pred = model.predict(X_test)
        y_test_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_test_prob = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                logging.warning(f"Model {model_name} has predict_proba but failed to produce probabilities on test set.")

        accuracy_val = accuracy_score(y_val, y_val_pred)
        precision_val = precision_score(y_val, y_val_pred, zero_division=0)
        recall_val = recall_score(y_val, y_val_pred, zero_division=0)
        f1_val = f1_score(y_val, y_val_pred, zero_division=0)
        
        roc_auc_val = np.nan
        if y_val_prob is not None and len(np.unique(y_val)) > 1:
            try:
                roc_auc_val = roc_auc_score(y_val, y_val_prob)
            except ValueError as e:
                logging.warning(f"Could not calculate ROC-AUC for {model_name} on validation set: {e}")
        elif y_val_prob is not None:
             logging.warning(f"ROC-AUC skipped for {model_name} on validation set: Only one class present in y_val.")


        logging.info(f"Validation Metrics for {model_name}:")
        logging.info(f"Accuracy: {accuracy_val:.4f}, Precision: {precision_val:.4f}, Recall: {recall_val:.4f}, F1-Score: {f1_val:.4f}, ROC-AUC: {roc_auc_val:.4f}")

        cv_accuracy = None
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        try:
            cv_scores = cross_val_score(model, X_train_val, y_train_val, cv=skf, scoring='accuracy', n_jobs=-1)
            cv_accuracy = cv_scores.mean()
            logging.info(f"Cross-validation accuracy for {model_name}: {cv_accuracy:.4f}")
        except Exception as e:
            logging.error(f"Error during cross-validation for {model_name}: {e}")
            cv_accuracy = np.nan


        new_row = pd.DataFrame([{
            'Model': model_name,
            'Accuracy': accuracy_val,
            'Precision': precision_val,
            'Recall': recall_val,
            'F1-Score': f1_val,
            'ROC-AUC': roc_auc_val,
            'Cross-Val Accuracy': cv_accuracy
        }])
        self.results = pd.concat([self.results, new_row], ignore_index=True)
        logging.info(f"Evaluation of {model_name} complete.")

        self.last_predictions[model_name] = {
            'y_true_val': y_val,
            'y_pred_val': y_val_pred,
            'y_prob_val': y_val_prob,
            'y_true_test': y_test,
            'y_pred_test': y_test_pred,
            'y_prob_test': y_test_prob
        }
        return self.results

    def get_results(self):
        return self.results.sort_values(by='Cross-Val Accuracy', ascending=False)
    
    def get_best_model_info(self, metric='Cross-Val Accuracy'):
        if self.results.empty:
            logging.warning("ModelEvaluator: Results DataFrame is empty, cannot determine best model.")
            return None, 0
        
        if metric not in self.results.columns:
            logging.error(f"ModelEvaluator: Metric '{metric}' not found in results columns. Cannot determine best model by this metric.")
            return None, 0
        
        if self.results[metric].isnull().all():
            logging.warning(f"ModelEvaluator: '{metric}' column is all NaN. Cannot determine best model by this metric.")
            return None, 0

        best_row = self.results.loc[self.results[metric].idxmax()]
        return best_row['Model'], best_row[metric]

    def plot_confusion_matrix(self, model_name, ax):
        logging.info(f"ModelEvaluator: Plotting confusion matrix for {model_name}")
        if model_name not in self.last_predictions:
            logging.error(f"ModelEvaluator: No predictions found for model: {model_name} during plot_confusion_matrix call.")
            return

        preds = self.last_predictions[model_name]
        y_true = preds.get('y_true_test')
        y_pred = preds.get('y_pred_test')

        if y_true is None or y_pred is None:
            logging.error(f"ModelEvaluator: y_true_test or y_pred_test is None for {model_name}. Cannot plot confusion matrix.")
            return
        
        if len(y_true) == 0:
            logging.warning(f"ModelEvaluator: Test set is empty for {model_name}. Cannot plot confusion matrix.")
            ax.set_title(f"Confusion Matrix - {model_name} (Test Set Empty)")
            return

        logging.info(f"ModelEvaluator: y_true_test shape: {y_true.shape}, y_pred_test shape: {y_pred.shape}")
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name} (Test Set)')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    def plot_metric_comparison(self, metric, ax):
        logging.info(f"ModelEvaluator: Plotting metric comparison for {metric}")
        if self.results.empty:
            logging.warning("ModelEvaluator: No results to plot metric comparison.")
            ax.set_title(f"No Results for {metric} Comparison")
            return
        
        if metric not in self.results.columns:
            logging.error(f"ModelEvaluator: Metric '{metric}' not found in results columns.")
            ax.set_title(f"Metric '{metric}' Not Found")
            return


        plot_data = self.results[['Model', metric]].sort_values(by=metric, ascending=False)
        sns.barplot(x='Model', y=metric, data=plot_data, ax=ax, palette='viridis')
        ax.set_title(f'{metric} Comparison Across Models')
        ax.set_ylabel(metric)
        ax.set_xlabel('Model')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()

    def plot_roc_curve(self, model_name, ax):
        logging.info(f"ModelEvaluator: Plotting ROC curve for {model_name}")
        if model_name not in self.last_predictions:
            logging.error(f"ModelEvaluator: No predictions found for model: {model_name} during plot_roc_curve call.")
            return
        
        preds = self.last_predictions[model_name]
        y_true = preds.get('y_true_test')
        y_prob = preds.get('y_prob_test')

        if y_true is None or y_prob is None:
            logging.warning(f"ModelEvaluator: y_true_test or y_prob_test is None for {model_name}. Cannot plot ROC curve.")
            ax.set_title(f"ROC Curve - {model_name} (Probabilities N/A)")
            return
        
        if len(np.unique(y_true)) < 2:
            logging.warning(f"ModelEvaluator: Only one class present in y_true_test for {model_name}. Cannot plot ROC curve.")
            ax.set_title(f"ROC Curve - {model_name} (Single Class)")
            return
        
        logging.info(f"ModelEvaluator: y_true_test shape: {y_true.shape}, y_prob_test shape: {y_prob.shape}")
        RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
        ax.set_title(f'ROC Curve - {model_name} (Test Set)')
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.legend(loc='lower right')

    def plot_pr_curve(self, model_name, ax):
        logging.info(f"ModelEvaluator: Plotting PR curve for {model_name}")
        if model_name not in self.last_predictions:
            logging.error(f"ModelEvaluator: No predictions found for model: {model_name} during plot_pr_curve call.")
            return
        
        preds = self.last_predictions[model_name]
        y_true = preds.get('y_true_test')
        y_prob = preds.get('y_prob_test')

        if y_true is None or y_prob is None:
            logging.warning(f"ModelEvaluator: y_true_test or y_prob_test is None for {model_name}. Cannot plot PR curve.")
            ax.set_title(f"PR Curve - {model_name} (Probabilities N/A)")
            return
        
        if len(np.unique(y_true)) < 2:
            logging.warning(f"ModelEvaluator: Only one class present in y_true_test for {model_name}. Cannot plot PR curve.")
            ax.set_title(f"PR Curve - {model_name} (Single Class)")
            return

        logging.info(f"ModelEvaluator: y_true_test shape: {y_true.shape}, y_prob_test shape: {y_prob.shape}")
        PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
        ax.set_title(f'Precision-Recall Curve - {model_name} (Test Set)')

    def plot_feature_importance(self, model_name, model, ax):
        logging.info(f"ModelEvaluator: Plotting feature importance for {model_name}")
        importances = None
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            logging.info(f"ModelEvaluator: Using feature_importances_ for {model_name}. Length: {len(importances) if importances is not None else 'None'}")
        elif hasattr(model, 'coef_') and len(model.coef_.shape) == 1:
            importances = np.abs(model.coef_)
            logging.info(f"ModelEvaluator: Using coef_ (1D) for {model_name}. Length: {len(importances) if importances is not None else 'None'}")
        elif hasattr(model, 'coef_') and len(model.coef_.shape) > 1:
            importances = np.abs(model.coef_[0])
            logging.warning(f"ModelEvaluator: Using coef_ (2D) for {model_name}. Interpreted as importance for class 1. Length: {len(importances) if importances is not None else 'None'}")
        else:
            logging.warning(f"ModelEvaluator: Model '{model_name}' does not have direct feature importances or coefficients suitable for plotting.")
            ax.set_title(f"No Feature Importance for {model_name}")
            return

        if importances is None or len(importances) == 0:
             logging.error(f"ModelEvaluator: Importances array is None or empty for {model_name}.")
             ax.set_title(f"No Feature Importance for {model_name}")
             return

        if len(self.feature_names) == 0:
            logging.error(f"ModelEvaluator: Feature names are not set in ModelEvaluator for {model_name}.")
            ax.set_title("Feature Names Not Available")
            return
        
        if len(importances) != len(self.feature_names):
            logging.error(f"ModelEvaluator: Mismatch between importance array length ({len(importances)}) and feature names length ({len(self.feature_names)}) for {model_name}. This is critical.")
            ax.set_title("Feature Importance Mismatch")
            return

        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), ax=ax)
        ax.set_title(f'Feature Importance - {model_name} (Top 15)')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')