import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier # pip install xgboost
from lightgbm import LGBMClassifier # pip install lightgbm
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTrainer:
    def __init__(self, model_dir='models'):
        self.models = {}
        self.param_grids = {} 
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.define_models()

    def define_models(self):
        self.models = {
            'LogisticRegression': LogisticRegression(random_state=42, solver='liblinear', max_iter=1000),
            'RandomForestClassifier': RandomForestClassifier(random_state=42),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
            'SVC': SVC(random_state=42, probability=True),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'XGBoostClassifier': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'LightGBMClassifier': LGBMClassifier(random_state=42),
        }
        logging.info("Base machine learning models defined.")

        self.param_grids = {
            'LogisticRegression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2']
            },
            'RandomForestClassifier': {
                'n_estimators': [50, 100],
                'max_depth': [10, 20],
                'min_samples_leaf': [1, 2]
            },
            'GradientBoostingClassifier': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'SVC': {
                'C': [0.1, 1],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf']
            },
            'KNeighborsClassifier': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            },
            'XGBoostClassifier': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            },
            'LightGBMClassifier': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [20, 31],
                'max_depth': [-1]
            }
        }
        logging.info("Hyperparameter grids for GridSearchCV defined.")

        self._define_ensemble_models()

    def _define_ensemble_models(self):
        if not self.models:
            self.define_models()

        estimators = [
            ('lr', LogisticRegression(random_state=42, solver='liblinear', max_iter=1000, C=1)),
            ('rf', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)),
            ('gb', GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1)),
            ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_estimators=100, learning_rate=0.1)),
            ('lgbm', LGBMClassifier(random_state=42, n_estimators=100, learning_rate=0.1))
        ]
        self.models['VotingClassifier'] = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        
        self.param_grids['VotingClassifier'] = {
            'weights': [[1,1,1,1,1], [0.5,1,1,1,1], [1,0.5,1,1,1]]
        }
        logging.info("VotingClassifier defined.")

        level0_estimators = [
            ('rf_stack', RandomForestClassifier(random_state=42, n_estimators=100)),
            ('xgb_stack', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_estimators=100)),
            ('lgbm_stack', LGBMClassifier(random_state=42, n_estimators=100))
        ]
        self.models['StackingClassifier'] = StackingClassifier(
            estimators=level0_estimators,
            final_estimator=LogisticRegression(random_state=42, solver='liblinear', max_iter=1000),
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            n_jobs=-1,
            passthrough=True
        )
        self.param_grids['StackingClassifier'] = {
            'final_estimator__C': [0.1, 1],
        }
        logging.info("StackingClassifier defined.")


    def train_model(self, model_name, X_train, y_train):
        if model_name not in self.models:
            logging.error(f"Model '{model_name}' is not defined.")
            return None
        
        logging.info(f"Starting training of base model: {model_name}")
        model = self.models[model_name]
        try:
            model.fit(X_train, y_train)
            logging.info(f"Model '{model_name}' successfully trained.")
            return model
        except Exception as e:
            logging.error(f"Error training model '{model_name}': {e}", exc_info=True)
            return None

    def train_all_models(self, X_train, y_train):
        trained_models = {}
        for name, model in self.models.items():
            trained_model = self.train_model(name, X_train, y_train)
            if trained_model:
                trained_models[name] = trained_model
        return trained_models

    def tune_hyperparameters(self, model_name, X_train, y_train, scoring='f1'):
        if model_name not in self.models:
            logging.error(f"Model '{model_name}' is not defined for hyperparameter tuning.")
            return None
        if model_name not in self.param_grids or not self.param_grids[model_name]:
            logging.warning(f"No parameter grid defined for model '{model_name}'. Skipping tuning and returning base model.")
            return self.models[model_name]

        base_model = self.models[model_name]
        param_grid = self.param_grids[model_name]
        
        logging.info(f"Starting hyperparameter tuning for model: {model_name} with scoring '{scoring}'")
        logging.info(f"Using parameter grid: {param_grid}")

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        try:
            grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, 
                                       cv=skf, scoring=scoring, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            logging.info(f"Hyperparameter tuning for {model_name} completed.")
            logging.info(f"Best parameters: {grid_search.best_params_}")
            logging.info(f"Best {scoring}-Score (CV): {grid_search.best_score_:.4f}")
            
            tuned_model_key = f"{model_name}_tuned_by_{scoring}"
            self.models[tuned_model_key] = grid_search.best_estimator_
            return grid_search.best_estimator_
        except Exception as e:
            logging.error(f"Error during hyperparameter tuning for model {model_name}: {e}", exc_info=True)
            return None

    def evaluate_with_cross_validation(self, model, X, y, cv=5, scoring='accuracy'):
        logging.info(f"Performing {cv}-fold stratified cross-validation for {model.__class__.__name__}...")
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        try:
            scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
            logging.info(f"Cross-validation results ({scoring}): {scores}")
            logging.info(f"Mean {scoring} CV: {scores.mean():.4f} (+/- {scores.std():.4f})")
            return scores.mean(), scores.std()
        except Exception as e:
            logging.error(f"Error during cross-validation for model {model.__class__.__name__}: {e}", exc_info=True)
            return None, None

    def save_model(self, model, model_name):
        filename = os.path.join(self.model_dir, f'{model_name}.joblib')
        try:
            joblib.dump(model, filename)
            logging.info(f"Model '{model_name}' successfully saved to '{filename}'")
            return True
        except Exception as e:
            logging.error(f"Error saving model '{model_name}': {e}", exc_info=True)
            return False

    def load_model(self, model_name):
        filename = os.path.join(self.model_dir, f'{model_name}.joblib')
        if os.path.exists(filename):
            try:
                model = joblib.load(filename)
                logging.info(f"Model '{model_name}' successfully loaded from '{filename}'")
                return model
            except Exception as e:
                logging.error(f"Error loading model '{model_name}': {e}", exc_info=True)
                return None
        else:
            logging.warning(f"Model '{model_name}' not found at path '{filename}'.")
            return None

    def load_model_from_path(self, file_path):
        if os.path.exists(file_path):
            try:
                model = joblib.load(file_path)
                logging.info(f"Model successfully loaded from '{file_path}'")
                return model
            except Exception as e:
                logging.error(f"Error loading model from '{file_path}': {e}", exc_info=True)
                return None
        logging.error(f"File not found at specified path: {file_path}")
        return None