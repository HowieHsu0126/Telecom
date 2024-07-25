import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from data import Datasets
import collections


class Models:
    @staticmethod
    def evaluate_models(X, y, logger, k=3, tune_hyperparameters=True):
        logger.info("Evaluating multiple models...")
        models = {
            # 'RandomForest': RandomForestClassifier(n_jobs=-1),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=5000, n_jobs=-1),
            # 'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
            # 'HistGradientBoosting': HistGradientBoostingClassifier(),
            'XGBoost': xgb.XGBClassifier(n_estimators=5000, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
            # 'LightGBM': lgb.LGBMClassifier(n_jobs=-1),
            'CatBoost': CatBoostClassifier(n_estimators=5000, verbose=0),
            # Using MLPClassifier from sklearn
            # 'MLP': MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=200)
        }

        search_spaces = {
            'RandomForest': {
                'n_estimators': [100],
                'criterion': ["gini", "entropy"],
                'max_features': np.arange(0.05, 1.01, 0.05),
                'min_samples_split': range(2, 21),
                'min_samples_leaf': range(1, 21),
                'bootstrap': [True, False],
            },
            'ExtraTrees': {
                'n_estimators': [100],
                'criterion': ["gini", "entropy"],
                'max_features': np.arange(0.05, 1.01, 0.05),
                'min_samples_split': range(2, 21),
                'min_samples_leaf': range(1, 21),
                'bootstrap': [True, False],
            },
            'AdaBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1, 10]
            },
            'HistGradientBoosting': {
                'learning_rate': [0.01, 0.1, 0.2],
                'max_iter': [100, 200, 300]
            },
            'XGBoost': {
                'n_estimators': [100],
                'max_depth': range(1, 11),
                'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                'subsample': np.arange(0.05, 1.01, 0.05),
                'min_child_weight': range(1, 21),
                'verbosity': [0],
            },
            'LightGBM': {
                'boosting_type': ['gbdt', 'dart'],
                'min_child_samples': [1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000],
                'num_leaves': [2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250, 500],
                'colsample_bytree': [0.7, 0.9, 1.0],
                'subsample': [0.7, 0.9, 1.0],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000, 1500, 2000],
            },
            'CatBoost': {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 6, 9]
            },
            'MLP': {
                'hidden_layer_sizes': [(128, 64, 32)],
                'activation': ['relu'],
                'solver': ['adam'],
                'max_iter': [200]
            }
        }

        best_models = []
        for name, model in models.items():
            if tune_hyperparameters and name in search_spaces:
                logger.info(f"Evaluating {name} with hyperparameter search...")
                search = HalvingGridSearchCV(
                    model, search_spaces[name], scoring='f1', cv=5, factor=2, random_state=42, n_jobs=-1)
                search.fit(X, y)
                best_model = search.best_estimator_
                mean_score = search.best_score_
            else:
                logger.info(
                    f"Evaluating {name} without hyperparameter search...")
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
                best_model = model.fit(X, y)
                mean_score = np.mean(cv_scores)

            best_models.append((name, best_model, mean_score))
            logger.info(f"{name} - Best F1 Score: {mean_score}")

        # 排序模型得分并选择最佳的k个模型
        sorted_models = sorted(best_models, key=lambda x: x[2], reverse=True)
        selected_models = [(name, model)
                           for name, model, score in sorted_models[:k]]
        logger.info("Best models selected for ensemble.")
        return selected_models

    @staticmethod
    def train_model(X, y, best_models, logger):
        logger.info("Training ensemble model...")
        estimators = [(name, model)
                      for name, model in best_models]
        meta_learner = LogisticRegression(max_iter=1000)
        ensemble_model = StackingClassifier(
            estimators=estimators, final_estimator=meta_learner, cv=5)
        cv_scores = cross_val_score(ensemble_model, X, y, cv=5, scoring='f1')
        logger.info(
            f'Ensemble Model - Cross-Validated F1 Score: {cv_scores.mean()}')
        ensemble_model.fit(X, y)
        logger.info("Ensemble model training completed successfully.")
        return ensemble_model

    @staticmethod
    def pseudo_labeling(model, X_train, y_train, X_val, logger, threshold=0.95):
        logger.info("Applying pseudo-labeling...")
        pseudo_labels = model.predict_proba(X_val)[:, 1]
        high_confidence_indices = np.where(pseudo_labels >= threshold)[0]
        X_pseudo = X_val.iloc[high_confidence_indices]
        y_pseudo = np.ones(X_pseudo.shape[0])
        logger.info(
            f"Pseudo-labeling completed with {len(y_pseudo)} pseudo-labels.")

        X_combined = pd.concat([X_train, X_pseudo])
        y_combined = np.concatenate([y_train, y_pseudo])

        return X_combined, y_combined

    @staticmethod
    def predict_and_save(model, X_val, output_path, logger):
        logger.info("Predicting and saving results...")

        X_val['msisdn'] = X_val['msisdn']
        X_val['is_sa'] = model.predict(X_val).astype(int)

        X_val[['msisdn', 'is_sa']].to_csv(output_path, index=False)
        logger.info("Results saved successfully.")

    def run_pipeline(self, X, y, X_val, output_path, logger, tune_hyperparameters=False, adversarial_val=False, use_pseudo_labeling=False):
        best_models = self.evaluate_models(
            X, y, logger, tune_hyperparameters=tune_hyperparameters)

        if adversarial_val:
            sample_weights = Datasets.adversarial_validation(X, X_val, logger)
            estimators = [(name, model) for name, model in best_models]
            meta_learner = LogisticRegression(max_iter=1000)
            ensemble_model = StackingClassifier(
                estimators=estimators, final_estimator=meta_learner, cv=5)
            cv_scores = cross_val_score(ensemble_model, X, y, cv=StratifiedKFold(
                n_splits=5), scoring='f1', fit_params={'sample_weight': sample_weights})
            logger.info(
                f'Ensemble Model - Cross-Validated F1 Score: {cv_scores.mean()}')
            ensemble_model.fit(X, y, sample_weight=sample_weights)
        else:
            ensemble_model = self.train_model(X, y, best_models, logger)

        if use_pseudo_labeling:
            X_combined, y_combined = self.pseudo_labeling(
                ensemble_model, X, y, X_val, logger)
            ensemble_model.fit(X_combined, y_combined)
        else:
            ensemble_model.fit(X, y)
        logger.info("Ensemble model training completed successfully.")

        self.predict_and_save(ensemble_model, X_val, output_path, logger)
