import sys
import os
import pickle

from Colors import Colors
from DataTransformer import DataTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score


class ModelGenerator:

    def __init__(self):
        # TODO
        pass

    def optimize_model_rf(self, configs, train_data):
        X = train_data
        y = configs['train_meta_table']
        n_iter = configs['n_iter']
        cv = configs['cross_val']
        seed = configs['seed']

        if configs['autotune_hyperparameters'] == "grid_search":
            param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }

            rf = RandomForestClassifier(random_state=seed)

            grid_search = GridSearchCV(
                rf, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1
            )

            grid_search.fit(X, y)

            return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_

        elif configs['autotune_hyperparameters'] == "random_search":
            # Define the hyperparameter grid
            param_grid = {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 10],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }

            # Initialize the Random Forest model
            rf = RandomForestClassifier(random_state=seed)

            # Use RandomizedSearchCV
            random_search = RandomizedSearchCV(
                rf, param_distributions=param_grid, n_iter=n_iter, 
                cv=cv, scoring='accuracy', n_jobs=-1, random_state=seed
            )

            # Fit on training data
            random_search.fit(X, y)

            # Return best parameters and best model
            return random_search.best_estimator_, random_search.best_score_, random_search.best_params_

    def optimize_model_svm(self, configs, train_data):
        X = train_data
        y = configs['train_meta_table']
        n_iter = configs['n_iter']
        cv = configs['cross_val']
        seed = configs['seed']
    
        if configs['autotune_hyperparameters'] == "grid_search":
            param_grid = {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
            }

            svm = SVC(random_state=seed)

            grid_search = GridSearchCV(
                svm, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1
            )

            grid_search.fit(X, y)

            return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_
        
        elif configs['autotune_hyperparameters'] == "random_search":
            # Define the hyperparameter grid
            param_grid = {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
            }

            # Initialize the SVM model
            svm = SVC(random_state=seed)

            # Use RandomizedSearchCV
            random_search = RandomizedSearchCV(
                svm, param_distributions=param_grid, n_iter=n_iter, 
                cv=cv, scoring='accuracy', n_jobs=-1, random_state=seed
            )

            # Fit on training data
            random_search.fit(X, y)

            # Return best parameters and best model
            print(f"Best Hyperparameters: {random_search.best_params_}")

            return random_search.best_estimator_, random_search.best_score_ , random_search.best_params_

    def train_model(self, configs, train_data):
        """Trains a classifier and returns the cv score and model."""
        if configs['model_type'] == "RandomForest":
            if configs['autotune_hyperparameters'] == "":
                try:
                    X = train_data
                    y = configs['train_meta_table']
                    rf = RandomForestClassifier(random_state=configs['seed'])
                    cv_scores = cross_val_score(rf, X, y, cv=configs['cross_val'], scoring='accuracy')
                    rf.fit(X, y)
                    return rf, cv_scores
                except Exception as e:
                    raise RuntimeError(f"Error training Random Forest: {e}") from e
                
            else:
                self.optimize_model_rf(configs, train_data)

        elif configs['model_type'] == "SVM":
            if configs['autotune_hyperparameters'] == "":
                try:
                    X = train_data
                    y = configs['train_meta_table']
                    svm = SVC(random_state=configs['seed'])
                    cv_scores = cross_val_score(svm, X, y, cv=configs['cross_val'], scoring='accuracy')
                    svm.fit(X, y)
                    return svm, cv_scores
                except Exception as e:
                    raise RuntimeError(f"Error training Random Forest: {e}") from e
                
            else:
                self.optimize_model_svm(configs, train_data)

    def validate_model(self, configs, validate_data):
        model = configs['model']
        X_val = validate_data
        y_val = configs['validate_meta_table']
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        return val_accuracy, val_precision, val_recall

    def save_model(self, model, output_file_path):
        with open(output_file_path, "wb") as f:
            pickle.dump(model, f)
        print(f"{Colors.INFO}INFO: Model saved to '{output_file_path}'.{Colors.END}", file=sys.stderr, flush=True)

    def save_model_information(self, metrics, output_file_path):
        # TODO: saves performance metrics and hyperparameter info to a file
        # performance metrics include: accuracy, precision, recall (train and validate)
        print(f"{Colors.INFO}INFO: Model performance metrics saved to '{output_file_path}'.{Colors.END}", file=sys.stderr, flush=True)

    def run_model_generator(self, configs):
        data_transformer = DataTransformer()
        if configs['impute_NA_missing']:
            print("ADDING MISSING PROTEINS AND IMPUTING NA", file=sys.stderr, flush=True)
            configs['train_quant_table'] = data_transformer.add_missing_proteins(feature_df=configs['feature_table'], quant_df=configs['train_quant_table'])
            configs['validate_quant_table'] = data_transformer.add_missing_proteins(feature_df=configs['feature_table'], quant_df=configs['validate_quant_table'])
        else:
            print("FILTERING OUT RULES CONTAINING MISSING PROTEINS")
            configs['feature_table'] = data_transformer.filter_rules(feature_df=configs['feature_table'], quant_df=configs['train_quant_table'])
            configs['feature_table'] = data_transformer.filter_rules(feature_df=configs['feature_table'], quant_df=configs['validate_quant_table'])
    
        print("TRANSFORMING DATA", file=sys.stderr, flush=True)
        train_bool_dict = data_transformer.transform_df(feature_df=configs['feature_table'], quant_df=configs['train_quant_table'])
        validate_bool_dict = data_transformer.transform_df(feature_df=configs['feature_table'], quant_df=configs['validate_quant_table'])

        train_matrix = data_transformer.prep_vectorized_pairs_for_scikitlearn(train_bool_dict)
        validate_matrix = data_transformer.prep_vectorized_pairs_for_scikitlearn(validate_bool_dict)
        #TODO Make sure meta and quant data are in same order
        # train model, 
        print("TRAINING MODEL", file=sys.stderr, flush=True)
        model, cv_scores = self.train_model(configs = configs, train_data = train_matrix)
        configs['model'] = model    # Save model in configs for validation step

        # validate model
        print("VALIDATING MODEL", file=sys.stderr, flush=True)
        val_accuracy, val_precision, val_recall = self.validate_model(configs = configs, validate_data = validate_matrix)

        # Save model to "trained_model.pkl" in the specified output dir (configs['output_dir'])
        print("SAVING MODEL", file=sys.stderr, flush=True)
        model_output_path = os.path.join(configs['output_dir'], "trained_model.pkl")
        self.save_model(model=model, output_file_path=model_output_path)

        # Store performance metrics in a dictionary
        performance_metrics = {'cross_validation_scores': cv_scores,'validation_accuracy': val_accuracy,'validation_precision': val_precision,'validation_recall': val_recall}

        # Save train/validate information to "model_performance_metrics.???" in the specified output dir
        print("SAVING MODEL PERFORMANCE METRICS", file=sys.stderr, flush=True)
        metrics_output_path = os.path.join(configs['output_dir'], "model_performance_metrics.txt")  # TODO: fix file extension when function is written
        self.save_performance_metrics(metrics=performance_metrics, output_file_path=metrics_output_path)
        
        return model, performance_metrics