import sys
import os
import pickle

from Colors import Colors
from DataTransformer import DataTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score


class ModelGenerator:

    def __init__(self):
        pass

    def optimize_model_rf(self, configs, train_data):
        X = train_data
        y = configs['train_meta_table']['classification_label'].tolist()
        n_iter = configs['n_iter']
        cv = configs['cross_val']
        seed = configs['seed']

        scoring = {
            'Accuracy': make_scorer(accuracy_score),
            'Precision': make_scorer(precision_score, average='weighted'),
            'Recall': make_scorer(recall_score, average='weighted')
        }

        if configs['autotune_hyperparameters'] == "grid":
            param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }

            if seed is not None:
                rf = RandomForestClassifier(random_state=seed)
            else:
                rf = RandomForestClassifier()

            grid_search = GridSearchCV(
                rf, param_grid=param_grid, cv=cv, scoring=scoring, refit='Accuracy', n_jobs=-1
            )

            grid_search.fit(X, y)

            return grid_search.best_estimator_, grid_search.cv_results_, grid_search.best_index_
        elif configs['autotune_hyperparameters'] == "random":
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
            if seed is not None:
                rf = RandomForestClassifier(random_state=seed)
            else:
                rf = RandomForestClassifier()

            # Use RandomizedSearchCV
            if seed is not None:
                random_search = RandomizedSearchCV(
                    rf, param_distributions=param_grid, n_iter=n_iter, 
                    cv=cv, scoring=scoring, refit='Accuracy', n_jobs=-1, random_state=seed
                )
            else:
                random_search = RandomizedSearchCV(
                    rf, param_distributions=param_grid, n_iter=n_iter, 
                    cv=cv, scoring=scoring, refit='Accuracy', n_jobs=-1
                )

            # Fit on training data
            random_search.fit(X, y)

            # Return best parameters and best model
            return random_search.best_estimator_, random_search.cv_results_, random_search.best_index_

    def optimize_model_svm(self, configs, train_data):
        X = train_data
        y = configs['train_meta_table']['classification_label'].tolist()
        n_iter = configs['n_iter']
        cv = configs['cross_val']
        seed = configs['seed']

        scoring = {
            'Accuracy': make_scorer(accuracy_score),
            'Precision': make_scorer(precision_score, average='weighted'),
            'Recall': make_scorer(recall_score, average='weighted')
        }
    
        if configs['autotune_hyperparameters'] == "grid":
            param_grid = {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
            }

            if seed is not None:
                svm = SVC(random_state=seed)
            else:
                svm = SVC()

            grid_search = GridSearchCV(
                svm, param_grid=param_grid, cv=cv, scoring=scoring, refit='Accuracy', n_jobs=-1
            )

            grid_search.fit(X, y)

            return grid_search.best_estimator_, grid_search.cv_results_, grid_search.best_index_
        elif configs['autotune_hyperparameters'] == "random":
            # Define the hyperparameter grid
            param_grid = {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
            }

            # Initialize the SVM model
            if seed is not None:
                svm = SVC(random_state=seed)
            else:
                svm = SVC()

            # Use RandomizedSearchCV
            if seed is not None:
                random_search = RandomizedSearchCV(
                    svm, param_distributions=param_grid, n_iter=n_iter, 
                    cv=cv, scoring=scoring, refit='Accuracy', n_jobs=-1, random_state=seed
                )
            else:
                random_search = RandomizedSearchCV(
                    svm, param_distributions=param_grid, n_iter=n_iter, 
                    cv=cv, scoring=scoring, refit='Accuracy', n_jobs=-1
                )

            # Fit on training data
            random_search.fit(X, y)

            # Return best parameters and best model
            print(f"Best Hyperparameters: {random_search.best_params_}")

            return random_search.best_estimator_, random_search.cv_results_, random_search.best_index_

    def train_model(self, configs, train_data):
        """Trains a classifier and returns the cv score and model."""

        scoring = {
            'Accuracy': make_scorer(accuracy_score),
            'Precision': make_scorer(precision_score, average='weighted'),
            'Recall': make_scorer(recall_score, average='weighted')
        }

        if configs['model_type'] == "RF":
            if configs['autotune_hyperparameters'] is None:
                try:
                    X = train_data
                    y = configs['train_meta_table']['classification_label'].tolist()
                    if configs['seed'] is not None:
                        rf = RandomForestClassifier(random_state=configs['seed'])
                    else:
                        rf = RandomForestClassifier()
                    cv = cross_validate(rf, X, y, cv=configs['cross_val'], scoring=scoring)
                    cv_scores = {
                        'Accuracy_Mean': cv['test_Accuracy'].mean(), 
                        'Accuracy_Std': cv['test_Accuracy'].std(), 
                        'Precision_Mean': cv['test_Precision'].mean(), 
                        'Precision_Std': cv['test_Precision'].std(), 
                        'Recall_Mean': cv['test_Recall'].mean(), 
                        'Recall_Std': cv['test_Recall'].std()
                    }
                    rf.fit(X, y)
                    params = rf.get_params()
                except Exception as e:
                    print(f"{Colors.ERROR}ERROR training RF model: {e}{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)
            else:
                try:
                    rf, cv, index = self.optimize_model_rf(configs, train_data)
                    cv_scores = {
                        'Accuracy_Mean': cv['mean_test_Accuracy'][index], 
                        'Accuracy_Std': cv['std_test_Accuracy'][index], 
                        'Precision_Mean': cv['mean_test_Precision'][index], 
                        'Precision_Std': cv['mean_test_Precision'][index], 
                        'Recall_Mean': cv['std_test_Recall'][index], 
                        'Recall_Std': cv['std_test_Recall'][index]
                    }
                    params = rf.get_params()
                except Exception as e:
                    print(f"{Colors.ERROR}ERROR tuning and training RF model: {e}{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

            model_information = {
                'cv_scores': cv_scores, 
                'params': params
            }
            return rf, model_information
        elif configs['model_type'] == "SVM":
            if configs['autotune_hyperparameters'] is None:
                try:
                    X = train_data
                    y = configs['train_meta_table']['classification_label'].tolist()
                    if configs['seed'] is not None:
                        svm = SVC(random_state=configs['seed'])
                    else:
                        svm = SVC()
                    cv = cross_validate(svm, X, y, cv=configs['cross_val'], scoring=scoring)
                    cv_scores = {
                        'Accuracy_Mean': cv['test_Accuracy'].mean(), 
                        'Accuracy_Std': cv['test_Accuracy'].std(), 
                        'Precision_Mean': cv['test_Precision'].mean(), 
                        'Precision_Std': cv['test_Precision'].std(), 
                        'Recall_Mean': cv['test_Recall'].mean(), 
                        'Recall_Std': cv['test_Recall'].std()
                    }
                    svm.fit(X, y)
                    params = svm.get_params()
                except Exception as e:
                    print(f"{Colors.ERROR}ERROR training SVM model: {e}{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)
            else:
                try:
                    svm, cv, index = self.optimize_model_svm(configs, train_data)
                    cv_scores = {
                        'Accuracy_Mean': cv['mean_test_Accuracy'][index], 
                        'Accuracy_Std': cv['std_test_Accuracy'][index], 
                        'Precision_Mean': cv['mean_test_Precision'][index], 
                        'Precision_Std': cv['std_test_Precision'][index], 
                        'Recall_Mean': cv['mean_test_Recall'][index], 
                        'Recall_Std': cv['std_test_Recall'][index]
                    }
                    params = svm.get_params()
                except Exception as e:
                    print(f"{Colors.ERROR}ERROR tuning and training SVM model: {e}{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

            model_information = {
                'cv_scores': cv_scores, 
                'params': params
            }
            return svm, model_information

    def validate_model(self, configs, model, validate_data):
        X_val = validate_data
        y_val = configs['validate_meta_table']['classification_label'].tolist()
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, average='weighted')
        val_recall = recall_score(y_val, y_val_pred, average='weighted')

        val_scores = {
            'Accuracy': val_accuracy, 
            'Precision': val_precision, 
            'Recall': val_recall
        }
        return val_scores

    def save_model(self, model, output_file_path):
        with open(output_file_path, "wb") as f:
            pickle.dump(model, f)
        print(f"{Colors.INFO}INFO: Model saved to '{output_file_path}'.{Colors.END}", file=sys.stderr, flush=True)

    def save_model_information(self, metrics, output_file_path):
        with open(output_file_path, "w") as out_file:
            # save model parameters
            out_file.write("---MODEL PARAMETERS---\n")
            for param, value in metrics['params'].items():
                out_file.write(f"{param}: {value}\n")

            # save train/test scores
            out_file.write("\n---TRAIN/TEST CV SCORES---\n")
            for score, value in metrics['cv_scores'].items():
                out_file.write(f"{score}: {value}\n")
                               
            # save validation scores
            out_file.write("\n---VALIDATION SCORES---\n")
            for score, value in metrics['val_scores'].items():
                out_file.write(f"{score}: {value}\n")
        print(f"{Colors.INFO}INFO: Model information saved to '{output_file_path}'.{Colors.END}", file=sys.stderr, flush=True)

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

        train_matrix = data_transformer.prep_vectorized_pairs_for_scikitlearn(feature_df=configs['feature_table'], bool_dict=train_bool_dict)
        train_matrix.index = configs['train_meta_table'].index.copy()
        validate_matrix = data_transformer.prep_vectorized_pairs_for_scikitlearn(feature_df=configs['feature_table'], bool_dict=validate_bool_dict)
        validate_matrix.index = configs['validate_meta_table'].index.copy()

        # train model
        print("TRAINING MODEL", file=sys.stderr, flush=True)
        model, model_information = self.train_model(configs=configs, train_data=train_matrix)

        # validate model
        print("VALIDATING MODEL", file=sys.stderr, flush=True)
        model_information['val_scores'] = self.validate_model(configs=configs, model=model, validate_data=validate_matrix)

        # Save model to "trained_model.pkl" in the specified output dir (configs['output_dir'])
        print("SAVING MODEL", file=sys.stderr, flush=True)
        model_output_path = os.path.join(configs['output_dir'], "trained_model.pkl")
        self.save_model(model=model, output_file_path=model_output_path)

        # Save train/validate information to "model_information.txt" in the specified output dir
        print("SAVING MODEL INFORMATION", file=sys.stderr, flush=True)
        metrics_output_path = os.path.join(configs['output_dir'], "model_information.txt")
        self.save_model_information(metrics=model_information, output_file_path=metrics_output_path)
        
        return model, model_information
    