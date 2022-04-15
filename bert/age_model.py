import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error
import optuna
from optuna.samplers import TPESampler


def feval_lgb_Age(preds, lgbm_train):
    labels =lgbm_train.get_label()
    return 'Age Error', round(1.0 / (1.0 + mean_absolute_error(y_true = labels, y_pred = preds)),7), True


def optuna_tuning(X, y):
    print(y)
    X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    def objective(trial):
        param_grid = {
            'num_leaves': trial.suggest_int('num_leaves', 2 ** 3, 2 ** 9),
            'num_boost_round': trial.suggest_int('num_boost_round', 100, 8000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'reg_alpha': trial.suggest_int('reg_alpha', 1, 50),
            'reg_lambda': trial.suggest_int('reg_lambda', 5, 100),
            'objective': 'regression',
            'metric': 'mse',
            'boosting': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1),
            'bagging_freq': 1,
            'bagging_seed': 66,
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1),
            'feature_fraction_seed': 66,
            'verbose': -1
        }
        trn_data = lgb.Dataset(X_trn, label=y_trn, categorical_feature="")
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature="")
        clf = lgb.train(param_grid, trn_data, valid_sets=[trn_data, val_data], verbose_eval=False,
                        early_stopping_rounds=100)
        pred_val = clf.predict(X_val)
        mae_ = mean_absolute_error(y_val, pred_val)
        return mae_
    train_time = 1 * 10 * 60  # h * m * s
    study = optuna.create_study(direction='minimize', sampler=TPESampler(), study_name='LgbRegressor')
    study.optimize(objective, timeout=train_time)

    print(f'Number of finished trials: {len(study.trials)}')
    print('Best trial:')
    trial = study.best_trial

    print(f'\tValue: {trial.value}')
    print('\tParams: ')
    for key, value in trial.params.items():
        print('\t\t{}: {}'.format(key, value))


def get_age_prediction(X_train, y_train, X_test):
    # optuna_tuning(X_train, y_train)

    lgb_params = {
        "objective": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 16,
        "num_boost_round": 2000,
        "max_depth": 7,
        "reg_alpha": 38,
        "reg_lambda": 51,
        "bagging_fraction": 0.79,
        "feature_fraction": 0.8,
        'early_stopping_rounds': 100,
        'learning_rate': 0.03
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lgb_age_models = []
    y_pred = 0
    y_val_pred = np.zeros((X_train.shape[0],))
    importance_df = None
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):

        X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
        y_trn, y_val = y_train[trn_idx], y_train[val_idx]

        lgbm_train = lgb.Dataset(X_trn, y_trn)
        lgbm_valid = lgb.Dataset(X_val, y_val)

        model_mae = lgb.train(params=lgb_params,
                        train_set=lgbm_train,
                        valid_sets=[lgbm_train, lgbm_valid],
                        num_boost_round=100000,
                        feval = feval_lgb_Age,
                        verbose_eval=100)
        y_val_pred[val_idx] = model_mae.predict(X_val)
        y_pred += model_mae.predict(X_test) / 5.0
        lgb_age_models.append(model_mae)

        importance = model_mae.feature_importance('gain')
        feature_importance = pd.DataFrame({'feature_name':X_train.columns, 'importance_%d'%fold:importance})
        if importance_df is None:
            importance_df = feature_importance
        else:
            importance_df = importance_df.merge(feature_importance, on='feature_name')
    print(1.0 / (1.0 + mean_absolute_error(y_train, y_val_pred)))

    importance_df['importance'] = importance_df[['importance_%d'%fold for fold in range(5)]].sum(axis=1)
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    importance_df.to_csv('age_feature_importance.csv', index=False)

    return 1.0 / (1.0 + mean_absolute_error(y_train, y_val_pred)), y_pred