from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'; OUT.mkdir(exist_ok=True)
CSV=ROOT/'titanic.csv'

# One and only network/cache loader in this module.
def load_once():
    try:
        df=sns.load_dataset('titanic')
        df.to_csv(CSV,index=False)
        return df
    except Exception as exc:
        print(f"sns.load_dataset unavailable ({exc}); using committed titanic.csv fallback.")
        return pd.read_csv(CSV)

def clean_for_eda(df):
    x=df.copy()
    missing=x.isna().mean().mul(100)
    missing=missing[missing>0].sort_values(ascending=False)
    missing.to_csv(OUT/'missing_percentages.csv',header=['missing_pct'])
    # Threshold rule: <5% drop rows; 5–30% median/mode imputation; >30% drop column.
    for c,pct in missing.items():
        if pct < 5:
            x=x.dropna(subset=[c])
        elif pct <= 30:
            if pd.api.types.is_numeric_dtype(x[c]): x[c]=x[c].fillna(x[c].median())
            else: x[c]=x[c].fillna(x[c].mode(dropna=True).iloc[0])
        else:
            # deck is >30%; preserve information by dropping it from EDA/model features.
            x=x.drop(columns=[c])
    return x

def iqr_count(s):
    q1,q3=s.quantile([.25,.75]); iqr=q3-q1
    return int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum())

def plot_eda(df):
    for col in ['age','fare']:
        fig,ax=plt.subplots(); ax.hist(df[col].dropna(),bins=30); ax.set_title(f'{col.title()} distribution'); fig.tight_layout(); fig.savefig(OUT/f'{col}_hist.png'); plt.close(fig)
        fig,ax=plt.subplots(); ax.boxplot(df[col].dropna(),vert=False); ax.set_title(f'{col.title()} box plot'); fig.tight_layout(); fig.savefig(OUT/f'{col}_box.png'); plt.close(fig)
    fare_stats={'mean':df.fare.mean(),'median':df.fare.median(),'mode':df.fare.mode().iloc[0]}
    pd.Series(fare_stats).to_csv(OUT/'fare_stats.csv')
    outliers={'age':iqr_count(df.age),'fare':iqr_count(df.fare)}; pd.Series(outliers).to_csv(OUT/'iqr_outliers.csv')
    # Boolean-mask breakdowns
    sex_rate=df.groupby('sex',dropna=False).survived.mean().rename('survival_rate')
    pclass_rate=df.groupby('pclass',dropna=False).survived.mean().rename('survival_rate')
    both=df.groupby(['sex','pclass'],dropna=False).survived.mean().rename('survival_rate')
    sex_rate.to_csv(OUT/'survival_by_sex.csv'); pclass_rate.to_csv(OUT/'survival_by_pclass.csv'); both.to_csv(OUT/'survival_by_sex_pclass.csv')
    corr=df[['survived','pclass','age','sibsp','parch','fare']].corr(); corr.to_csv(OUT/'correlation_matrix.csv')
    fig,ax=plt.subplots(figsize=(7,6)); sns.heatmap(corr,annot=True,cmap='vlag',ax=ax); fig.tight_layout(); fig.savefig(OUT/'correlation_heatmap.png'); plt.close(fig)
    pairs=[]; cols=corr.columns
    for i in range(len(cols)):
        for j in range(i+1,len(cols)): pairs.append((abs(corr.iloc[i,j]),cols[i],cols[j],corr.iloc[i,j]))
    pd.DataFrame(sorted(pairs,reverse=True)[:2],columns=['abs_corr','feature_1','feature_2','corr']).to_csv(OUT/'strongest_correlations.csv',index=False)
    # 4+ multivariate charts
    fig,ax=plt.subplots(); df.groupby(['sex','pclass'])['survived'].mean().unstack().plot(kind='bar',ax=ax); ax.set_title('Survival rate by sex and class'); fig.tight_layout(); fig.savefig(OUT/'story_1_sex_pclass.png'); plt.close(fig)
    fig,ax=plt.subplots(); df.boxplot(column='fare',by=['survived','sex'],ax=ax); plt.suptitle(''); ax.set_title('Fare distribution by survival and sex'); fig.tight_layout(); fig.savefig(OUT/'story_2_fare_survival.png'); plt.close(fig)
    fig,ax=plt.subplots()
    for val in sorted(df['survived'].unique()):
        sub=df[df['survived']==val]; ax.scatter(sub['age'],sub['fare'],label=f'survived={val}')
    ax.legend(); ax.set_title('Age, fare and survival'); fig.tight_layout(); fig.savefig(OUT/'story_3_age_fare.png'); plt.close(fig)
    fig,ax=plt.subplots(); df.groupby(['pclass','sex'])['survived'].mean().unstack().plot(kind='line',marker='o',ax=ax); ax.set_title('Survival probability by passenger class'); fig.tight_layout(); fig.savefig(OUT/'story_4_class_sex.png'); plt.close(fig)
    # exploratory standardization, not fed to modeling
    z=df[['age','fare']].copy(); z=(z-z.mean())/z.std(); z.agg(['mean','std']).to_csv(OUT/'eda_standardization_check.csv')

def model(df):
    target='survived'; features=['pclass','sex','age','sibsp','parch','fare','embarked']
    X=df[features].copy(); y=df[target].astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    num=['pclass','age','sibsp','parch','fare']; cat=['sex','embarked']
    pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
    models={'Logistic Regression':LogisticRegression(max_iter=1000,random_state=42),'Decision Tree':DecisionTreeClassifier(max_depth=5,random_state=42),'Random Forest':RandomForestClassifier(n_estimators=300,random_state=42)}
    rows=[]
    for name,est in models.items():
        pipe=Pipeline([('preprocess',pre),('model',est)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); proba=pipe.predict_proba(Xte)[:,1]
        rows.append({'model':name,'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred),'recall':recall_score(yte,pred),'f1':f1_score(yte,pred),'auc':roc_auc_score(yte,proba)})
        cm=confusion_matrix(yte,pred); pd.DataFrame(cm,index=['actual_0','actual_1'],columns=['pred_0','pred_1']).to_csv(OUT/f'{name.lower().replace(" ","_")}_confusion_matrix.csv')
        fpr,tpr,_=roc_curve(yte,proba); fig,ax=plt.subplots(); ax.plot(fpr,tpr,label=f'AUC={rows[-1]["auc"]:.3f}'); ax.plot([0,1],[0,1],linestyle='--'); ax.legend(); ax.set_title(f'{name} ROC'); fig.tight_layout(); fig.savefig(OUT/f'{name.lower().replace(" ","_")}_roc.png'); plt.close(fig)
        if name=='Decision Tree':
            names=pipe.named_steps['preprocess'].get_feature_names_out(); fig,ax=plt.subplots(figsize=(18,9)); plot_tree(pipe.named_steps['model'],feature_names=names,class_names=['0','1'],filled=False,ax=ax); fig.tight_layout(); fig.savefig(OUT/'decision_tree.png'); plt.close(fig)
    metrics=pd.DataFrame(rows); metrics.to_csv(OUT/'classifier_metrics.csv',index=False)
    # imbalance comparison using Random Forest
    imb=[]
    variants={'baseline':RandomForestClassifier(n_estimators=300,random_state=42),'balanced':RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=42)}
    for label,est in variants.items():
        p=Pipeline([('preprocess',pre),('model',est)]); p.fit(Xtr,ytr); pr=p.predict(Xte); imb.append({'variant':label,'precision':precision_score(yte,pr),'recall':recall_score(yte,pr),'f1':f1_score(yte,pr)})
    sm=ImbPipeline([('preprocess',pre),('smote',SMOTE(random_state=42)),('model',RandomForestClassifier(n_estimators=300,random_state=42))]); sm.fit(Xtr,ytr); pr=sm.predict(Xte); imb.append({'variant':'SMOTE_train_only','precision':precision_score(yte,pr),'recall':recall_score(yte,pr),'f1':f1_score(yte,pr)})
    pd.DataFrame(imb).to_csv(OUT/'imbalance_comparison.csv',index=False)
    # Grid search and OOB refit
    grid=GridSearchCV(Pipeline([('preprocess',pre),('model',RandomForestClassifier(random_state=42))]),{'model__n_estimators':[100,200,300],'model__max_depth':[None,5,10],'model__max_features':['sqrt','log2']},cv=5,scoring='f1',n_jobs=-1); grid.fit(Xtr,ytr)
    best_params={k.replace('model__',''):v for k,v in grid.best_params_.items()}; tuned=RandomForestClassifier(oob_score=True,random_state=42,**best_params); final=Pipeline([('preprocess',pre),('model',tuned)]); final.fit(Xtr,ytr); (OUT/'gridsearch.txt').write_text(f'best_params={best_params}\ncv_best_f1={grid.best_score_:.6f}\noob_score={tuned.oob_score_:.6f}\n')
    # Regression: predict fare from non-target features only.
    reg_features=['survived','pclass','age','sibsp','parch','sex','embarked']
    RX=df[reg_features]; ry=df['fare']; Rtr,Rte,rytr,ryte=train_test_split(RX,ry,test_size=.2,random_state=42)
    rnum=['pclass','age','sibsp','parch']; rcat=['sex','embarked']; rpre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),rnum),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),rcat)])
    rp=Pipeline([('preprocess',rpre),('model',LinearRegression())]); rp.fit(Rtr,rytr); pred=rp.predict(Rte); mae=mean_absolute_error(ryte,pred); rmse=mean_squared_error(ryte,pred)**.5; r2=r2_score(ryte,pred); n=len(Rte); p=rp.named_steps['preprocess'].transform(Rte).shape[1]; adj=1-(1-r2)*(n-1)/(n-p-1); pd.DataFrame([{'MAE':mae,'RMSE':rmse,'R2':r2,'Adjusted_R2':adj}]).to_csv(OUT/'regression_metrics.csv',index=False)
    resid=ryte-pred; fig,ax=plt.subplots(); ax.scatter(pred,resid); ax.axhline(0,linestyle='--'); ax.set_xlabel('Predicted fare'); ax.set_ylabel('Residual'); ax.set_title('Fare regression residuals'); fig.tight_layout(); fig.savefig(OUT/'regression_residuals.png'); plt.close(fig)
    # complete end-to-end classifier pipeline persisted
    final.fit(Xtr,ytr); joblib.dump(final,ROOT/'best_pipeline.joblib')
    loaded=joblib.load(ROOT/'best_pipeline.joblib'); sample_pred=loaded.predict(Xte.head(3)); (OUT/'reload_check.txt').write_text(f'Predictions after joblib reload on raw rows: {sample_pred.tolist()}\n')
    metrics.to_csv(OUT/'final_model_comparison_classification.csv',index=False)
    return metrics

def main():
    df=load_once(); print('INFO'); df.info(); print(df.describe(include='all')); print('shape=',df.shape)
    cleaned=clean_for_eda(df); plot_eda(cleaned); metrics=model(cleaned); print(metrics.to_string(index=False)); print('Outputs:',OUT)
if __name__=='__main__': main()
