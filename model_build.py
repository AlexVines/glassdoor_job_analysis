import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('Cleaned data.csv')

#pick valuable columns

df_model = df[['avg', 'Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'Competitors', 'hourly',
          'provided', 'State', 'same_state', 'age', 'Python', 'Spark', 'aws', 'Excel', 'job_simp',
          'Seniority', 'descr_len']]


#get dummy data

df_dum = pd.get_dummies(df_model)

#train test split

X = df_dum.drop('avg', axis=1)
y = df_dum.avg.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#linear regress (we got a  score = -21)

X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)

lrm = LinearRegression()
lrm.fit(X_train, y_train)
cvs = cross_val_score(lrm, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)

#lasso (we got a score = -17)

las = Lasso()
cvs2 = cross_val_score(las, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)

alph = []
err = []
for i in range(1, 100):
    alph.append(i/100)
    las = Lasso(alpha=(i/100))
    cvs2 = cross_val_score(las, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)
    err.append(np.mean(cvs2))


plt.plot(alph, err)
#plt.show()
error = tuple(zip(alph, err))
df_err = pd.DataFrame(error, columns=['alph', 'err'])
bs = df_err[df_err.err == max(df_err.err)]
alpha = bs.alph.values[0]
las2 = Lasso(alpha=alpha)
las2.fit(X_train, y_train)
#print(alpha)

#random forest (we got a score = -15)

rf = RandomForestRegressor()
cvs3 = cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)
#print(np.mean(cvs3))

#tune model with gridsearchCV (score = -15)

# parameters = {'n_estimators':range(10, 300, 10), 'criterion':('mse', 'mae'), 'max_features':('auto', 'sqrt', 'log2')}
# gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
# gs.fit(X_train, y_train)
# #print(gs.best_score_)
# print(gs.best_estimator_)

#test
pred_lmr = lrm.predict(X_test)
pred_las = las2.predict(X_test)
pred_gs = gs.best_estimator_.predict(X_test)
#print(model.fit().summary())
