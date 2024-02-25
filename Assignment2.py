import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn import linear_model
from scipy.spatial import Delaunay
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error as mse
from scipy import linalg
from scipy.interpolate import interp1d, LinearNDInterpolator, NearestNDInterpolator
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from usearch.index import search, MetricKind, Matches, BatchMatches
import copy

def Gaussian(x):
    return np.where(np.abs(x)>4,0,1/(np.sqrt(2*np.pi))*np.exp(-1/2*x**2))

def Tricubic(x):
    return np.where(np.abs(x)>1,0,(1-np.abs(x)**3)**3)
def Epanechnikov(x):
    return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2))
def Quartic(x):
    return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2)

def weight_function(u,v,kern=Gaussian,tau=0.5):
    return kern(cdist(u, v, metric='euclidean')/(2*tau))

class Lowess:
    def __init__(self, kernel = Gaussian, tau=0.05):
        self.kernel = kernel
        self.tau = tau

    def fit(self, x, y):
        kernel = self.kernel
        tau = self.tau
        self.xtrain_ = x
        self.yhat_ = y

    def predict(self, x_new):
        check_is_fitted(self)
        x = self.xtrain_
        y = self.yhat_
        lm = linear_model.Ridge(alpha=0.001)
        w = weight_function(x,x_new,self.kernel,self.tau)

        if np.isscalar(x_new):
            lm.fit(np.diag(w)@(x.reshape(-1,1)),np.diag(w)@(y.reshape(-1,1)))
            yest = lm.predict([[x_new]])[0][0]
        else:
            
            n = len(x_new)
            yest_test = []
            #Looping through all x-points
            for i in range(n):
                lm.fit(np.diag(w[:,i])@x,np.diag(w[:,i])@y)
                yest_test.append(lm.predict([x_new[i]]))
        return np.array(yest_test).flatten()



'''
Part 1

Includes model class

Includes a basic example of fitting the model

Includes a kfold comparison to Xgboost where GBLowess beats mse

Inludes kfold comparison with scalars (Used canned sk function)


'''


class GradientBoostingLowess:
    '''
    Intilization statment
    
    '''
    def __init__(self, n_estimators=100, lowess_kernel=Gaussian, tau=0.05, use_learning_rate = False, lr = .1):
        self.n_estimators = n_estimators
        self.lowess_kernel = lowess_kernel
        self.tau = tau
        self.models = []
        self.curr_residuals = None
        self.use_learning_rate = use_learning_rate
        self.lr = lr

        
    '''Fitting Method, optional learning rate '''
    def fit(self, X, y):
        self.models = []
        for n in range(self.n_estimators+1):
            model = Lowess(kernel=self.lowess_kernel, tau=self.tau)
            #First model initializing 
            if self.curr_residuals is None:
                model.fit(X,y)
                y_pred = model.predict(X)
                self.curr_residuals =  y - y_pred.reshape(-1,1)
                self.models.append(model)
            #Boosting models, could be sped up. TODO combine model inference and residuals addition
            else:
                temp = ([temp_model.predict(X).reshape(-1,1) for temp_model in self.models])
                temp_arr = np.array(len(X)*[0], dtype = np.float64).reshape(-1,1)
                if self.use_learning_rate:
                    for index,res in enumerate(temp):
                        if index > 0:
                            temp_arr += self.lr*res
                        else:
                            temp_arr += res
                else:
                    for res in temp:
                        temp_arr += res
                self.curr_residuals = y - temp_arr
                model.fit(X,self.curr_residuals)
                self.models.append(model)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        #same as above
        check_is_fitted(self, 'is_fitted_')
        temp = ([x.predict(X).reshape(-1,1) for x in self.models])
        temp_arr = np.array(len(X)*[0], dtype = np.float64).reshape(-1,1)
        if self.use_learning_rate:    
            for index,res in enumerate(temp):
                if index > 0:
                    temp_arr += self.lr*res
                else:
                    temp_arr += res
        else:
            for res in temp:
                temp_arr += res
        return temp_arr
    
    #Functions needed for Sklearn compliance
    def get_params(self, deep=True):
        return {'n_estimators': self.n_estimators, 'lowess_kernel': self.lowess_kernel, 'tau': self.tau}
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    

'Basic Example'
data = pd.read_csv('cars.csv')
x = data.drop(columns=['MPG']).values
y = data['MPG'].values.reshape(-1,1)
scale  = StandardScaler()
x = scale.fit_transform(x[0:20,:])
y = y[0:20,:]
model = GradientBoostingLowess(n_estimators=3, tau = .5,use_learning_rate = True)
model.fit(x,y)
mse(y , model.predict(x))


'Concrete Example with kfold comparison to xgboost'
concrete = pd.read_csv('concrete.csv')
X = concrete.iloc[:,:-1].values
y = concrete.iloc[:,-1].values.reshape(-1,1)

mse_lwr = []
mse_rf = []
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
model_xb = XGBRegressor(n_estimators=5, max_depth=7, eta=.1, subsample=0.7, colsample_bytree=0.8)
model_lw = GradientBoostingLowess(n_estimators=3, tau = .5)
for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)

    model_lw.fit(xtrain,ytrain)
    yhat_lw = model_lw.predict(xtest)

    model_xb.fit(xtrain,ytrain)
    yhat_rf = model_xb.predict(xtest)

    mse_lwr.append(mse(ytest,yhat_lw))
    mse_rf.append(mse(ytest,yhat_rf))
print('The Cross-validated Mean Squared Error for Locally Weighted Boosting is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Xgboost is : '+str(np.mean(mse_rf)))



'Scalers comparision, 5-fold '
scalers = [StandardScaler(), MinMaxScaler(), QuantileTransformer(n_quantiles = 800)]
for scaler in scalers:
    pipeline = make_pipeline(scaler, model)
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"{scaler.__class__.__name__} - Average MSE: {-scores.mean()}")


'''
Part 2

Includes KNN Usearch Class

Includes Example using simulated data



'''

class UsearchNN:
    def __init__(self, k, metric = MetricKind.L2sq):
        '''
        k: Int (range 1, n)
        
        '''
        self.vectors = None
        self.targets = None
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        """Store the dataset and sets is_fitted"""
        self.vectors = X
        self.targets = y
        self.fitted_ = True
        return self
        
    def predict(self, X):
        '''
        Uses Usearch to find k nearest neighbors
        Expects X to be shape (batch_size, vector_size)
        
        Returns predictions of shape (batch_size,1)
        '''
        check_is_fitted(self, 'fitted_')
        output: BatchMatches = search(self.vectors, X, self.k,self.metric, exact=True)
        temp = []
        for set_ in output.keys:
            temp.append([self.targets[set_].mean()])
        return np.array(temp)
    
'Example'
knn = UsearchNN(k = 5)
vectors = np.random.rand(1000, 1024).astype(np.float32)
y = np.random.rand(1000, 1).astype(np.float32)
knn.fit(vectors,y)
output = (knn.predict(np.random.rand(10, 1024)))
