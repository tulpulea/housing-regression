import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
import pymc_bart as pmb
import pymc as pm

if __name__ == "__main__":
    seed = 123
    np.random.seed(seed)

    data = pd.read_csv("Housing.csv")

    quart1 = data["price"].quantile(0.25)
    quart3 = data["price"].quantile(0.75)
    IQR = quart3-quart1
    l = quart1 - 1.5*IQR
    u = quart3 + 1.5*IQR

    data_cleaned = data[(data["price"] >= l) & (data["price"] <= u)].copy()
    data_cleaned["date"] = pd.to_datetime(data["date"], format="ISO8601")
    data_cleaned["year_constructed"] = pd.to_datetime(data["yr_built"],format="%Y")
    data_cleaned["is_renovated"] = ""
    data_cleaned["is_renovated"] = data_cleaned["yr_renovated"].apply((lambda r: int(bool(r))))
    data_cleaned["years_since_construction"] = data_cleaned["date"].dt.year - data_cleaned["year_constructed"].dt.year
    data_cleaned["log_sqft_above"] = np.log(data_cleaned["sqft_above"])

    data_processed = data_cleaned.drop(labels=["id","date","year_constructed","sqft_above","yr_built","yr_renovated","lat","long"],axis = 1)

    enc = OneHotEncoder(sparse_output=False)
    enc.fit(pd.DataFrame(data_processed["zipcode"]))
    enc.set_output(transform="pandas")
    zipcode_one_hot = enc.transform(pd.DataFrame(data_processed["zipcode"]))

    binary_vars = data_processed.loc[:,["waterfront","is_renovated"]] 
    data_processed.drop(["waterfront","is_renovated","zipcode"],axis=1,inplace=True) 

    scaler = StandardScaler() 
    scaler.set_output(transform="pandas")
    scaler.fit(data_processed)
    data_standardized = scaler.transform(data_processed)

    X_y = data_standardized.join([binary_vars,zipcode_one_hot])
    y_data = X_y["price"]
    X_data = X_y.drop("price",axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=1)

    y_bar = np.mean(y_train)
    print("training error:",mean_squared_error(y_train,np.array([y_bar]*len(y_train))))
    print("baseline testing error:",mean_squared_error(y_test,np.array([y_bar]*len(y_test))))

    ridge = Ridge() 
    cv = GridSearchCV(ridge, param_grid={"alpha":[0.1,0.5,1,5,10]},scoring="neg_mean_squared_error")
    cv.fit(X_train,y_train)
    print("ridge regression testing error:",-1*cv.score(X_test,y_test)) 

    with pm.Model() as mod: #standard syntax for modeling in PyMC
        X = pm.MutableData("X",X_train) #saving as mutable for easy change from training to testing data
        Y = y_train
        mu = pmb.BART("mu",X=X,Y=Y,m=200) #200 tree bart model
        sigma = pm.HalfNormal("sigma", sigma=1) #assuming standard deviation of predictions is half normal
        y_pred = pm.Normal("y_pred",mu=mu,sigma=sigma,observed=y_train,shape = mu.shape) 
        bart_idata = pm.sample(random_seed=seed,tune=100,draws=1000) #1000 burn-in iterations and 1000 draws
        posterior_pred_train = pm.sample_posterior_predictive(trace=bart_idata,random_seed=seed) #sample from posterior

    with mod:
        X.set_value(X_test) #set test data
        posterior_pred_test = pm.sample_posterior_predictive(trace=bart_idata, random_seed=seed)

    stacked = posterior_pred_test.posterior_predictive.mean(dim=["chain","draw"]) #take mean over chains and draws
    y_pred_test = stacked.y_pred.values

    print("BART ensemble testing error:",mean_squared_error(y_test,y_pred_test))