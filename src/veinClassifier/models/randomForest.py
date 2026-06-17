from sklearn.ensemble import RandomForestClassifier  

def randomForest_train(X_train, y_train, n_estimators):
      
    # Create model  
    model = RandomForestClassifier(  
        n_estimators=n_estimators, 
        max_depth=25, 
        random_state=42,
        n_jobs=-1
    )  
      
    # Train  
    model.fit(X_train, y_train)  
    return model
  
def randomForest_eval(model, X_test):
    # Predict  
    y_pred = model.predict(X_test) 
    return y_pred


