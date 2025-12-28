from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mlvern import Forge

def main():
    # Load dataset
    data = load_iris(as_frame=True)
    df = data.frame
    target = 'target'

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target]), df[target], test_size=0.2, random_state=42
    )

    # Initialize Forge
    forge = Forge(project="Iris Classification")

    # Inspect data
    inspect_report = forge.inspect(df, target=target)
    print("Data Inspection Report:", inspect_report)

    # Compute statistics
    stats_report = forge.statistics(df, target="target")
    print("Statistical Analysis Report:", stats_report)

    # Run risk checks
    risk_report = forge.risk_check(df, target=target, sensitive=['sepal width (cm)'], baseline=X_train, train=X_train, test=X_test)
    print("Risk Check Report:", risk_report)
    
    plots = forge.eda(df, target=target)
    print("EDA Plots Generated:", plots)
    

if __name__ == "__main__":
    main()
