import pandas as pd
import yaml
import argparse
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class CustomModelProcessorDataset3:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()
        self.original_data = pd.read_csv(self.config['dataset']['original_path'])
        self.masked_data = pd.read_csv(self.config['dataset']['masked_path'])
        self.target_variable = self.config['dataset']['target_variable']

    def load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def prepare_data(self, df):
        X = df.drop(columns=[self.target_variable])
        y = df[self.target_variable]
        X = pd.get_dummies(X, drop_first=True)
        return X, y

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        return accuracy_score(y_test, preds)

    def process(self):
        Xo, yo = self.prepare_data(self.original_data)
        Xo_train, Xo_test, yo_train, yo_test = train_test_split(Xo, yo, test_size=0.3, random_state=42)
        acc_orig = self.train_and_evaluate(Xo_train, Xo_test, yo_train, yo_test)

        Xm, ym = self.prepare_data(self.masked_data)
        Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xm, ym, test_size=0.3, random_state=42)
        acc_masked = self.train_and_evaluate(Xm_train, Xm_test, ym_train, ym_test)

        acc_diff = abs(acc_orig - acc_masked) * 100.0

        print(f"Original Accuracy: {acc_orig}")
        print(f"Masked Accuracy:   {acc_masked}")
        print(f"Accuracy Difference: {acc_diff}%")

def main():
    parser = argparse.ArgumentParser(description='Gaussian Naive Bayes for Dataset 3')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()
    CustomModelProcessorDataset3(args.config).process()

if __name__ == "__main__":
    main()