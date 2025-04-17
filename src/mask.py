import pandas as pd
import yaml
import argparse
import os

class MaskingProcessor:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()
        self.original_path = self.config['dataset']['original_path']
        self.masked_path = self.config['dataset']['masked_path']
        self.target_variable = self.config['dataset']['target_variable']
        
        self.data = pd.read_csv(self.original_path)
        self.masked_data = self.data.copy()
        self.masking_attributes = self.config['masking']['attributes']

    def load_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def generalize(self, values, M):
        values = pd.Series(values).astype(float)
        min_val = values.min()
        max_val = values.max()
        
        if min_val == max_val:
            return pd.Series([f"{int(min_val)}:{int(min_val)}" for _ in values.index], index=values.index)
        
        total_range = max_val - min_val
        bucket_size = total_range // M
        remainder = int(total_range % M)
        
        buckets = []
        start = int(min_val)
        for i in range(M):
            extra = 1 if i < remainder else 0
            end = start + bucket_size + extra
            buckets.append((start, end))
            start = end + 1
        
        def assign_bucket(v):
            for (s, e) in buckets:
                if s <= v <= e:
                    return f"{s}:{e}"
            s, e = buckets[-1]
            return f"{s}:{e}"
        
        return values.apply(assign_bucket)

    def erase_digits(self, values, num_digits):
        def erase(v):
            if pd.isnull(v):
                return v
            v_str = str(int(float(v)))
            if len(v_str) <= num_digits:
                erased_str = '0' * len(v_str)
                return f"{erased_str}:{erased_str}"
            else:
                prefix = v_str[:-num_digits]
                suffix = '0' * num_digits
                new_val = prefix + suffix
                return f"{new_val}:{new_val}"
        return values.apply(erase)

    def suppress(self, values):
        return pd.Series(['*' for _ in values.index], index=values.index)

    def apply_masking(self):
        for attribute, function_list in self.masking_attributes.items():
            if attribute not in self.masked_data.columns:
                continue
            original_values = self.masked_data[attribute]
            if not isinstance(function_list, list) or not function_list:
                continue
            
            for func_info in function_list:
                func_name = func_info.get('function')
                params = func_info.get('params', {})
                
                if func_name == 'generalize':
                    M = params.get('M')
                    if M is None:
                        continue
                    original_values = self.generalize(original_values, M)
                
                elif func_name == 'erase_digits':
                    num_digits = params.get('num_digits', 1)
                    original_values = self.erase_digits(original_values, num_digits)
                
                elif func_name == 'suppress':
                    original_values = self.suppress(original_values)
                
                else:
                    pass
            
            self.masked_data[attribute] = original_values

    def save_masked_data(self):
        output_dir = os.path.dirname(self.masked_path)
        os.makedirs(output_dir, exist_ok=True)
        self.masked_data.to_csv(self.masked_path, index=False)

    def process(self):
        self.apply_masking()
        self.save_masked_data()

def main(config_path):
    processor = MaskingProcessor(config_path)
    processor.process()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config.yaml file')
    args = parser.parse_args()
    main(args.config)
