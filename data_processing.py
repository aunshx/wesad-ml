import numpy as np
import pandas as pd
import pickle
import os
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class WESADProcessor:
    def __init__(self, window_size=700):
        """
        Initialize the WESAD data processor
        """
        self.window_size = window_size
        self.scaler = StandardScaler()
        
        current_dir = os.getcwd()
        
        self.base_path = os.path.join(current_dir, 'data')
        
    def load_all_subjects(self):
        """
        Load data from all subjects with improved error handling
        """
        all_subjects_data = []
        
        print(f"\nLooking for subject folders in: {self.base_path}")
        
        
        try:
            available_subjects = [d for d in os.listdir(self.base_path) 
                                if os.path.isdir(os.path.join(self.base_path, d)) 
                                and d.startswith('S')]
            print(f"Found subject folders: {available_subjects}")
        except Exception as e:
            print(f"Error accessing data folder: {str(e)}")
            return []
        
        for subject_folder in available_subjects:
            subject_path = os.path.join(self.base_path, subject_folder)
            pickle_file = os.path.join(subject_path, f"{subject_folder}.pkl")
            
            if os.path.exists(pickle_file):
                try:
                    print(f"Loading data for {subject_folder}...")
                    with open(pickle_file, 'rb') as f:
                        data = pickle.load(f, encoding='latin1')
                        
                        
                        if 'signal' in data and 'label' in data:
                            all_subjects_data.append(data)
                            print(f"Successfully loaded subject {subject_folder}")
                        else:
                            print(f"Warning: Invalid data structure for {subject_folder}")
                except Exception as e:
                    print(f"Error loading subject {subject_folder}: {str(e)}")
            else:
                print(f"Pickle file not found: {pickle_file}")
        
        print(f"\nSuccessfully loaded {len(all_subjects_data)} subjects")
        return all_subjects_data

    def clean_signal(self, data):
        """
        Clean raw physiological signals
        """
        
        data = np.nan_to_num(data, nan=np.nanmean(data))
        
        
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data_clean = np.clip(data, lower_bound, upper_bound)
        
        
        window = signal.windows.hanning(5)
        data_smooth = signal.convolve(data_clean, window, mode='same') / sum(window)
        
        return data_smooth

    def extract_features(self, window_data, sampling_rate=64):
        """
        Extract both time and frequency domain features
        """
        
        time_features = [
            np.mean(window_data),
            np.std(window_data),
            np.max(window_data),
            np.min(window_data),
            np.sqrt(np.mean(np.square(window_data))),  
            stats.skew(window_data),
            stats.kurtosis(window_data),
            np.ptp(window_data)  
        ]
        
        
        try:
            fft_vals = np.abs(np.fft.rfft(window_data))
            fft_freq = np.fft.rfftfreq(len(window_data), 1/sampling_rate)
            
            freq_features = [
                fft_freq[np.argmax(fft_vals)],  
                np.average(fft_freq, weights=fft_vals),  
                np.median(fft_freq),  
                np.std(fft_vals)  
            ]
        except Exception as e:
            print(f"Warning: Error in frequency feature extraction: {str(e)}")
            freq_features = [0, 0, 0, 0]  
        
        return np.array(time_features + freq_features)

    def process_signal(self, signal_data):
        """
        Process entire signal and extract features
        """
        features = []
        signal_clean = self.clean_signal(signal_data)
        
        for i in range(0, len(signal_clean), self.window_size):
            window = signal_clean[i:i + self.window_size]
            if len(window) == self.window_size:
                window_features = self.extract_features(window)
                features.append(window_features)
        
        return np.array(features) if features else np.array([])

    def process_subject(self, subject_data):
        """
        Process all signals for one subject
        """
        try:
            wrist_data = subject_data['signal']['wrist']
            
            
            bvp_features = self.process_signal(wrist_data['BVP'])
            eda_features = self.process_signal(wrist_data['EDA'])
            temp_features = self.process_signal(wrist_data['TEMP'])
            
            if len(bvp_features) == 0 or len(eda_features) == 0 or len(temp_features) == 0:
                return None, None
            
            
            features = np.hstack((bvp_features, eda_features, temp_features))
            
            
            labels = subject_data['label'][::self.window_size]
            labels = labels[:len(features)]
            
            return features, labels
            
        except Exception as e:
            print(f"Error processing subject: {str(e)}")
            return None, None

    def process_all_data(self):
        """
        Process all subjects and prepare final dataset
        """
        print("Starting data processing pipeline...")
        
        
        all_subjects_data = self.load_all_subjects()
        if not all_subjects_data:
            raise ValueError("No subjects were successfully loaded!")
        
        
        all_features = []
        all_labels = []
        
        print("\nProcessing subjects...")
        for i, subject_data in enumerate(all_subjects_data):
            print(f"Processing subject {i+2}...")
            features, labels = self.process_subject(subject_data)
            if features is not None and labels is not None:
                all_features.append(features)
                all_labels.append(labels)
        
        if not all_features:
            raise ValueError("No features were successfully extracted!")
        
        
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        
        mask = np.logical_or(y == 1, y == 2)
        X = X[mask]
        y = y[mask]
        
        
        y = (y == 2).astype(int)
        
        
        X_scaled = self.scaler.fit_transform(X)
        
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test

def main():
    try:
        
        processor = WESADProcessor()
        
        
        X_train, X_test, y_train, y_test = processor.process_all_data()
        
        
        print("\nDataset Information:")
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")
        print(f"Number of features: {X_train.shape[1]}")
        print("\nClass distribution:")
        print(f"Training set - Baseline: {sum(y_train == 0)}, Stress: {sum(y_train == 1)}")
        print(f"Testing set - Baseline: {sum(y_test == 0)}, Stress: {sum(y_test == 1)}")
        
        
        np.save('processed_data/X_train.npy', X_train)
        np.save('processed_data/X_test.npy', X_test)
        np.save('processed_data/y_train.npy', y_train)
        np.save('processed_data/y_test.npy', y_test)
        print("\nProcessed data saved to 'processed_data' directory")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease verify that:")
        print("1. The script is in the same directory as the 'data' folder")
        print("2. The 'data' folder contains all subject folders (S2-S17)")
        print("3. Each subject folder contains the required .pkl file")

if __name__ == "__main__":
    main()