import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
from torchvision import transforms
import numpy as np
import os
from typing import Optional, Tuple, Dict

class DataMushroom(Dataset):
    def __init__(self, file_csv_x, file_csv_y=None):
        """
        Инициализация датасета грибов
        Args:
            file_csv_x: DataFrame с входными признаками
            file_csv_y: DataFrame с целевыми переменными (опционально)
        """
        self.file_csv_x = file_csv_x.copy().astype(int, errors='ignore') 
        self.file_csv_y = file_csv_y.copy().astype(int, errors='ignore') if file_csv_y is not None else pd.DataFrame()
        self.encoders_x = {}
        self.encoders_y = {}

        print('СУПЕР ОСНОВНОЙ X')
        print(self.file_csv_x)
        print('СУПЕР ОСНОВНОЙ Y')
        print(self.file_csv_y)
        
        # Сохраняем служебные столбцы отдельно
        self.observation_ids = self.file_csv_x['observationID'].copy()
        self.filenames = self.file_csv_x['filename'].copy()
        
        # Удаляем служебные столбцы перед обработкой
        self.file_csv_x = self.file_csv_x.drop(['observationID', 'filename'], axis=1)
        
        # Сохраняем исходные метки до токенизации
        if not self.file_csv_y.empty:
            self.original_labels = self.file_csv_y.copy()
            # Сохраняем category_id отдельно
            self.category_ids = self.file_csv_y['category_id'].copy()
            # Удаляем category_id перед токенизацией
            self.file_csv_y = self.file_csv_y.drop(['category_id'], axis=1)
        
        self._preprocess_data()
        
        print('ПОСЛЕ ОБРАБОТКИ X')
        print(self.token_x)
        print('ПОСЛЕ ОБРАБОТКИ Y')
        print(self.token_y)
        
    def _preprocess_data(self):
        """Предобработка данных: токенизация и заполнение пропусков"""
        self.token_x, self.token_y = self._tokenize_features()
        print('ОСНОВНОЙ X')
        print(self.token_x)
        print('ОСНОВНОЙ Y')
        print(self.token_y)
        self.token_x, self.token_y = self._fill_missing_values()
        print('НЕ ОСНОВНОЙ X')
        print(self.token_x)
        print('НЕ ОСНОВНОЙ Y')
        print(self.token_y)
        
    def _tokenize_features(self):
        """Токенизация категориальных признаков"""
        # Токенизация X
        for column in self.file_csv_x.select_dtypes(include=['object']).columns:
            self.encoders_x[column] = LabelEncoder()
            self.file_csv_x[column] = self.encoders_x[column].fit_transform(
                self.file_csv_x[column].astype(str)
            )
            
        # Токенизация y (кроме category_id)
        if not self.file_csv_y.empty:
            for column in self.file_csv_y.columns:
                self.encoders_y[column] = LabelEncoder()
                self.file_csv_y[column] = self.encoders_y[column].fit_transform(
                    self.file_csv_y[column].astype(str)
                )
        
        return self.file_csv_x, self.file_csv_y
    
    def _fill_missing_values(self):
        """Заполнение пропущенных значений с помощью RandomForest"""
        def fill_df_missing_values(df):
            for column in df.columns:
                if df[column].isnull().any():
                    mask = df[column].isnull()
                    non_null_data = df[~mask]
                    
                    if len(non_null_data) > 0:
                        features = non_null_data.drop(column, axis=1)
                        target = non_null_data[column]
                        
                        rf = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf.fit(features, target)
                        
                        missing_data = df.loc[mask].drop(column, axis=1)
                        df.loc[mask, column] = rf.predict(missing_data)
            return df

        self.token_x = fill_df_missing_values(self.token_x)
        if not self.token_y.empty:
            self.token_y = fill_df_missing_values(self.token_y)
            
        return self.token_x, self.token_y
    
    def __len__(self):
        return len(self.observation_ids.unique())
    
    def __getitem__(self, idx):
        """
        Получение элемента датасета по индексу
        Args:
            idx: индекс элемента
        Returns:
            dict: словарь с данными элемента
        """
        # Получаем все уникальные observation_ids и их индексы
        unique_obs_ids = self.observation_ids.unique()
        
        # Получаем все строки данных для этого observation_id
        obs_mask = self.observation_ids == unique_obs_ids[idx]
        x_data = self.token_x[obs_mask]
        
        # Получаем все связанные изображения
        patch_filenames = self.filenames[obs_mask].values
        
        max_patches = 10
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        images_tensor = torch.zeros(max_patches, 3, 224, 224)
        
        for i in range(min(max_patches, len(patch_filenames))):
            img = Image.open(f'E:/jupyter/Грибы/images/images/train/300p/{patch_filenames[i]}')
            images_tensor[i] = transform(img)
        
        # Преобразуем x_data в матрицу нужной формы
        num_features = x_data.shape[1]  # Количество признаков
        x_tensor = torch.tensor(x_data.values[0], dtype=torch.float32)
        
        sample = {
            'x_data': x_tensor,
            # 'filenames': patch_filenames.tolist(),
            # 'observation_id': unique_obs_ids[idx],
            # 'images_pil': images_pil,
            'images_tensor': images_tensor
        }
        
        if not self.token_y.empty:
            # Получаем токенизированные метки и category_id
            y_data = self.token_y[obs_mask].values
            category_id = self.category_ids[obs_mask].values
            
            # Убедимся что все тензоры одинакового размера
            max_rows = max(len(y_data), len(category_id))
            
            # Дополним тензоры до одинакового размера
            if len(y_data) < max_rows:
                y_data = np.tile(y_data, (max_rows // len(y_data) + 1, 1))[:max_rows]
            if len(category_id) < max_rows:
                category_id = np.tile(category_id, (max_rows // len(category_id) + 1))[:max_rows]
                
            # Объединяем их
            y_combined = np.hstack([category_id.reshape(-1, 1), y_data])
            y_tensor = torch.tensor(y_combined[0], dtype=torch.int64)
            sample['y_data'] = y_tensor
            
        return sample