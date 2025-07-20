"""
Data Preprocessing Module for E-Commerce Pricing Analysis
Handles data cleaning, standardization, and feature engineering
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, data_path="archive/"):
        self.data_path = data_path
        self.sales_data = None
        self.pricing_data = None
        self.international_data = None
        
    def load_data(self):
        """Load all available datasets"""
        print("Loading datasets...")
        
        # Load Sale Report (inventory data)
        sale_report_path = os.path.join(self.data_path, "Sale Report.csv")
        if os.path.exists(sale_report_path):
            self.sales_data = pd.read_csv(sale_report_path)
            print(f"Loaded Sale Report: {self.sales_data.shape}")
        
        # Load May 2022 pricing data
        may_2022_path = os.path.join(self.data_path, "May-2022.csv")
        if os.path.exists(may_2022_path):
            self.pricing_data = pd.read_csv(may_2022_path)
            print(f"Loaded May 2022 pricing data: {self.pricing_data.shape}")
        
        # Load International sale report
        intl_path = os.path.join(self.data_path, "International sale Report.csv")
        if os.path.exists(intl_path):
            self.international_data = pd.read_csv(intl_path)
            print(f"Loaded International sales data: {self.international_data.shape}")
            
    def clean_sales_data(self):
        """Clean and standardize sales data"""
        if self.sales_data is not None:
            # Remove any unnamed columns
            self.sales_data = self.sales_data.loc[:, ~self.sales_data.columns.str.contains('^Unnamed')]
            
            # Clean SKU codes
            self.sales_data['SKU Code'] = self.sales_data['SKU Code'].astype(str).str.strip()
            
            # Clean stock data
            self.sales_data['Stock'] = pd.to_numeric(self.sales_data['Stock'], errors='coerce')
            self.sales_data = self.sales_data.dropna(subset=['Stock'])
            
            print(f"Cleaned sales data shape: {self.sales_data.shape}")
    
    def clean_pricing_data(self):
        """Clean and standardize pricing data"""
        if self.pricing_data is not None:
            # Clean SKU
            self.pricing_data['Sku'] = self.pricing_data['Sku'].astype(str).str.strip()
            
            # Convert pricing columns to numeric
            price_columns = ['TP', 'MRP Old', 'Final MRP Old', 'Ajio MRP', 'Amazon MRP', 
                           'Amazon FBA MRP', 'Flipkart MRP', 'Limeroad MRP', 'Myntra MRP', 
                           'Paytm MRP', 'Snapdeal MRP']
            
            for col in price_columns:
                if col in self.pricing_data.columns:
                    self.pricing_data[col] = pd.to_numeric(self.pricing_data[col], errors='coerce')
            
            # Remove rows with all NaN prices
            self.pricing_data = self.pricing_data.dropna(subset=price_columns, how='all')
            
            print(f"Cleaned pricing data shape: {self.pricing_data.shape}")
    
    def clean_international_data(self):
        """Clean international sales data"""
        if self.international_data is not None:
            # Convert date
            self.international_data['DATE'] = pd.to_datetime(self.international_data['DATE'], 
                                                            format='%d-%m-%y', errors='coerce')
            
            # Clean numeric columns
            numeric_cols = ['PCS', 'RATE', 'GROSS AMT']
            for col in numeric_cols:
                if col in self.international_data.columns:
                    self.international_data[col] = pd.to_numeric(self.international_data[col], errors='coerce')
            
            # Remove invalid data
            self.international_data = self.international_data.dropna(subset=['DATE', 'RATE', 'PCS'])
            
            print(f"Cleaned international data shape: {self.international_data.shape}")
    
    def create_unified_dataset(self):
        """Create a unified dataset for analysis"""
        unified_data = []
        
        # Process international sales data (has actual sales and pricing)
        if self.international_data is not None:
            intl_processed = self.international_data.copy()
            intl_processed['source'] = 'international'
            intl_processed['quantity_sold'] = intl_processed['PCS']
            intl_processed['unit_price'] = intl_processed['RATE']
            intl_processed['total_revenue'] = intl_processed['GROSS AMT']
            intl_processed['sku'] = intl_processed['SKU']
            intl_processed['category'] = intl_processed['Style']
            intl_processed['date'] = intl_processed['DATE']
            
            # Extract size from SKU
            intl_processed['size'] = intl_processed['sku'].str.extract(r'-(S|M|L|XL|XXL|2XL|3XL)$')
            
            unified_data.append(intl_processed[['sku', 'category', 'size', 'quantity_sold', 
                                             'unit_price', 'total_revenue', 'date', 'source']])
        
        # Add pricing data context (for elasticity analysis)
        if self.pricing_data is not None:
            pricing_processed = self.pricing_data.copy()
            
            # Calculate average marketplace price
            price_cols = ['Ajio MRP', 'Amazon MRP', 'Flipkart MRP', 'Myntra MRP']
            pricing_processed['avg_marketplace_price'] = pricing_processed[price_cols].mean(axis=1, skipna=True)
            
            # Create synthetic sales data based on pricing
            # This is a simplification - in real scenario we'd have actual sales data
            np.random.seed(42)
            
            # Calculate lambda values with proper bounds
            lambda_values = 50 - (pricing_processed['avg_marketplace_price'].fillna(2000) - 1000) / 100
            lambda_values = np.maximum(1, np.minimum(lambda_values, 100))  # Bound between 1 and 100
            
            pricing_processed['estimated_monthly_sales'] = np.random.poisson(
                lam=lambda_values, 
                size=len(pricing_processed)
            )
            
            pricing_processed['source'] = 'domestic'
            pricing_processed['sku'] = pricing_processed['Sku']
            pricing_processed['category'] = pricing_processed['Category']
            pricing_processed['quantity_sold'] = pricing_processed['estimated_monthly_sales']
            pricing_processed['unit_price'] = pricing_processed['avg_marketplace_price']
            pricing_processed['total_revenue'] = pricing_processed['quantity_sold'] * pricing_processed['unit_price']
            
            # Extract size
            pricing_processed['size'] = pricing_processed['sku'].str.extract(r'_(S|M|L|XL|2XL|3XL)$')
            
            unified_data.append(pricing_processed[['sku', 'category', 'size', 'quantity_sold', 
                                                 'unit_price', 'total_revenue', 'source']])
        
        if unified_data:
            self.unified_dataset = pd.concat(unified_data, ignore_index=True)
            
            # Clean unified dataset
            self.unified_dataset = self.unified_dataset.dropna(subset=['unit_price', 'quantity_sold'])
            self.unified_dataset = self.unified_dataset[self.unified_dataset['unit_price'] > 0]
            self.unified_dataset = self.unified_dataset[self.unified_dataset['quantity_sold'] > 0]
            
            # Add derived features
            self.unified_dataset['price_per_unit'] = self.unified_dataset['unit_price']
            self.unified_dataset['log_price'] = np.log(self.unified_dataset['price_per_unit'])
            self.unified_dataset['log_quantity'] = np.log(self.unified_dataset['quantity_sold'])
            
            print(f"Created unified dataset: {self.unified_dataset.shape}")
            return self.unified_dataset
        
        return None
    
    def get_category_summary(self):
        """Get summary statistics by category"""
        if hasattr(self, 'unified_dataset') and self.unified_dataset is not None:
            summary = self.unified_dataset.groupby('category').agg({
                'quantity_sold': ['count', 'mean', 'std'],
                'unit_price': ['mean', 'std', 'min', 'max'],
                'total_revenue': ['sum', 'mean']
            }).round(2)
            
            summary.columns = ['_'.join(col) for col in summary.columns]
            return summary
        return None
    
    def process_all_data(self):
        """Main method to process all data"""
        print("Starting data preprocessing...")
        
        self.load_data()
        self.clean_sales_data()
        self.clean_pricing_data()
        self.clean_international_data()
        
        unified_data = self.create_unified_dataset()
        
        if unified_data is not None:
            print(f"Data preprocessing completed successfully!")
            print(f"Final dataset shape: {unified_data.shape}")
            print(f"Categories: {unified_data['category'].nunique()}")
            print(f"SKUs: {unified_data['sku'].nunique()}")
            
            return unified_data
        else:
            print("Failed to create unified dataset")
            return None

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor("../archive/")
    data = preprocessor.process_all_data()
    
    if data is not None:
        print("\nDataset head:")
        print(data.head())
        
        print("\nCategory summary:")
        summary = preprocessor.get_category_summary()
        if summary is not None:
            print(summary)
