"""
Price Elasticity Analysis Module
Estimates demand curves and elasticity coefficients for pricing optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PriceElasticityAnalyzer:
    def __init__(self, data):
        self.data = data.copy()
        self.elasticity_results = {}
        self.models = {}
        
    def calculate_basic_elasticity(self, category=None):
        """Calculate basic price elasticity using log-log regression"""
        
        if category:
            subset = self.data[self.data['category'] == category].copy()
            analysis_name = f"category_{category}"
        else:
            subset = self.data.copy()
            analysis_name = "overall"
        
        if len(subset) < 10:
            print(f"Insufficient data for {analysis_name}")
            return None
        
        # Remove outliers
        Q1_price = subset['log_price'].quantile(0.25)
        Q3_price = subset['log_price'].quantile(0.75)
        IQR_price = Q3_price - Q1_price
        
        Q1_qty = subset['log_quantity'].quantile(0.25)
        Q3_qty = subset['log_quantity'].quantile(0.75)
        IQR_qty = Q3_qty - Q1_qty
        
        subset = subset[
            (subset['log_price'] >= Q1_price - 1.5 * IQR_price) &
            (subset['log_price'] <= Q3_price + 1.5 * IQR_price) &
            (subset['log_quantity'] >= Q1_qty - 1.5 * IQR_qty) &
            (subset['log_quantity'] <= Q3_qty + 1.5 * IQR_qty)
        ]
        
        if len(subset) < 5:
            return None
        
        # Fit log-log regression: log(Q) = a + b*log(P)
        # where b is the price elasticity
        X = subset[['log_price']]
        y = subset['log_quantity']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate metrics
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        
        # Price elasticity is the coefficient
        elasticity = model.coef_[0]
        
        # Statistical significance test
        n = len(subset)
        se = np.sqrt(mse / (n - 2)) / np.sqrt(np.sum((subset['log_price'] - subset['log_price'].mean())**2))
        t_stat = elasticity / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        result = {
            'elasticity': elasticity,
            'intercept': model.intercept_,
            'r2': r2,
            'mse': mse,
            'n_observations': n,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'model': model
        }
        
        self.elasticity_results[analysis_name] = result
        self.models[analysis_name] = model
        
        return result
    
    def analyze_by_category(self):
        """Calculate elasticity for each product category"""
        categories = self.data['category'].value_counts()
        category_results = {}
        
        print("Analyzing price elasticity by category...")
        
        for category in categories.index:
            if categories[category] >= 10:  # Minimum observations
                result = self.calculate_basic_elasticity(category)
                if result:
                    category_results[category] = result
                    print(f"{category}: Elasticity = {result['elasticity']:.3f}, R² = {result['r2']:.3f}")
        
        return category_results
    
    def analyze_by_size(self):
        """Analyze elasticity by product size"""
        size_results = {}
        
        sizes = self.data['size'].value_counts()
        
        for size in sizes.index:
            if pd.notna(size) and sizes[size] >= 10:
                subset = self.data[self.data['size'] == size]
                
                if len(subset) >= 10:
                    X = subset[['log_price']]
                    y = subset['log_quantity']
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)
                    
                    size_results[size] = {
                        'elasticity': model.coef_[0],
                        'r2': r2,
                        'n_observations': len(subset)
                    }
        
        return size_results
    
    def advanced_elasticity_model(self):
        """Create advanced elasticity model with multiple features"""
        
        # Create feature matrix
        features = ['log_price']
        
        # Add categorical features if available
        if 'size' in self.data.columns:
            size_dummies = pd.get_dummies(self.data['size'], prefix='size', dummy_na=True)
            feature_data = pd.concat([self.data[features], size_dummies], axis=1)
        else:
            feature_data = self.data[features]
        
        # Remove any NaN values
        valid_idx = feature_data.notna().all(axis=1) & self.data['log_quantity'].notna()
        X = feature_data[valid_idx]
        y = self.data.loc[valid_idx, 'log_quantity']
        
        if len(X) < 10:
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit ridge regression for regularization
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Extract price elasticity (first coefficient corresponds to log_price)
        price_elasticity = model.coef_[0] / scaler.scale_[0]
        
        result = {
            'price_elasticity': price_elasticity,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'model': model,
            'scaler': scaler,
            'feature_importance': dict(zip(X.columns, model.coef_))
        }
        
        self.elasticity_results['advanced_model'] = result
        
        return result
    
    def predict_demand(self, price, category=None, size=None):
        """Predict demand for given price using elasticity model"""
        
        if category and f"category_{category}" in self.elasticity_results:
            model_key = f"category_{category}"
        else:
            model_key = "overall"
        
        if model_key not in self.elasticity_results:
            return None
        
        result = self.elasticity_results[model_key]
        model = result['model']
        
        log_price = np.log(price)
        log_quantity_pred = model.predict([[log_price]])[0]
        quantity_pred = np.exp(log_quantity_pred)
        
        return quantity_pred
    
    def simulate_price_changes(self, price_changes=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]):
        """Simulate demand changes for different price adjustments"""
        
        simulation_results = []
        
        for category in self.data['category'].unique():
            if f"category_{category}" in self.elasticity_results:
                result = self.elasticity_results[f"category_{category}"]
                elasticity = result['elasticity']
                
                # Calculate average current metrics
                cat_data = self.data[self.data['category'] == category]
                avg_price = cat_data['unit_price'].mean()
                avg_quantity = cat_data['quantity_sold'].mean()
                current_revenue = avg_price * avg_quantity
                
                for price_change in price_changes:
                    new_price = avg_price * (1 + price_change)
                    
                    # Predict new quantity using elasticity
                    quantity_change = price_change * elasticity
                    new_quantity = avg_quantity * (1 + quantity_change)
                    new_revenue = new_price * new_quantity
                    
                    revenue_change = (new_revenue - current_revenue) / current_revenue
                    
                    simulation_results.append({
                        'category': category,
                        'price_change_pct': price_change * 100,
                        'new_price': new_price,
                        'predicted_quantity': new_quantity,
                        'predicted_revenue': new_revenue,
                        'revenue_change_pct': revenue_change * 100,
                        'elasticity': elasticity
                    })
        
        return pd.DataFrame(simulation_results)
    
    def create_elasticity_summary(self):
        """Create comprehensive summary of elasticity analysis"""
        
        summary = {
            'overall_analysis': {},
            'category_analysis': {},
            'key_insights': []
        }
        
        # Overall elasticity
        if 'overall' in self.elasticity_results:
            overall = self.elasticity_results['overall']
            summary['overall_analysis'] = {
                'price_elasticity': overall['elasticity'],
                'r_squared': overall['r2'],
                'significance': 'Significant' if overall['is_significant'] else 'Not significant',
                'interpretation': self._interpret_elasticity(overall['elasticity'])
            }
        
        # Category analysis
        for key, result in self.elasticity_results.items():
            if key.startswith('category_'):
                category = key.replace('category_', '')
                summary['category_analysis'][category] = {
                    'elasticity': result['elasticity'],
                    'r_squared': result['r2'],
                    'n_observations': result['n_observations'],
                    'interpretation': self._interpret_elasticity(result['elasticity'])
                }
        
        # Generate insights
        if summary['category_analysis']:
            elasticities = [v['elasticity'] for v in summary['category_analysis'].values()]
            
            most_elastic_cat = min(summary['category_analysis'].items(), key=lambda x: x[1]['elasticity'])
            least_elastic_cat = max(summary['category_analysis'].items(), key=lambda x: x[1]['elasticity'])
            
            summary['key_insights'] = [
                f"Most price-sensitive category: {most_elastic_cat[0]} (elasticity: {most_elastic_cat[1]['elasticity']:.3f})",
                f"Least price-sensitive category: {least_elastic_cat[0]} (elasticity: {least_elastic_cat[1]['elasticity']:.3f})",
                f"Average elasticity across categories: {np.mean(elasticities):.3f}",
                f"Categories with inelastic demand (|elasticity| < 1): {len([e for e in elasticities if abs(e) < 1])} out of {len(elasticities)}"
            ]
        
        return summary
    
    def _interpret_elasticity(self, elasticity):
        """Interpret elasticity coefficient"""
        abs_elasticity = abs(elasticity)
        
        if abs_elasticity > 1:
            sensitivity = "highly elastic (price-sensitive)"
        elif abs_elasticity > 0.5:
            sensitivity = "moderately elastic"
        else:
            sensitivity = "inelastic (price-insensitive)"
        
        direction = "normal" if elasticity < 0 else "unusual positive"
        
        return f"{sensitivity}, {direction} relationship"
    
    def plot_elasticity_results(self, save_path=None):
        """Create visualization of elasticity results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Price vs Quantity scatter for overall data
        axes[0, 0].scatter(self.data['unit_price'], self.data['quantity_sold'], alpha=0.6)
        axes[0, 0].set_xlabel('Unit Price')
        axes[0, 0].set_ylabel('Quantity Sold')
        axes[0, 0].set_title('Price vs Quantity Relationship')
        
        # Plot 2: Elasticity by category
        if any(key.startswith('category_') for key in self.elasticity_results.keys()):
            categories = []
            elasticities = []
            
            for key, result in self.elasticity_results.items():
                if key.startswith('category_'):
                    categories.append(key.replace('category_', ''))
                    elasticities.append(result['elasticity'])
            
            axes[0, 1].barh(categories, elasticities)
            axes[0, 1].set_xlabel('Price Elasticity')
            axes[0, 1].set_title('Price Elasticity by Category')
            axes[0, 1].axvline(x=-1, color='red', linestyle='--', alpha=0.7, label='Unit Elastic')
            axes[0, 1].legend()
        
        # Plot 3: Log-log relationship
        axes[1, 0].scatter(self.data['log_price'], self.data['log_quantity'], alpha=0.6)
        axes[1, 0].set_xlabel('Log(Price)')
        axes[1, 0].set_ylabel('Log(Quantity)')
        axes[1, 0].set_title('Log-Log Price-Quantity Relationship')
        
        # Plot 4: R-squared by category
        if any(key.startswith('category_') for key in self.elasticity_results.keys()):
            r_squares = []
            for key, result in self.elasticity_results.items():
                if key.startswith('category_'):
                    r_squares.append(result['r2'])
            
            axes[1, 1].barh(categories, r_squares)
            axes[1, 1].set_xlabel('R-squared')
            axes[1, 1].set_title('Model Fit Quality by Category')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig

if __name__ == "__main__":
    # Example usage would go here
    pass
