"""
Dynamic Pricing Simulation Module
Proposes pricing adjustments and simulates profit outcomes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DynamicPricingSimulator:
    def __init__(self, data, elasticity_results):
        self.data = data.copy()
        self.elasticity_results = elasticity_results
        self.pricing_strategies = {}
        self.simulation_results = {}
        
    def calculate_profit_margins(self, cost_ratio=0.6):
        """Calculate profit margins assuming cost ratio"""
        self.data['estimated_cost'] = self.data['unit_price'] * cost_ratio
        self.data['profit_per_unit'] = self.data['unit_price'] - self.data['estimated_cost']
        self.data['total_profit'] = self.data['profit_per_unit'] * self.data['quantity_sold']
        
        return self.data[['sku', 'category', 'unit_price', 'estimated_cost', 'profit_per_unit', 'total_profit']]
    
    def optimize_category_pricing(self, category, target_metric='profit', constraints=None):
        """Optimize pricing for a specific category"""
        
        if f"category_{category}" not in self.elasticity_results:
            print(f"No elasticity data available for category: {category}")
            return None
        
        elasticity_data = self.elasticity_results[f"category_{category}"]
        elasticity = elasticity_data['elasticity']
        
        # Get current category data
        cat_data = self.data[self.data['category'] == category].copy()
        
        if len(cat_data) == 0:
            return None
        
        # Current metrics
        current_price = cat_data['unit_price'].mean()
        current_quantity = cat_data['quantity_sold'].mean()
        current_cost = cat_data['estimated_cost'].mean() if 'estimated_cost' in cat_data.columns else current_price * 0.6
        current_profit_per_unit = current_price - current_cost
        current_total_profit = current_profit_per_unit * current_quantity
        current_revenue = current_price * current_quantity
        
        # Test different price points
        price_multipliers = np.arange(0.7, 1.5, 0.05)  # 70% to 150% of current price
        optimization_results = []
        
        for multiplier in price_multipliers:
            new_price = current_price * multiplier
            
            # Apply constraints if specified
            if constraints:
                if 'min_price' in constraints and new_price < constraints['min_price']:
                    continue
                if 'max_price' in constraints and new_price > constraints['max_price']:
                    continue
                if 'max_change' in constraints:
                    price_change = abs(multiplier - 1)
                    if price_change > constraints['max_change']:
                        continue
            
            # Calculate demand change using elasticity
            price_change = (new_price - current_price) / current_price
            quantity_change = price_change * elasticity
            new_quantity = current_quantity * (1 + quantity_change)
            
            # Ensure quantity doesn't go negative
            new_quantity = max(0, new_quantity)
            
            # Calculate new metrics
            new_profit_per_unit = new_price - current_cost
            new_total_profit = new_profit_per_unit * new_quantity
            new_revenue = new_price * new_quantity
            
            # Calculate changes
            profit_change = (new_total_profit - current_total_profit) / current_total_profit if current_total_profit > 0 else 0
            revenue_change = (new_revenue - current_revenue) / current_revenue if current_revenue > 0 else 0
            
            optimization_results.append({
                'price_multiplier': multiplier,
                'new_price': new_price,
                'new_quantity': new_quantity,
                'new_profit_per_unit': new_profit_per_unit,
                'new_total_profit': new_total_profit,
                'new_revenue': new_revenue,
                'profit_change_pct': profit_change * 100,
                'revenue_change_pct': revenue_change * 100,
                'price_change_pct': price_change * 100
            })
        
        results_df = pd.DataFrame(optimization_results)
        
        if len(results_df) == 0:
            return None
        
        # Find optimal price point based on target metric
        if target_metric == 'profit':
            optimal_idx = results_df['new_total_profit'].idxmax()
        elif target_metric == 'revenue':
            optimal_idx = results_df['new_revenue'].idxmax()
        else:
            optimal_idx = results_df['new_total_profit'].idxmax()
        
        optimal_strategy = results_df.iloc[optimal_idx].to_dict()
        optimal_strategy['category'] = category
        optimal_strategy['elasticity'] = elasticity
        optimal_strategy['current_price'] = current_price
        optimal_strategy['current_profit'] = current_total_profit
        optimal_strategy['current_revenue'] = current_revenue
        
        self.pricing_strategies[category] = {
            'optimal_strategy': optimal_strategy,
            'all_scenarios': results_df,
            'elasticity': elasticity
        }
        
        return optimal_strategy
    
    def simulate_seasonal_pricing(self, seasonal_factors=None):
        """Simulate pricing across different seasons"""
        
        if seasonal_factors is None:
            seasonal_factors = {
                'Spring': 1.0,
                'Summer': 1.2,  # Peak season
                'Fall': 1.1,
                'Winter': 0.9   # Off season
            }
        
        seasonal_results = {}
        
        for season, demand_factor in seasonal_factors.items():
            season_results = []
            
            for category in self.data['category'].unique():
                if f"category_{category}" in self.elasticity_results:
                    # Adjust base demand by seasonal factor
                    cat_data = self.data[self.data['category'] == category]
                    base_quantity = cat_data['quantity_sold'].mean()
                    seasonal_quantity = base_quantity * demand_factor
                    
                    # Use current pricing strategy if available
                    if category in self.pricing_strategies:
                        optimal_price = self.pricing_strategies[category]['optimal_strategy']['new_price']
                    else:
                        optimal_price = cat_data['unit_price'].mean()
                    
                    # Calculate seasonal metrics
                    estimated_cost = optimal_price * 0.6
                    profit_per_unit = optimal_price - estimated_cost
                    total_profit = profit_per_unit * seasonal_quantity
                    total_revenue = optimal_price * seasonal_quantity
                    
                    season_results.append({
                        'category': category,
                        'season': season,
                        'demand_factor': demand_factor,
                        'seasonal_quantity': seasonal_quantity,
                        'price': optimal_price,
                        'profit_per_unit': profit_per_unit,
                        'total_profit': total_profit,
                        'total_revenue': total_revenue
                    })
            
            seasonal_results[season] = pd.DataFrame(season_results)
        
        self.simulation_results['seasonal'] = seasonal_results
        return seasonal_results
    
    def simulate_competitive_response(self, competitor_actions=None):
        """Simulate pricing in response to competitive actions"""
        
        if competitor_actions is None:
            competitor_actions = {
                'price_war': -0.15,      # Competitors drop prices by 15%
                'premium_positioning': 0.1,  # Competitors raise prices by 10%
                'stable_market': 0.0      # No competitor changes
            }
        
        competitive_results = {}
        
        for scenario, competitor_change in competitor_actions.items():
            scenario_results = []
            
            for category in self.data['category'].unique():
                if f"category_{category}" in self.elasticity_results:
                    elasticity = self.elasticity_results[f"category_{category}"]['elasticity']
                    
                    cat_data = self.data[self.data['category'] == category]
                    current_price = cat_data['unit_price'].mean()
                    current_quantity = cat_data['quantity_sold'].mean()
                    
                    # Calculate our optimal response
                    if competitor_change < 0:  # Competitors lower prices
                        # We might need to match or find our sweet spot
                        our_price_change = competitor_change * 0.7  # Match 70% of competitor reduction
                    elif competitor_change > 0:  # Competitors raise prices
                        # Opportunity to capture market share
                        our_price_change = competitor_change * 0.3  # Small increase to capture value
                    else:  # Stable market
                        our_price_change = 0
                    
                    new_price = current_price * (1 + our_price_change)
                    
                    # Calculate demand response
                    # In competitive scenario, demand is more sensitive
                    effective_elasticity = elasticity * 1.5  # Increased sensitivity
                    quantity_change = our_price_change * effective_elasticity
                    
                    # Additional market share effect from competitive positioning
                    market_share_gain = 0
                    market_share_loss = 0
                    
                    if competitor_change < 0 and our_price_change > competitor_change:
                        # We're more expensive than competitors
                        market_share_loss = -0.1
                    elif competitor_change > 0 and our_price_change < competitor_change:
                        # We're cheaper than competitors
                        market_share_gain = 0.15
                    
                    total_quantity_change = quantity_change + market_share_gain + market_share_loss
                    new_quantity = current_quantity * (1 + total_quantity_change)
                    new_quantity = max(0, new_quantity)
                    
                    # Calculate metrics
                    estimated_cost = current_price * 0.6
                    new_profit_per_unit = new_price - estimated_cost
                    new_total_profit = new_profit_per_unit * new_quantity
                    
                    current_profit = (current_price - estimated_cost) * current_quantity
                    profit_change = (new_total_profit - current_profit) / current_profit if current_profit > 0 else 0
                    
                    scenario_results.append({
                        'category': category,
                        'scenario': scenario,
                        'competitor_change_pct': competitor_change * 100,
                        'our_price_change_pct': our_price_change * 100,
                        'new_price': new_price,
                        'new_quantity': new_quantity,
                        'new_profit': new_total_profit,
                        'profit_change_pct': profit_change * 100,
                        'elasticity': elasticity
                    })
            
            competitive_results[scenario] = pd.DataFrame(scenario_results)
        
        self.simulation_results['competitive'] = competitive_results
        return competitive_results
    
    def create_pricing_recommendations(self):
        """Generate comprehensive pricing recommendations"""
        
        recommendations = {
            'executive_summary': [],
            'category_recommendations': {},
            'implementation_plan': [],
            'risk_assessment': []
        }
        
        # Executive summary
        total_current_profit = 0
        total_optimized_profit = 0
        
        for category, strategy in self.pricing_strategies.items():
            optimal = strategy['optimal_strategy']
            total_current_profit += optimal['current_profit']
            total_optimized_profit += optimal['new_total_profit']
        
        if total_current_profit > 0:
            profit_improvement = (total_optimized_profit - total_current_profit) / total_current_profit * 100
            recommendations['executive_summary'].append(
                f"Dynamic pricing optimization can increase total profit by {profit_improvement:.1f}%"
            )
        
        # Category-specific recommendations
        for category, strategy in self.pricing_strategies.items():
            optimal = strategy['optimal_strategy']
            elasticity = strategy['elasticity']
            
            price_change = optimal['price_change_pct']
            profit_change = optimal['profit_change_pct']
            
            if abs(price_change) < 2:
                action = "maintain current pricing"
            elif price_change > 0:
                action = f"increase prices by {price_change:.1f}%"
            else:
                action = f"decrease prices by {abs(price_change):.1f}%"
            
            recommendations['category_recommendations'][category] = {
                'action': action,
                'expected_profit_change': f"{profit_change:.1f}%",
                'optimal_price': optimal['new_price'],
                'elasticity': elasticity,
                'risk_level': 'High' if abs(price_change) > 10 else 'Medium' if abs(price_change) > 5 else 'Low'
            }
        
        # Implementation plan
        recommendations['implementation_plan'] = [
            "Phase 1: Implement low-risk pricing changes (< 5% adjustment)",
            "Phase 2: A/B test medium-risk changes on subset of products",
            "Phase 3: Monitor competitor responses and market reaction",
            "Phase 4: Roll out high-impact optimizations with demand monitoring"
        ]
        
        # Risk assessment
        high_risk_categories = [cat for cat, rec in recommendations['category_recommendations'].items() 
                              if rec['risk_level'] == 'High']
        
        if high_risk_categories:
            recommendations['risk_assessment'].append(
                f"High-risk categories requiring careful monitoring: {', '.join(high_risk_categories)}"
            )
        
        recommendations['risk_assessment'].extend([
            "Monitor competitor pricing reactions",
            "Track customer satisfaction and retention",
            "Implement gradual rollout to minimize market disruption"
        ])
        
        return recommendations
    
    def plot_optimization_results(self, save_path=None):
        """Visualize pricing optimization results"""
        
        if not self.pricing_strategies:
            print("No pricing strategies to plot. Run optimization first.")
            return None
        
        n_categories = len(self.pricing_strategies)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Optimal price changes by category
        categories = list(self.pricing_strategies.keys())
        price_changes = [strategy['optimal_strategy']['price_change_pct'] 
                        for strategy in self.pricing_strategies.values()]
        
        axes[0, 0].barh(categories, price_changes)
        axes[0, 0].set_xlabel('Price Change (%)')
        axes[0, 0].set_title('Optimal Price Changes by Category')
        axes[0, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 2: Expected profit improvements
        profit_changes = [strategy['optimal_strategy']['profit_change_pct'] 
                         for strategy in self.pricing_strategies.values()]
        
        axes[0, 1].barh(categories, profit_changes)
        axes[0, 1].set_xlabel('Profit Change (%)')
        axes[0, 1].set_title('Expected Profit Improvements')
        axes[0, 1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 3: Price vs Profit curve for first category
        if categories:
            first_category = categories[0]
            scenarios = self.pricing_strategies[first_category]['all_scenarios']
            
            axes[1, 0].plot(scenarios['new_price'], scenarios['new_total_profit'], 'b-', linewidth=2)
            axes[1, 0].set_xlabel('Price')
            axes[1, 0].set_ylabel('Total Profit')
            axes[1, 0].set_title(f'Price-Profit Curve: {first_category}')
            
            # Mark optimal point
            optimal = self.pricing_strategies[first_category]['optimal_strategy']
            axes[1, 0].plot(optimal['new_price'], optimal['new_total_profit'], 'ro', markersize=8, label='Optimal')
            axes[1, 0].legend()
        
        # Plot 4: Elasticity vs Optimal Price Change
        elasticities = [strategy['elasticity'] for strategy in self.pricing_strategies.values()]
        
        axes[1, 1].scatter(elasticities, price_changes, s=100, alpha=0.7)
        axes[1, 1].set_xlabel('Price Elasticity')
        axes[1, 1].set_ylabel('Optimal Price Change (%)')
        axes[1, 1].set_title('Elasticity vs Optimal Price Adjustment')
        
        # Add category labels
        for i, category in enumerate(categories):
            axes[1, 1].annotate(category, (elasticities[i], price_changes[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def export_pricing_table(self):
        """Export optimized pricing recommendations as a table"""
        
        pricing_table = []
        
        for category, strategy in self.pricing_strategies.items():
            optimal = strategy['optimal_strategy']
            
            pricing_table.append({
                'Category': category,
                'Current_Price': optimal['current_price'],
                'Recommended_Price': optimal['new_price'],
                'Price_Change_%': optimal['price_change_pct'],
                'Expected_Profit_Change_%': optimal['profit_change_pct'],
                'Expected_Revenue_Change_%': optimal['revenue_change_pct'],
                'Price_Elasticity': optimal['elasticity'],
                'Risk_Level': 'High' if abs(optimal['price_change_pct']) > 10 else 'Medium' if abs(optimal['price_change_pct']) > 5 else 'Low'
            })
        
        return pd.DataFrame(pricing_table).round(2)

if __name__ == "__main__":
    # Example usage would go here
    pass
