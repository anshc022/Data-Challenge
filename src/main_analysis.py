"""
Main Analysis Script for E-Commerce Pricing Optimization
Orchestrates the complete pricing analysis pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import DataPreprocessor
from price_elasticity_analysis import PriceElasticityAnalyzer
from dynamic_pricing_simulation import DynamicPricingSimulator

class PricingAnalysisPipeline:
    def __init__(self, data_path="archive/", results_path="results/"):
        self.data_path = data_path
        self.results_path = results_path
        self.preprocessor = None
        self.analyzer = None
        self.simulator = None
        self.data = None
        
        # Create results directory if it doesn't exist
        os.makedirs(results_path, exist_ok=True)
        
    def run_complete_analysis(self):
        """Run the complete pricing optimization analysis"""
        
        print("="*60)
        print("E-COMMERCE PRICING OPTIMIZATION ANALYSIS")
        print("="*60)
        print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Step 1: Data Preprocessing
        print("STEP 1: DATA PREPROCESSING")
        print("-" * 30)
        self.preprocessor = DataPreprocessor(self.data_path)
        self.data = self.preprocessor.process_all_data()
        
        if self.data is None or len(self.data) == 0:
            print("ERROR: No data available for analysis")
            return None
        
        print(f"✓ Processed {len(self.data)} records across {self.data['category'].nunique()} categories")
        print()
        
        # Step 2: Price Elasticity Analysis
        print("STEP 2: PRICE ELASTICITY ANALYSIS")
        print("-" * 35)
        self.analyzer = PriceElasticityAnalyzer(self.data)
        
        # Overall elasticity
        overall_elasticity = self.analyzer.calculate_basic_elasticity()
        if overall_elasticity:
            print(f"✓ Overall price elasticity: {overall_elasticity['elasticity']:.3f}")
            print(f"  R²: {overall_elasticity['r2']:.3f}, Significant: {overall_elasticity['is_significant']}")
        
        # Category-specific elasticity
        category_elasticities = self.analyzer.analyze_by_category()
        print(f"✓ Analyzed {len(category_elasticities)} product categories")
        
        # Advanced model
        advanced_result = self.analyzer.advanced_elasticity_model()
        if advanced_result:
            print(f"✓ Advanced model R²: {advanced_result['test_r2']:.3f}")
        
        print()
        
        # Step 3: Dynamic Pricing Simulation
        print("STEP 3: DYNAMIC PRICING OPTIMIZATION")
        print("-" * 40)
        self.simulator = DynamicPricingSimulator(self.data, self.analyzer.elasticity_results)
        
        # Calculate profit margins
        profit_data = self.simulator.calculate_profit_margins()
        print("✓ Calculated profit margins")
        
        # Optimize pricing for each category
        optimization_count = 0
        for category in self.data['category'].unique():
            result = self.simulator.optimize_category_pricing(category, target_metric='profit')
            if result:
                optimization_count += 1
                print(f"  - {category}: {result['price_change_pct']:.1f}% price change → {result['profit_change_pct']:.1f}% profit change")
        
        print(f"✓ Optimized pricing for {optimization_count} categories")
        print()
        
        # Step 4: Scenario Analysis
        print("STEP 4: SCENARIO ANALYSIS")
        print("-" * 25)
        
        # Seasonal pricing
        seasonal_results = self.simulator.simulate_seasonal_pricing()
        print(f"✓ Simulated {len(seasonal_results)} seasonal scenarios")
        
        # Competitive response
        competitive_results = self.simulator.simulate_competitive_response()
        print(f"✓ Simulated {len(competitive_results)} competitive scenarios")
        print()
        
        # Step 5: Generate Reports and Visualizations
        print("STEP 5: GENERATING REPORTS")
        print("-" * 28)
        self._generate_reports()
        self._create_visualizations()
        print("✓ Reports and visualizations saved to results/ directory")
        print()
        
        # Step 6: Summary and Recommendations
        print("STEP 6: SUMMARY AND RECOMMENDATIONS")
        print("-" * 38)
        summary = self._generate_executive_summary()
        self._print_summary(summary)
        
        print("="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return {
            'data': self.data,
            'elasticity_results': self.analyzer.elasticity_results,
            'pricing_strategies': self.simulator.pricing_strategies,
            'seasonal_results': seasonal_results,
            'competitive_results': competitive_results,
            'summary': summary
        }
    
    def _generate_reports(self):
        """Generate detailed analysis reports"""
        
        try:
            # Basic summary report
            with open(os.path.join(self.results_path, "analysis_summary.txt"), 'w') as f:
                f.write("E-COMMERCE PRICING OPTIMIZATION ANALYSIS SUMMARY\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset Size: {len(self.data)} records\n")
                f.write(f"Categories Analyzed: {len(self.data['category'].unique())}\n")
                f.write(f"SKUs: {len(self.data['sku'].unique())}\n\n")
                
                # Elasticity results
                if hasattr(self.analyzer, 'elasticity_results') and self.analyzer.elasticity_results:
                    f.write("PRICE ELASTICITY ANALYSIS:\n")
                    if 'overall' in self.analyzer.elasticity_results:
                        overall = self.analyzer.elasticity_results['overall']
                        f.write(f"Overall elasticity: {overall['elasticity']:.3f}\n")
                        f.write(f"R-squared: {overall['r2']:.3f}\n")
                        f.write(f"Significant: {overall['is_significant']}\n\n")
                    
                    category_count = len([k for k in self.analyzer.elasticity_results.keys() if k.startswith('category_')])
                    f.write(f"Categories with elasticity data: {category_count}\n\n")
                
                # Pricing optimization results
                if hasattr(self.simulator, 'pricing_strategies') and self.simulator.pricing_strategies:
                    f.write("PRICING OPTIMIZATION RESULTS:\n")
                    total_categories = len(self.simulator.pricing_strategies)
                    f.write(f"Categories optimized: {total_categories}\n")
                    
                    # Calculate potential profit improvement
                    total_current = sum(s['optimal_strategy']['current_profit'] 
                                      for s in self.simulator.pricing_strategies.values() 
                                      if 'optimal_strategy' in s)
                    total_new = sum(s['optimal_strategy']['new_total_profit'] 
                                  for s in self.simulator.pricing_strategies.values() 
                                  if 'optimal_strategy' in s)
                    
                    if total_current > 0:
                        improvement = (total_new - total_current) / total_current * 100
                        f.write(f"Potential profit improvement: {improvement:.1f}%\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("Analysis completed successfully!\n")
                f.write("For detailed results, see the generated CSV files.\n")
            
            print("✓ Generated analysis summary report")
            
        except Exception as e:
            print(f"Warning: Report generation failed with error: {e}")
            print("Analysis data is still available in the results object.")
        
        try:
            # Try to generate detailed reports if methods exist
            if hasattr(self.analyzer, 'create_elasticity_summary'):
                elasticity_summary = self.analyzer.create_elasticity_summary()
                with open(os.path.join(self.results_path, "elasticity_analysis.txt"), 'w') as f:
                    f.write("PRICE ELASTICITY ANALYSIS REPORT\n")
                    f.write("="*50 + "\n\n")
                    
                    if elasticity_summary.get('overall_analysis'):
                        f.write("OVERALL ANALYSIS:\n")
                        for key, value in elasticity_summary['overall_analysis'].items():
                            f.write(f"  {key}: {value}\n")
                        f.write("\n")
                    
                    if elasticity_summary.get('category_analysis'):
                        f.write("CATEGORY ANALYSIS:\n")
                        for category, analysis in list(elasticity_summary['category_analysis'].items())[:10]:  # Limit to first 10
                            f.write(f"\n{category}:\n")
                            for key, value in analysis.items():
                                f.write(f"  {key}: {value}\n")
                    
                    if elasticity_summary.get('key_insights'):
                        f.write("\nKEY INSIGHTS:\n")
                        for insight in elasticity_summary['key_insights']:
                            f.write(f"• {insight}\n")
                print("✓ Generated elasticity analysis report")
            
        except Exception as e:
            print(f"Warning: Detailed elasticity report failed: {e}")
        
        try:
            if hasattr(self.simulator, 'create_pricing_recommendations'):
                recommendations = self.simulator.create_pricing_recommendations()
                with open(os.path.join(self.results_path, "pricing_recommendations.txt"), 'w') as f:
                    f.write("DYNAMIC PRICING RECOMMENDATIONS\n")
                    f.write("="*50 + "\n\n")
                    
                    if recommendations.get('executive_summary'):
                        f.write("EXECUTIVE SUMMARY:\n")
                        for summary in recommendations['executive_summary']:
                            f.write(f"• {summary}\n")
                        f.write("\n")
                    
                    if recommendations.get('category_recommendations'):
                        f.write("CATEGORY RECOMMENDATIONS:\n")
                        for category, rec in list(recommendations['category_recommendations'].items())[:10]:  # Limit to first 10
                            f.write(f"\n{category}:\n")
                            f.write(f"  Action: {rec.get('action', 'N/A')}\n")
                            f.write(f"  Expected profit change: {rec.get('expected_profit_change', 'N/A')}\n")
                            f.write(f"  Risk level: {rec.get('risk_level', 'N/A')}\n")
                    
                    if recommendations.get('implementation_plan'):
                        f.write("\nIMPLEMENTATION PLAN:\n")
                        for i, step in enumerate(recommendations['implementation_plan'], 1):
                            f.write(f"{i}. {step}\n")
                    
                    if recommendations.get('risk_assessment'):
                        f.write("\nRISK ASSESSMENT:\n")
                        for risk in recommendations['risk_assessment']:
                            f.write(f"• {risk}\n")
                print("✓ Generated pricing recommendations report")
        
        except Exception as e:
            print(f"Warning: Detailed pricing report failed: {e}")
        
        # Export pricing table
        pricing_table = self.simulator.export_pricing_table()
        pricing_table.to_csv(os.path.join(self.results_path, "pricing_optimization_table.csv"), index=False)
        
        # Data summary
        summary_stats = self.data.groupby('category').agg({
            'unit_price': ['mean', 'std', 'min', 'max'],
            'quantity_sold': ['sum', 'mean', 'std'],
            'total_revenue': 'sum'
        }).round(2)
        summary_stats.to_csv(os.path.join(self.results_path, "category_summary_stats.csv"))
    
    def _create_visualizations(self):
        """Create and save visualizations"""
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Elasticity visualizations
        elasticity_fig = self.analyzer.plot_elasticity_results(
            save_path=os.path.join(self.results_path, "elasticity_analysis.png")
        )
        plt.close()
        
        # Optimization visualizations
        optimization_fig = self.simulator.plot_optimization_results(
            save_path=os.path.join(self.results_path, "pricing_optimization.png")
        )
        plt.close()
        
        # Additional custom visualizations
        self._create_dashboard()
    
    def _create_dashboard(self):
        """Create executive dashboard visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Revenue by Category
        category_revenue = self.data.groupby('category')['total_revenue'].sum().sort_values(ascending=True)
        axes[0, 0].barh(category_revenue.index, category_revenue.values)
        axes[0, 0].set_title('Total Revenue by Category')
        axes[0, 0].set_xlabel('Revenue')
        
        # Plot 2: Price Distribution
        axes[0, 1].hist(self.data['unit_price'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Price Distribution')
        axes[0, 1].set_xlabel('Unit Price')
        axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: Quantity vs Price
        axes[0, 2].scatter(self.data['unit_price'], self.data['quantity_sold'], alpha=0.6)
        axes[0, 2].set_title('Quantity vs Price')
        axes[0, 2].set_xlabel('Unit Price')
        axes[0, 2].set_ylabel('Quantity Sold')
        
        # Plot 4: Profit Improvement by Category
        if self.simulator.pricing_strategies:
            categories = list(self.simulator.pricing_strategies.keys())
            improvements = [strategy['optimal_strategy']['profit_change_pct'] 
                          for strategy in self.simulator.pricing_strategies.values()]
            
            colors = ['green' if x > 0 else 'red' for x in improvements]
            axes[1, 0].bar(categories, improvements, color=colors, alpha=0.7)
            axes[1, 0].set_title('Profit Improvement by Category')
            axes[1, 0].set_ylabel('Profit Change (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Elasticity vs Current Price
        if self.analyzer.elasticity_results:
            elasticity_data = []
            price_data = []
            category_names = []
            
            for key, result in self.analyzer.elasticity_results.items():
                if key.startswith('category_'):
                    category = key.replace('category_', '')
                    cat_data = self.data[self.data['category'] == category]
                    if len(cat_data) > 0:
                        elasticity_data.append(result['elasticity'])
                        price_data.append(cat_data['unit_price'].mean())
                        category_names.append(category)
            
            if elasticity_data:
                axes[1, 1].scatter(price_data, elasticity_data, s=100, alpha=0.7)
                axes[1, 1].set_title('Price Elasticity vs Average Price')
                axes[1, 1].set_xlabel('Average Price')
                axes[1, 1].set_ylabel('Price Elasticity')
                axes[1, 1].axhline(y=-1, color='red', linestyle='--', alpha=0.7, label='Unit Elastic')
                axes[1, 1].legend()
        
        # Plot 6: Size Distribution
        if 'size' in self.data.columns:
            size_counts = self.data['size'].value_counts()
            axes[1, 2].pie(size_counts.values, labels=size_counts.index, autopct='%1.1f%%')
            axes[1, 2].set_title('Sales Distribution by Size')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, "executive_dashboard.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_executive_summary(self):
        """Generate executive summary"""
        
        summary = {
            'data_overview': {},
            'key_findings': [],
            'recommendations': [],
            'financial_impact': {}
        }
        
        # Data overview
        summary['data_overview'] = {
            'total_records': len(self.data),
            'categories': self.data['category'].nunique(),
            'skus': self.data['sku'].nunique(),
            'total_revenue': self.data['total_revenue'].sum(),
            'avg_price': self.data['unit_price'].mean(),
            'total_quantity': self.data['quantity_sold'].sum()
        }
        
        # Key findings
        if self.analyzer.elasticity_results:
            avg_elasticity = np.mean([result['elasticity'] for key, result in self.analyzer.elasticity_results.items() 
                                    if key.startswith('category_')])
            summary['key_findings'].append(f"Average price elasticity across categories: {avg_elasticity:.3f}")
            
            elastic_categories = len([result for key, result in self.analyzer.elasticity_results.items() 
                                    if key.startswith('category_') and abs(result['elasticity']) > 1])
            total_categories = len([key for key in self.analyzer.elasticity_results.keys() if key.startswith('category_')])
            
            summary['key_findings'].append(f"{elastic_categories} out of {total_categories} categories show elastic demand")
        
        # Financial impact
        if self.simulator.pricing_strategies:
            total_current_profit = sum([strategy['optimal_strategy']['current_profit'] 
                                      for strategy in self.simulator.pricing_strategies.values()])
            total_optimized_profit = sum([strategy['optimal_strategy']['new_total_profit'] 
                                        for strategy in self.simulator.pricing_strategies.values()])
            
            if total_current_profit > 0:
                improvement = (total_optimized_profit - total_current_profit) / total_current_profit * 100
                summary['financial_impact'] = {
                    'current_profit': total_current_profit,
                    'optimized_profit': total_optimized_profit,
                    'improvement_pct': improvement,
                    'additional_profit': total_optimized_profit - total_current_profit
                }
        
        # Top recommendations
        if self.simulator.pricing_strategies:
            best_opportunities = sorted(
                [(cat, strategy['optimal_strategy']['profit_change_pct']) 
                 for cat, strategy in self.simulator.pricing_strategies.items()],
                key=lambda x: x[1], reverse=True
            )[:3]
            
            for category, improvement in best_opportunities:
                strategy = self.simulator.pricing_strategies[category]['optimal_strategy']
                action = "increase" if strategy['price_change_pct'] > 0 else "decrease"
                summary['recommendations'].append(
                    f"{category}: {action} price by {abs(strategy['price_change_pct']):.1f}% "
                    f"for {improvement:.1f}% profit improvement"
                )
        
        return summary
    
    def _print_summary(self, summary):
        """Print executive summary to console"""
        
        print("DATA OVERVIEW:")
        print(f"  • Total records analyzed: {summary['data_overview']['total_records']:,}")
        print(f"  • Product categories: {summary['data_overview']['categories']}")
        print(f"  • Unique SKUs: {summary['data_overview']['skus']:,}")
        print(f"  • Total revenue: ${summary['data_overview']['total_revenue']:,.2f}")
        print(f"  • Average price: ${summary['data_overview']['avg_price']:.2f}")
        print()
        
        print("KEY FINDINGS:")
        for finding in summary['key_findings']:
            print(f"  • {finding}")
        print()
        
        if summary['financial_impact']:
            print("FINANCIAL IMPACT:")
            print(f"  • Current total profit: ${summary['financial_impact']['current_profit']:,.2f}")
            print(f"  • Optimized total profit: ${summary['financial_impact']['optimized_profit']:,.2f}")
            print(f"  • Improvement: {summary['financial_impact']['improvement_pct']:.1f}%")
            print(f"  • Additional profit: ${summary['financial_impact']['additional_profit']:,.2f}")
            print()
        
        print("TOP RECOMMENDATIONS:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")

def main():
    """Main execution function"""
    
    # Initialize and run analysis
    pipeline = PricingAnalysisPipeline(data_path="archive/", results_path="results/")
    results = pipeline.run_complete_analysis()
    
    if results:
        print(f"\nAnalysis complete! Check the 'results/' directory for detailed outputs.")
        print("Files generated:")
        print("  • elasticity_analysis.txt - Detailed elasticity analysis")
        print("  • pricing_recommendations.txt - Pricing strategy recommendations")
        print("  • pricing_optimization_table.csv - Optimized pricing table")
        print("  • category_summary_stats.csv - Category performance summary")
        print("  • elasticity_analysis.png - Elasticity visualizations")
        print("  • pricing_optimization.png - Optimization visualizations")
        print("  • executive_dashboard.png - Executive dashboard")
        
        return results
    else:
        print("Analysis failed. Please check your data files.")
        return None

if __name__ == "__main__":
    main()
