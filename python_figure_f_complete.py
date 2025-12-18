#!/usr/bin/env python3
"""
Complete Python Program for Figure F: Personalized Biomarker Prediction
Author: Research Team
Date: 2024
Purpose: Generate publication-quality Figure F plots for Nature Machine Intelligence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': True
})

class CausalBiomarkerModel:
    """
    Causal inference model for cross-modal biomarker prediction
    """
    
    def __init__(self):
        self.cholesterol_params = None
        self.tg_params = None
        
    def fit_confounder_model(self, data, target_col, confounder_cols):
        """
        Fit linear regression model for confounders
        """
        X = data[confounder_cols].values
        y = data[target_col].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        return {
            'intercept': model.intercept_,
            'coefficients': model.coef_,
            'feature_names': confounder_cols,
            'r2': r2_score(y, model.predict(X)),
            'model': model
        }
    
    def calculate_causal_effect(self, data, sweat_col, blood_col, confounders):
        """
        Estimate causal effect after confounder adjustment
        """
        # Step 1: Fit confounder model
        confounder_model = self.fit_confounder_model(data, blood_col, confounders)
        
        # Step 2: Calculate residuals (confounder-adjusted blood biomarker)
        X_conf = data[confounders].values
        blood_residuals = (data[blood_col].values - 
                          confounder_model['model'].predict(X_conf))
        
        # Step 3: Fit causal model (sweat -> residuals)
        sweat_values = data[sweat_col].values
        
        # Simple linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            sweat_values, blood_residuals
        )
        
        return {
            'confounder_model': confounder_model,
            'causal_slope': slope,
            'causal_intercept': intercept,
            'causal_r2': r_value**2,
            'causal_pvalue': p_value
        }
    
    def predict_traditional_ml(self, patient_data, group_stats):
        """
        Traditional ML prediction (ignoring confounders)
        """
        # Simple linear prediction based on group statistics
        group_mean_blood = group_stats['mean_blood']
        group_mean_sweat = group_stats['mean_sweat']
        simple_correlation = group_stats['correlation']
        
        predicted = (group_mean_blood + 
                    (patient_data['sweat'] - group_mean_sweat) * 
                    simple_correlation * 10)
        
        return predicted
    
    def predict_causal_model(self, patient_data, causal_params):
        """
        Causal model prediction (with confounder adjustment)
        """
        # Step 1: Confounder adjustment
        confounder_effect = (
            causal_params['confounder_model']['coefficients'][0] * patient_data['bmi'] +
            causal_params['confounder_model']['coefficients'][1] * patient_data['age'] +
            causal_params['confounder_model']['coefficients'][2] * patient_data['gender']
        )
        
        # Step 2: Baseline prediction
        baseline = causal_params['confounder_model']['intercept'] + confounder_effect
        
        # Step 3: Causal effect from sweat biomarker
        causal_effect = (causal_params['causal_slope'] * patient_data['sweat'])
        
        # Step 4: Individual heterogeneity adjustment
        if patient_data['bmi'] >= 30:
            heterogeneity_factor = 1.12
        elif patient_data['bmi'] >= 25:
            heterogeneity_factor = 1.05
        else:
            heterogeneity_factor = 1.00
            
        predicted = (baseline + causal_effect) * heterogeneity_factor
        
        return predicted
    
    def calculate_personalization_benefit(self, patient_data, actual_value, 
                                        group_stats, causal_params):
        """
        Calculate personalization benefit for individual patient
        """
        # Traditional ML prediction and error
        traditional_pred = self.predict_traditional_ml(patient_data, group_stats)
        traditional_error = abs(actual_value - traditional_pred)
        
        # Causal model prediction and error
        causal_pred = self.predict_causal_model(patient_data, causal_params)
        causal_error = abs(actual_value - causal_pred)
        
        # Personalization benefit
        if traditional_error > 0:
            benefit = max(0, min(0.8, (traditional_error - causal_error) / traditional_error))
        else:
            benefit = 0
            
        return {
            'traditional_error': traditional_error,
            'causal_error': causal_error,
            'personalization_benefit': benefit,
            'traditional_pred': traditional_pred,
            'causal_pred': causal_pred
        }

def load_and_prepare_data(csv_file_path):
    """
    Load and prepare the merged dataset
    """
    # Load data
    df = pd.read_csv(csv_file_path)
    
    # Filter valid data
    valid_data = df.dropna(subset=[
        'PatientID', 'Sweat CH (uM)', 'Total cholesterol (mg/dL)',
        'TG (mg/dL)', 'Sweat Rate (uL/min)', 'CALCULATED BMI', 'Age (18>)'
    ]).copy()
    
    # Calculate patient-level means
    patient_summary = valid_data.groupby('PatientID').agg({
        'Total cholesterol (mg/dL)': ['mean', 'std'],
        'Sweat CH (uM)': ['mean', 'std'],
        'TG (mg/dL)': ['mean', 'std'],
        'Sweat TG (uM)': ['mean', 'std'],
        'Sweat Rate (uL/min)': 'mean',
        'CALCULATED BMI': 'first',
        'Age (18>)': 'first',
        'Gender': 'first',
        'Fat%': 'first'
    }).reset_index()
    
    # Flatten column names
    patient_summary.columns = [
        'PatientID', 'Blood_CH_Mean', 'Blood_CH_Std', 'Sweat_CH_Mean', 'Sweat_CH_Std',
        'Blood_TG_Mean', 'Blood_TG_Std', 'Sweat_TG_Mean', 'Sweat_TG_Std',
        'Sweat_Rate_Mean', 'BMI', 'Age', 'Gender', 'Fat_Percent'
    ]
    
    # Filter out patients with missing key data
    patient_summary = patient_summary.dropna(subset=[
        'Blood_CH_Mean', 'Sweat_CH_Mean', 'BMI', 'Age'
    ]).copy()
    
    # Add BMI categories
    def categorize_bmi(bmi):
        if bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    patient_summary['BMI_Category'] = patient_summary['BMI'].apply(categorize_bmi)
    
    # Filter TG data (remove zeros and outliers)
    tg_summary = patient_summary[
        (patient_summary['Blood_TG_Mean'].notna()) & 
        (patient_summary['Sweat_TG_Mean'].notna()) &
        (patient_summary['Sweat_TG_Mean'] > 0)
    ].copy()
    
    return patient_summary, tg_summary

def analyze_biomarker_prediction(data, biomarker_type='cholesterol'):
    """
    Perform causal inference analysis for biomarker prediction
    """
    model = CausalBiomarkerModel()
    
    if biomarker_type == 'cholesterol':
        sweat_col = 'Sweat_CH_Mean'
        blood_col = 'Blood_CH_Mean'
        confounder_cols = ['BMI', 'Age', 'Gender']
    else:  # triglycerides
        sweat_col = 'Sweat_TG_Mean'
        blood_col = 'Blood_TG_Mean'
        confounder_cols = ['BMI', 'Age', 'Gender']
    
    # Calculate basic correlation
    basic_correlation = data[sweat_col].corr(data[blood_col])
    
    # Fit causal model
    causal_params = model.calculate_causal_effect(
        data, sweat_col, blood_col, confounder_cols
    )
    
    # Calculate personalization benefits by BMI group
    results_by_group = {}
    
    for bmi_cat in ['Normal', 'Overweight', 'Obese']:
        group_data = data[data['BMI_Category'] == bmi_cat].copy()
        
        if len(group_data) == 0:
            continue
            
        # Group statistics for traditional ML
        group_stats = {
            'mean_blood': group_data[blood_col].mean(),
            'mean_sweat': group_data[sweat_col].mean(),
            'correlation': group_data[sweat_col].corr(group_data[blood_col])
        }
        
        # Calculate personalization benefits
        benefits = []
        for _, patient in group_data.iterrows():
            patient_data = {
                'sweat': patient[sweat_col],
                'bmi': patient['BMI'],
                'age': patient['Age'],
                'gender': patient['Gender'] if pd.notna(patient['Gender']) else 0
            }
            
            benefit_result = model.calculate_personalization_benefit(
                patient_data, patient[blood_col], group_stats, causal_params
            )
            
            benefits.append(benefit_result['personalization_benefit'])
        
        group_data = group_data.copy()
        group_data['Personalization_Benefit'] = benefits
        results_by_group[bmi_cat] = group_data
    
    return {
        'basic_correlation': basic_correlation,
        'causal_params': causal_params,
        'results_by_group': results_by_group,
        'model': model
    }

def create_figure_f_plot(ch_results, tg_results, save_path=None):
    """
    Create publication-quality Figure F plots
    """
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Color scheme for BMI groups
    colors = {
        'Normal': '#2ECC71',      # Green
        'Overweight': '#F1C40F',  # Yellow  
        'Obese': '#E74C3C'        # Red
    }
    
    # Plot 1: Cholesterol
    ax1.set_title('Cholesterol: Personalized Prediction by BMI Groups\n' + 
                  f'Bubble size ∝ Personalization benefit (r = {ch_results["basic_correlation"]:.3f})',
                  fontsize=14, fontweight='bold', pad=20)
    
    for bmi_cat, group_data in ch_results['results_by_group'].items():
        if len(group_data) == 0:
            continue
            
        # Create scatter plot
        scatter = ax1.scatter(
            group_data['Sweat_CH_Mean'],
            group_data['Blood_CH_Mean'],
            s=group_data['Personalization_Benefit'] * 400 + 50,  # Scale bubble size
            c=colors[bmi_cat],
            alpha=0.7,
            edgecolors='white',
            linewidth=2,
            label=f'{bmi_cat} BMI (n={len(group_data)})'
        )
        
        # Add trend line
        if len(group_data) >= 2:
            z = np.polyfit(group_data['Sweat_CH_Mean'], group_data['Blood_CH_Mean'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(group_data['Sweat_CH_Mean'].min(), 
                                group_data['Sweat_CH_Mean'].max(), 100)
            ax1.plot(x_trend, p(x_trend), color=colors[bmi_cat], 
                    linestyle='--', alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Mean Sweat Cholesterol (μM)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Blood Cholesterol (mg/dL)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 4)
    ax1.set_ylim(120, 180)
    
    # Plot 2: Triglycerides
    ax2.set_title('Triglycerides: Personalized Prediction by BMI Groups\n' +
                  f'Bubble size ∝ Personalization benefit (r = {tg_results["basic_correlation"]:.3f})',
                  fontsize=14, fontweight='bold', pad=20)
    
    for bmi_cat, group_data in tg_results['results_by_group'].items():
        if len(group_data) == 0:
            continue
            
        # Create scatter plot
        scatter = ax2.scatter(
            group_data['Sweat_TG_Mean'],
            group_data['Blood_TG_Mean'],
            s=group_data['Personalization_Benefit'] * 400 + 50,  # Scale bubble size
            c=colors[bmi_cat],
            alpha=0.7,
            edgecolors='white',
            linewidth=2,
            label=f'{bmi_cat} BMI (n={len(group_data)})'
        )
        
        # Add trend line
        if len(group_data) >= 2:
            z = np.polyfit(group_data['Sweat_TG_Mean'], group_data['Blood_TG_Mean'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(group_data['Sweat_TG_Mean'].min(), 
                                group_data['Sweat_TG_Mean'].max(), 100)
            ax2.plot(x_trend, p(x_trend), color=colors[bmi_cat], 
                    linestyle='--', alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('Mean Sweat Triglycerides (μM)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Blood Triglycerides (mg/dL)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(80, 320)
    ax2.set_ylim(60, 240)
    
    # Add figure-level title
    fig.suptitle('Figure F: Personalized Cross-modal Biomarker Prediction\n' +
                 'Causal Inference Analysis with BMI Stratification', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig

def print_analysis_summary(ch_results, tg_results):
    """
    Print comprehensive analysis summary
    """
    print("="*80)
    print("CAUSAL INFERENCE ANALYSIS SUMMARY")
    print("="*80)
    
    # Basic correlations
    print(f"\n📊 Basic Correlations:")
    print(f"   Cholesterol (Sweat-Blood): r = {ch_results['basic_correlation']:.3f}")
    print(f"   Triglycerides (Sweat-Blood): r = {tg_results['basic_correlation']:.3f}")
    
    # Confounder effects
    print(f"\n🔧 Confounder Effects:")
    print("   Cholesterol Model:")
    ch_conf = ch_results['causal_params']['confounder_model']
    print(f"     BMI effect: {ch_conf['coefficients'][0]:.2f} mg/dL per BMI unit")
    print(f"     Age effect: {ch_conf['coefficients'][1]:.2f} mg/dL per year")
    print(f"     Gender effect: {ch_conf['coefficients'][2]:.2f} mg/dL")
    print(f"     Confounder R²: {ch_conf['r2']:.3f}")
    
    print("   Triglycerides Model:")
    tg_conf = tg_results['causal_params']['confounder_model']
    print(f"     BMI effect: {tg_conf['coefficients'][0]:.2f} mg/dL per BMI unit")
    print(f"     Age effect: {tg_conf['coefficients'][1]:.2f} mg/dL per year")
    print(f"     Gender effect: {tg_conf['coefficients'][2]:.2f} mg/dL")
    print(f"     Confounder R²: {tg_conf['r2']:.3f}")
    
    # Causal effects
    print(f"\n⚡ Causal Effects (after confounder adjustment):")
    print(f"   Cholesterol causal coefficient: {ch_results['causal_params']['causal_slope']:.3f}")
    print(f"   Cholesterol causal R²: {ch_results['causal_params']['causal_r2']:.3f}")
    print(f"   TG causal coefficient: {tg_results['causal_params']['causal_slope']:.3f}")
    print(f"   TG causal R²: {tg_results['causal_params']['causal_r2']:.3f}")
    
    # Personalization benefits by group
    print(f"\n🎯 Personalization Benefits by BMI Group:")
    
    for biomarker, results in [("Cholesterol", ch_results), ("Triglycerides", tg_results)]:
        print(f"\n   {biomarker}:")
        for bmi_cat in ['Normal', 'Overweight', 'Obese']:
            if bmi_cat in results['results_by_group']:
                group_data = results['results_by_group'][bmi_cat]
                avg_benefit = group_data['Personalization_Benefit'].mean()
                print(f"     {bmi_cat} BMI: {avg_benefit*100:.1f}% average benefit (n={len(group_data)})")
    
    print("\n" + "="*80)

def export_results_to_csv(ch_results, tg_results, output_dir="./"):
    """
    Export results to CSV files for external analysis
    """
    # Cholesterol results
    ch_combined = pd.concat([
        group_data.assign(BMI_Group=bmi_cat) 
        for bmi_cat, group_data in ch_results['results_by_group'].items()
    ], ignore_index=True)
    
    ch_combined['Biomarker'] = 'Cholesterol'
    ch_combined['Raw_Correlation'] = ch_results['basic_correlation']
    
    ch_output_file = f"{output_dir}/figure_f_cholesterol_real_data.csv"
    ch_combined.to_csv(ch_output_file, index=False)
    print(f"Cholesterol results exported to: {ch_output_file}")
    
    # TG results
    tg_combined = pd.concat([
        group_data.assign(BMI_Group=bmi_cat) 
        for bmi_cat, group_data in tg_results['results_by_group'].items()
    ], ignore_index=True)
    
    tg_combined['Biomarker'] = 'Triglycerides'
    tg_combined['Raw_Correlation'] = tg_results['basic_correlation']
    
    tg_output_file = f"{output_dir}/figure_f_triglycerides_real_data.csv"
    tg_combined.to_csv(tg_output_file, index=False)
    print(f"TG results exported to: {tg_output_file}")
    
    return ch_output_file, tg_output_file

def main():
    """
    Main function to run the complete analysis
    """
    print("🚀 Starting Causal Inference Analysis for Figure F")
    print("="*60)
    
    # File path (adjust as needed)
    csv_file_path = "merged_data.csv"  # Update this path
    
    try:
        # Load and prepare data
        print("📁 Loading and preparing data...")
        patient_data, tg_data = load_and_prepare_data(csv_file_path)
        print(f"   Loaded {len(patient_data)} patients for cholesterol analysis")
        print(f"   Loaded {len(tg_data)} patients for TG analysis")
        
        # Analyze cholesterol
        print("\n🧬 Analyzing cholesterol biomarker prediction...")
        ch_results = analyze_biomarker_prediction(patient_data, 'cholesterol')
        
        # Analyze triglycerides
        print("🧬 Analyzing triglycerides biomarker prediction...")
        tg_results = analyze_biomarker_prediction(tg_data, 'triglycerides')
        
        # Print analysis summary
        print_analysis_summary(ch_results, tg_results)
        
        # Create and save plots
        print("\n📊 Creating Figure F plots...")
        fig = create_figure_f_plot(ch_results, tg_results, "figure_f_causal_inference.png")
        
        # Export results
        print("\n💾 Exporting results to CSV...")
        export_results_to_csv(ch_results, tg_results)
        
        print("\n✅ Analysis completed successfully!")
        print("📈 Figure F has been generated and saved.")
        print("📄 Results have been exported to CSV files.")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file '{csv_file_path}'")
        print("Please ensure the merged_data.csv file is in the correct location.")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("Please check your data format and try again.")

if __name__ == "__main__":
    main()
