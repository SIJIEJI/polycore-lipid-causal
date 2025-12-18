import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_real_data():
    """
    加载您的真实数据
    """
    try:
        # 读取真实数据
        df = pd.read_csv('merged_data.csv')
        print(f"成功加载数据: {len(df)} 行, {len(df.columns)} 列")
        print(f"患者数量: {df['PatientID'].nunique()}")
        
        # 检查关键列是否存在
        required_cols = ['PatientID', 'Sweat CH (uM)', 'Sweat Rate (uL/min)', 
                        'Total cholesterol (mg/dL)', 'Age (18>)', 'Gender', 
                        'CALCULATED BMI', 'HgA1C']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"警告: 缺失以下列: {missing_cols}")
        
        return df
        
    except FileNotFoundError:
        print("错误: 找不到 merged_data.csv 文件")
        return None
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def clean_and_prepare_data(df):
    """
    清洗和准备数据
    """
    # 创建工作副本
    data = df.copy()
    
    # 重命名列以便于使用
    column_mapping = {
        'Sweat CH (uM)': 'SweatCH',
        'Sweat Rate (uL/min)': 'SweatRate', 
        'Total cholesterol (mg/dL)': 'BloodCH',
        'TG (mg/dL)': 'BloodTG',
        'Age (18>)': 'Age',
        'CALCULATED BMI': 'BMI',
        'HgA1C': 'HbA1c',
        'Blood Pressure H': 'BloodPressure_H',
        'Blood Pressure L': 'BloodPressure_L',
        'Fat%': 'FatPercent'
    }
    
    # 只重命名存在的列
    existing_mapping = {k: v for k, v in column_mapping.items() if k in data.columns}
    data = data.rename(columns=existing_mapping)
    
    # 移除缺失关键数据的行
    key_columns = ['SweatCH', 'SweatRate', 'BloodCH', 'PatientID', 'Age', 'Gender', 'BMI']
    available_key_cols = [col for col in key_columns if col in data.columns]
    
    print(f"清洗前数据行数: {len(data)}")
    data = data.dropna(subset=available_key_cols)
    print(f"清洗后数据行数: {len(data)}")
    
    # 填充其他缺失值
    if 'HbA1c' in data.columns:
        data['HbA1c'] = data['HbA1c'].fillna(data['HbA1c'].median())
    if 'BloodPressure_H' in data.columns:
        data['BloodPressure_H'] = data['BloodPressure_H'].fillna(data['BloodPressure_H'].median())
    if 'FatPercent' in data.columns:
        data['FatPercent'] = data['FatPercent'].fillna(data['FatPercent'].median())
    
    return data

def calculate_real_confounding_strength(df):
    """
    基于真实数据计算混杂因素强度
    """
    # 定义可用的混杂因素
    potential_confounders = {
        'BMI': 'BMI',
        'Age': 'Age', 
        'Gender': 'Gender',
        'HbA1c': 'HbA1c',
        'Blood Pressure': 'BloodPressure_H',
        'Fat%': 'FatPercent'
    }
    
    # 检查哪些混杂因素实际可用
    available_confounders = {}
    for name, col in potential_confounders.items():
        if col in df.columns and df[col].notna().sum() > 0:
            available_confounders[name] = col
    
    print(f"可用的混杂因素: {list(available_confounders.keys())}")
    
    # 汗液和血液生物标志物
    sweat_biomarkers = ['SweatCH', 'SweatRate']
    blood_biomarkers = ['BloodCH']
    
    # 如果有甘油三酯数据，也包含进来
    if 'BloodTG' in df.columns:
        blood_biomarkers.append('BloodTG')
    
    confounder_results = []
    
    for conf_name, conf_col in available_confounders.items():
        for sweat_bio in sweat_biomarkers:
            if sweat_bio not in df.columns:
                continue
                
            for blood_bio in blood_biomarkers:
                if blood_bio not in df.columns:
                    continue
                
                # 获取有效数据
                valid_data = df[[conf_col, sweat_bio, blood_bio]].dropna()
                
                if len(valid_data) < 10:  # 需要足够的数据点
                    continue
                
                # 计算相关系数
                try:
                    corr_conf_sweat = valid_data[conf_col].corr(valid_data[sweat_bio])
                    corr_conf_blood = valid_data[conf_col].corr(valid_data[blood_bio])
                    
                    # 混杂强度 = |相关系数的乘积|
                    confounding_strength = abs(corr_conf_sweat * corr_conf_blood)
                    
                    confounder_results.append({
                        'Confounder': conf_name,
                        'SweatBiomarker': sweat_bio,
                        'BloodBiomarker': blood_bio,
                        'Corr_Confounder_Sweat': corr_conf_sweat,
                        'Corr_Confounder_Blood': corr_conf_blood,
                        'ConfoundingStrength': confounding_strength,
                        'SampleSize': len(valid_data)
                    })
                    
                except Exception as e:
                    print(f"计算相关性时出错 ({conf_name}, {sweat_bio}, {blood_bio}): {e}")
                    continue
    
    return pd.DataFrame(confounder_results)

def calculate_real_causal_adjustment_benefit(df):
    """
    基于真实数据计算因果调整收益 - 使用LinearRegression和RandomForest两种模型
    """
    # 确定可用的混杂因素
    potential_confounders = ['BMI', 'Age', 'Gender', 'HbA1c', 'BloodPressure_H', 'FatPercent']
    available_confounders = [col for col in potential_confounders if col in df.columns]
    
    sweat_features = ['SweatCH', 'SweatRate']
    available_sweat = [col for col in sweat_features if col in df.columns]
    
    target = 'BloodCH'
    
    if target not in df.columns or len(available_sweat) == 0:
        print("错误: 缺少必要的目标变量或汗液特征")
        return pd.DataFrame()
    
    benefits = []
    
    for conf in available_confounders:
        try:
            # 获取有效数据
            required_cols = available_sweat + [target, conf]
            valid_data = df[required_cols].dropna()
            
            if len(valid_data) < 20:  # 需要足够的数据进行建模
                print(f"跳过 {conf}: 数据不足 (只有 {len(valid_data)} 个样本)")
                continue
            
            # 准备数据
            X_simple = valid_data[available_sweat]
            y = valid_data[target]
            X_adjusted = valid_data[available_sweat + [conf]]
            
            # === 线性回归模型 ===
            # 模型1: 仅使用汗液指标
            model_lr_simple = LinearRegression()
            model_lr_simple.fit(X_simple, y)
            pred_lr_simple = model_lr_simple.predict(X_simple)
            r2_lr_simple = r2_score(y, pred_lr_simple)
            rmse_lr_simple = np.sqrt(mean_squared_error(y, pred_lr_simple))
            
            # 模型2: 汗液指标 + 当前混杂因素
            model_lr_adjusted = LinearRegression()
            model_lr_adjusted.fit(X_adjusted, y)
            pred_lr_adjusted = model_lr_adjusted.predict(X_adjusted)
            r2_lr_adjusted = r2_score(y, pred_lr_adjusted)
            rmse_lr_adjusted = np.sqrt(mean_squared_error(y, pred_lr_adjusted))
            
            # === 随机森林模型 ===
            # 模型1: 仅使用汗液指标
            model_rf_simple = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
            model_rf_simple.fit(X_simple, y)
            pred_rf_simple = model_rf_simple.predict(X_simple)
            r2_rf_simple = r2_score(y, pred_rf_simple)
            rmse_rf_simple = np.sqrt(mean_squared_error(y, pred_rf_simple))
            
            # 模型2: 汗液指标 + 当前混杂因素
            model_rf_adjusted = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
            model_rf_adjusted.fit(X_adjusted, y)
            pred_rf_adjusted = model_rf_adjusted.predict(X_adjusted)
            r2_rf_adjusted = r2_score(y, pred_rf_adjusted)
            rmse_rf_adjusted = np.sqrt(mean_squared_error(y, pred_rf_adjusted))
            
            # 计算改善程度
            lr_r2_improvement = (r2_lr_adjusted - r2_lr_simple) / max(r2_lr_simple, 0.001)
            rf_r2_improvement = (r2_rf_adjusted - r2_rf_simple) / max(r2_rf_simple, 0.001)
            
            lr_rmse_improvement = (rmse_lr_simple - rmse_lr_adjusted) / rmse_lr_simple
            rf_rmse_improvement = (rmse_rf_simple - rmse_rf_adjusted) / rmse_rf_simple
            
            benefits.append({
                'Confounder': conf,
                # Linear Regression结果
                'LR_R2_Simple': r2_lr_simple,
                'LR_R2_Adjusted': r2_lr_adjusted,
                'LR_R2_Improvement': lr_r2_improvement,
                'LR_RMSE_Simple': rmse_lr_simple,
                'LR_RMSE_Adjusted': rmse_lr_adjusted,
                'LR_RMSE_Improvement': lr_rmse_improvement,
                # Random Forest结果
                'RF_R2_Simple': r2_rf_simple,
                'RF_R2_Adjusted': r2_rf_adjusted,
                'RF_R2_Improvement': rf_r2_improvement,
                'RF_RMSE_Simple': rmse_rf_simple,
                'RF_RMSE_Adjusted': rmse_rf_adjusted,
                'RF_RMSE_Improvement': rf_rmse_improvement,
                # 通用信息
                'AbsoluteImprovement_LR': r2_lr_adjusted - r2_lr_simple,
                'AbsoluteImprovement_RF': r2_rf_adjusted - r2_rf_simple,
                'SampleSize': len(valid_data)
            })
            
            print(f"{conf}:")
            print(f"  线性回归: R²从 {r2_lr_simple:.3f} 到 {r2_lr_adjusted:.3f} (改善 {lr_r2_improvement:.1%})")
            print(f"  随机森林: R²从 {r2_rf_simple:.3f} 到 {r2_rf_adjusted:.3f} (改善 {rf_r2_improvement:.1%})")
            
        except Exception as e:
            print(f"计算 {conf} 的调整收益时出错: {e}")
            continue
    
    return pd.DataFrame(benefits)

def create_real_figure1e_data(df):
    """
    基于真实数据创建Figure 1E数据
    """
    print("\n=== 计算混杂因素强度 ===")
    confounder_analysis = calculate_real_confounding_strength(df)
    
    if confounder_analysis.empty:
        print("错误: 无法计算混杂因素强度")
        return None, None, None, None
    
    print(f"计算了 {len(confounder_analysis)} 个混杂因素组合")
    
    print("\n=== 计算因果调整收益 ===")
    adjustment_benefits = calculate_real_causal_adjustment_benefit(df)
    
    if adjustment_benefits.empty:
        print("错误: 无法计算因果调整收益")
        return None, None, None, None
    
    # 为Figure 1E准备汇总数据
    blood_ch_analysis = confounder_analysis[
        confounder_analysis['BloodBiomarker'] == 'BloodCH'
    ]
    
    if blood_ch_analysis.empty:
        print("警告: 没有找到与血液胆固醇相关的混杂分析")
        blood_ch_analysis = confounder_analysis
    
    # 按混杂因素汇总
    conf_summary = blood_ch_analysis.groupby('Confounder').agg({
        'ConfoundingStrength': 'max',
        'Corr_Confounder_Sweat': 'mean',
        'Corr_Confounder_Blood': 'mean'
    }).reset_index()
    
    # 合并调整收益数据 - 分别为LR和RF创建数据
    figure1e_data_lr = conf_summary.merge(
        adjustment_benefits[['Confounder', 'LR_R2_Improvement', 'AbsoluteImprovement_LR']], 
        on='Confounder',
        how='left'
    )
    
    figure1e_data_rf = conf_summary.merge(
        adjustment_benefits[['Confounder', 'RF_R2_Improvement', 'AbsoluteImprovement_RF']], 
        on='Confounder',
        how='left'
    )
    
    # 填充缺失值并重命名
    figure1e_data_lr['LR_R2_Improvement'] = figure1e_data_lr['LR_R2_Improvement'].fillna(0)
    figure1e_data_lr['AbsoluteImprovement_LR'] = figure1e_data_lr['AbsoluteImprovement_LR'].fillna(0)
    
    figure1e_data_rf['RF_R2_Improvement'] = figure1e_data_rf['RF_R2_Improvement'].fillna(0)
    figure1e_data_rf['AbsoluteImprovement_RF'] = figure1e_data_rf['AbsoluteImprovement_RF'].fillna(0)
    
    # 重命名列
    figure1e_data_lr = figure1e_data_lr.rename(columns={
        'ConfoundingStrength': 'Confounding_Strength',
        'LR_R2_Improvement': 'Causal_Adjustment_Benefit',
        'AbsoluteImprovement_LR': 'Absolute_R2_Improvement'
    })
    
    figure1e_data_rf = figure1e_data_rf.rename(columns={
        'ConfoundingStrength': 'Confounding_Strength',
        'RF_R2_Improvement': 'Causal_Adjustment_Benefit',
        'AbsoluteImprovement_RF': 'Absolute_R2_Improvement'
    })
    
    # 按混杂强度排序
    figure1e_data_lr = figure1e_data_lr.sort_values('Confounding_Strength', ascending=False)
    figure1e_data_rf = figure1e_data_rf.sort_values('Confounding_Strength', ascending=False)
    
    return confounder_analysis, adjustment_benefits, figure1e_data_lr, figure1e_data_rf

def plot_comparison_figure1e_matplotlib(figure1e_data_lr, figure1e_data_rf, save_path='Figure1E_Comparison.png'):
    """
    创建比较LinearRegression和RandomForest的Figure 1E
    """
    if figure1e_data_lr is None or figure1e_data_lr.empty:
        print("错误: 没有数据可以绘图")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # === 左图：Linear Regression ===
    x_pos = np.arange(len(figure1e_data_lr))
    confounders = figure1e_data_lr['Confounder'].values
    
    # 混杂强度 (左Y轴)
    color1 = '#FF9F40'
    bars1 = ax1.bar(x_pos - 0.2, figure1e_data_lr['Confounding_Strength'], 
                    width=0.4, label='Confounding Strength', 
                    color=color1, alpha=0.8, edgecolor='#FF6B35', linewidth=1.5)
    
    ax1.set_xlabel('Confounding Variables', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Confounding Strength', color=color1, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, max(figure1e_data_lr['Confounding_Strength']) * 1.3)
    
    # 因果调整收益 (右Y轴)
    ax1_twin = ax1.twinx()
    color2 = '#4BC0C0'
    bars2 = ax1_twin.bar(x_pos + 0.2, figure1e_data_lr['Causal_Adjustment_Benefit'], 
                        width=0.4, label='Linear Regression Benefit',
                        color=color2, alpha=0.8, edgecolor='#36A2A2', linewidth=1.5)
    
    ax1_twin.set_ylabel('Prediction Improvement (LR)', color=color2, fontsize=12, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    ax1_twin.set_ylim(0, max(figure1e_data_lr['Causal_Adjustment_Benefit']) * 1.3)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(confounders, rotation=45, ha='right', fontsize=10)
    ax1.set_title('Linear Regression Model\nConfounder Analysis', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_facecolor('#fafafa')
    
    # 添加数值标签
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        ax1.annotate(f'{height1:.3f}',
                    xy=(bar1.get_x() + bar1.get_width() / 2, height1),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        if height2 > 0:
            ax1_twin.annotate(f'+{height2:.1%}' if height2 < 1 else f'+{height2:.1f}x',
                            xy=(bar2.get_x() + bar2.get_width() / 2, height2),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # === 右图：Random Forest ===
    x_pos_rf = np.arange(len(figure1e_data_rf))
    confounders_rf = figure1e_data_rf['Confounder'].values
    
    # 混杂强度 (左Y轴)
    bars3 = ax2.bar(x_pos_rf - 0.2, figure1e_data_rf['Confounding_Strength'], 
                    width=0.4, label='Confounding Strength', 
                    color=color1, alpha=0.8, edgecolor='#FF6B35', linewidth=1.5)
    
    ax2.set_xlabel('Confounding Variables', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Confounding Strength', color=color1, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.set_ylim(0, max(figure1e_data_rf['Confounding_Strength']) * 1.3)
    
    # 因果调整收益 (右Y轴)
    ax2_twin = ax2.twinx()
    color3 = '#9B59B6'  # 紫色用于区分Random Forest
    bars4 = ax2_twin.bar(x_pos_rf + 0.2, figure1e_data_rf['Causal_Adjustment_Benefit'], 
                        width=0.4, label='Random Forest Benefit',
                        color=color3, alpha=0.8, edgecolor='#8E44AD', linewidth=1.5)
    
    ax2_twin.set_ylabel('Prediction Improvement (RF)', color=color3, fontsize=12, fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor=color3)
    ax2_twin.set_ylim(0, max(figure1e_data_rf['Causal_Adjustment_Benefit']) * 1.3)
    
    ax2.set_xticks(x_pos_rf)
    ax2.set_xticklabels(confounders_rf, rotation=45, ha='right', fontsize=10)
    ax2.set_title('Random Forest Model\nConfounder Analysis', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('#fafafa')
    
    # 添加数值标签
    for i, (bar3, bar4) in enumerate(zip(bars3, bars4)):
        height3 = bar3.get_height()
        height4 = bar4.get_height()
        
        ax2.annotate(f'{height3:.3f}',
                    xy=(bar3.get_x() + bar3.get_width() / 2, height3),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        if height4 > 0:
            ax2_twin.annotate(f'+{height4:.1%}' if height4 < 1 else f'+{height4:.1f}x',
                            xy=(bar4.get_x() + bar4.get_width() / 2, height4),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 总标题
    fig.suptitle('Confounder Analysis Comparison: Linear Regression vs Random Forest\n' + 
                'Based on Real Patient Data', fontsize=16, fontweight='bold', y=0.98)
    
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    lines3, labels3 = ax2_twin.get_legend_handles_labels()
    
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)
    ax2.legend(lines1 + lines3, labels1 + labels3, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"对比图表已保存至: {save_path}")

def save_comprehensive_results(df, confounder_analysis, adjustment_benefits, 
                             figure1e_data_lr, figure1e_data_rf):
    """
    保存所有分析结果，包括两种模型的对比
    """
    # 保存清洗后的原始数据
    df.to_csv('real_patient_data_cleaned.csv', index=False)
    print("✅ 清洗后的真实患者数据已保存至: real_patient_data_cleaned.csv")
    
    # 保存混杂因素详细分析
    if confounder_analysis is not None and not confounder_analysis.empty:
        confounder_analysis.to_csv('real_confounder_detailed_analysis.csv', index=False)
        print("✅ 混杂因素详细分析已保存至: real_confounder_detailed_analysis.csv")
    
    # 保存因果调整收益 (包含两种模型的结果)
    if adjustment_benefits is not None and not adjustment_benefits.empty:
        adjustment_benefits.to_csv('real_causal_adjustment_benefits_comparison.csv', index=False)
        print("✅ 双模型因果调整收益已保存至: real_causal_adjustment_benefits_comparison.csv")
    
    # 保存Figure 1E绘图数据 - Linear Regression版本
    if figure1e_data_lr is not None and not figure1e_data_lr.empty:
        figure1e_data_lr.to_csv('Figure1E_LinearRegression_plot_data.csv', index=False)
        print("📊 线性回归Figure 1E数据已保存至: Figure1E_LinearRegression_plot_data.csv")
    
    # 保存Figure 1E绘图数据 - Random Forest版本
    if figure1e_data_rf is not None and not figure1e_data_rf.empty:
        figure1e_data_rf.to_csv('Figure1E_RandomForest_plot_data.csv', index=False)
        print("🌲 随机森林Figure 1E数据已保存至: Figure1E_RandomForest_plot_data.csv")
    
    # 创建模型对比汇总
    if (figure1e_data_lr is not None and not figure1e_data_lr.empty and 
        figure1e_data_rf is not None and not figure1e_data_rf.empty):
        
        comparison_data = figure1e_data_lr[['Confounder', 'Confounding_Strength']].copy()
        comparison_data['LR_Benefit'] = figure1e_data_lr['Causal_Adjustment_Benefit']
        comparison_data['RF_Benefit'] = figure1e_data_rf['Causal_Adjustment_Benefit']
        comparison_data['Benefit_Difference'] = comparison_data['RF_Benefit'] - comparison_data['LR_Benefit']
        comparison_data['Better_Model'] = comparison_data['Benefit_Difference'].apply(
            lambda x: 'Random Forest' if x > 0.01 else ('Linear Regression' if x < -0.01 else 'Similar')
        )
        
        comparison_data.to_csv('Model_Comparison_Summary.csv', index=False)
        print("⚖️  模型对比汇总已保存至: Model_Comparison_Summary.csv")
        
        # 显示对比结果预览
        print("\n=== 模型性能对比预览 ===")
        print(comparison_data.round(4))
    
    # 创建详细说明文件
   # description = f"""
# 双模型因果推断分析结果说明

## 分析概览
# - 原始数据: {len(df)} 个样本, {df['PatientID'].nunique()} 名患者
# - 分析模型: Linear Regression vs Random Forest Regressor
# - 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## 文件说明

### 📊 绘图数据文件
# 1. **Figure1E_LinearRegression_plot_data.csv** - 线性回归模型结果
# 2. **Figure1E_RandomForest_plot_data.csv** - 随机森林模型结果
# 3. **Model_Comparison_Summary.csv** - 两模型直接对比

### 📈 详细分析文件
# 4. **real_causal_adjustment_benefits_comparison.csv** - 包含两种模型完整结果
# 5. **real_confounder_detailed_analysis.csv** - 混杂因素详细分析
# 6. **real_patient_data_cleaned.csv** - 清洗后的原始数据

## 主要发现

### 混import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def load_real_data():
    """
    加载您的真实数据
    """
    try:
        # 读取真实数据
        df = pd.read_csv('merged_data.csv')
        print(f"成功加载数据: {len(df)} 行, {len(df.columns)} 列")
        print(f"患者数量: {df['PatientID'].nunique()}")
        
        # 检查关键列是否存在
        required_cols = ['PatientID', 'Sweat CH (uM)', 'Sweat Rate (uL/min)', 
                        'Total cholesterol (mg/dL)', 'Age (18>)', 'Gender', 
                        'CALCULATED BMI', 'HgA1C']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"警告: 缺失以下列: {missing_cols}")
        
        return df
        
    except FileNotFoundError:
        print("错误: 找不到 merged_data.csv 文件")
        return None
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def clean_and_prepare_data(df):
    """
    清洗和准备数据
    """
    # 创建工作副本
    data = df.copy()
    
    # 重命名列以便于使用
    column_mapping = {
        'Sweat CH (uM)': 'SweatCH',
        'Sweat Rate (uL/min)': 'SweatRate', 
        'Total cholesterol (mg/dL)': 'BloodCH',
        'TG (mg/dL)': 'BloodTG',
        'Age (18>)': 'Age',
        'CALCULATED BMI': 'BMI',
        'HgA1C': 'HbA1c',
        'Blood Pressure H': 'BloodPressure_H',
        'Blood Pressure L': 'BloodPressure_L',
        'Fat%': 'FatPercent'
    }
    
    # 只重命名存在的列
    existing_mapping = {k: v for k, v in column_mapping.items() if k in data.columns}
    data = data.rename(columns=existing_mapping)
    
    # 移除缺失关键数据的行
    key_columns = ['SweatCH', 'SweatRate', 'BloodCH', 'PatientID', 'Age', 'Gender', 'BMI']
    available_key_cols = [col for col in key_columns if col in data.columns]
    
    print(f"清洗前数据行数: {len(data)}")
    data = data.dropna(subset=available_key_cols)
    print(f"清洗后数据行数: {len(data)}")
    
    # 填充其他缺失值
    if 'HbA1c' in data.columns:
        data['HbA1c'] = data['HbA1c'].fillna(data['HbA1c'].median())
    if 'BloodPressure_H' in data.columns:
        data['BloodPressure_H'] = data['BloodPressure_H'].fillna(data['BloodPressure_H'].median())
    if 'FatPercent' in data.columns:
        data['FatPercent'] = data['FatPercent'].fillna(data['FatPercent'].median())
    
    return data

def calculate_real_confounding_strength(df):
    """
    基于真实数据计算混杂因素强度
    """
    # 定义可用的混杂因素
    potential_confounders = {
        'BMI': 'BMI',
        'Age': 'Age', 
        'Gender': 'Gender',
        'HbA1c': 'HbA1c',
        'Blood Pressure': 'BloodPressure_H',
        'Fat%': 'FatPercent'
    }
    
    # 检查哪些混杂因素实际可用
    available_confounders = {}
    for name, col in potential_confounders.items():
        if col in df.columns and df[col].notna().sum() > 0:
            available_confounders[name] = col
    
    print(f"可用的混杂因素: {list(available_confounders.keys())}")
    
    # 汗液和血液生物标志物
    sweat_biomarkers = ['SweatCH', 'SweatRate']
    blood_biomarkers = ['BloodCH']
    
    # 如果有甘油三酯数据，也包含进来
    if 'BloodTG' in df.columns:
        blood_biomarkers.append('BloodTG')
    
    confounder_results = []
    
    for conf_name, conf_col in available_confounders.items():
        for sweat_bio in sweat_biomarkers:
            if sweat_bio not in df.columns:
                continue
                
            for blood_bio in blood_biomarkers:
                if blood_bio not in df.columns:
                    continue
                
                # 获取有效数据
                valid_data = df[[conf_col, sweat_bio, blood_bio]].dropna()
                
                if len(valid_data) < 10:  # 需要足够的数据点
                    continue
                
                # 计算相关系数
                try:
                    corr_conf_sweat = valid_data[conf_col].corr(valid_data[sweat_bio])
                    corr_conf_blood = valid_data[conf_col].corr(valid_data[blood_bio])
                    
                    # 混杂强度 = |相关系数的乘积|
                    confounding_strength = abs(corr_conf_sweat * corr_conf_blood)
                    
                    confounder_results.append({
                        'Confounder': conf_name,
                        'SweatBiomarker': sweat_bio,
                        'BloodBiomarker': blood_bio,
                        'Corr_Confounder_Sweat': corr_conf_sweat,
                        'Corr_Confounder_Blood': corr_conf_blood,
                        'ConfoundingStrength': confounding_strength,
                        'SampleSize': len(valid_data)
                    })
                    
                except Exception as e:
                    print(f"计算相关性时出错 ({conf_name}, {sweat_bio}, {blood_bio}): {e}")
                    continue
    
    return pd.DataFrame(confounder_results)

def calculate_real_causal_adjustment_benefit(df):
    """
    基于真实数据计算因果调整收益
    """
    # 确定可用的混杂因素
    potential_confounders = ['BMI', 'Age', 'Gender', 'HbA1c', 'BloodPressure_H', 'FatPercent']
    available_confounders = [col for col in potential_confounders if col in df.columns]
    
    sweat_features = ['SweatCH', 'SweatRate']
    available_sweat = [col for col in sweat_features if col in df.columns]
    
    target = 'BloodCH'
    
    if target not in df.columns or len(available_sweat) == 0:
        print("错误: 缺少必要的目标变量或汗液特征")
        return pd.DataFrame()
    
    benefits = []
    
    for conf in available_confounders:
        try:
            # 获取有效数据
            required_cols = available_sweat + [target, conf]
            valid_data = df[required_cols].dropna()
            
            if len(valid_data) < 20:  # 需要足够的数据进行建模
                print(f"跳过 {conf}: 数据不足 (只有 {len(valid_data)} 个样本)")
                continue
            
            # 模型1: 仅使用汗液指标 
            X_simple = valid_data[available_sweat]
            y = valid_data[target]
            
            model_simple = LinearRegression()
            model_simple.fit(X_simple, y)
            pred_simple = model_simple.predict(X_simple)
            r2_simple = r2_score(y, pred_simple)
            
            # 模型2: 汗液指标 + 当前混杂因素
            X_adjusted = valid_data[available_sweat + [conf]]
            
            model_adjusted = LinearRegression()
            model_adjusted.fit(X_adjusted, y)
            pred_adjusted = model_adjusted.predict(X_adjusted)
            r2_adjusted = r2_score(y, pred_adjusted)
            
            # 计算改善程度
            absolute_improvement = r2_adjusted - r2_simple
            relative_improvement = absolute_improvement / max(r2_simple, 0.001)
            
            benefits.append({
                'Confounder': conf,
                'R2_Simple': r2_simple,
                'R2_Adjusted': r2_adjusted,
                'R2_Improvement': relative_improvement,
                'AbsoluteImprovement': absolute_improvement,
                'SampleSize': len(valid_data)
            })
            
            print(f"{conf}: R²从 {r2_simple:.3f} 提升到 {r2_adjusted:.3f} (改善 {relative_improvement:.1%})")
            
        except Exception as e:
            print(f"计算 {conf} 的调整收益时出错: {e}")
            continue
    
    return pd.DataFrame(benefits)

def create_real_figure1e_data(df):
    """
    基于真实数据创建Figure 1E数据
    """
    print("\n=== 计算混杂因素强度 ===")
    confounder_analysis = calculate_real_confounding_strength(df)
    
    if confounder_analysis.empty:
        print("错误: 无法计算混杂因素强度")
        return None, None, None
    
    print(f"计算了 {len(confounder_analysis)} 个混杂因素组合")
    
    print("\n=== 计算因果调整收益 ===")
    adjustment_benefits = calculate_real_causal_adjustment_benefit(df)
    
    if adjustment_benefits.empty:
        print("错误: 无法计算因果调整收益")
        return None, None, None
    
    # 为Figure 1E准备汇总数据
    # 对于每个混杂因素，取与血液胆固醇相关的最大混杂强度
    blood_ch_analysis = confounder_analysis[
        confounder_analysis['BloodBiomarker'] == 'BloodCH'
    ]
    
    if blood_ch_analysis.empty:
        print("警告: 没有找到与血液胆固醇相关的混杂分析")
        # 使用所有数据
        blood_ch_analysis = confounder_analysis
    
    # 按混杂因素汇总
    conf_summary = blood_ch_analysis.groupby('Confounder').agg({
        'ConfoundingStrength': 'max',  # 取最大混杂强度
        'Corr_Confounder_Sweat': 'mean',  # 平均相关性
        'Corr_Confounder_Blood': 'mean'
    }).reset_index()
    
    # 合并调整收益数据
    figure1e_data = conf_summary.merge(
        adjustment_benefits[['Confounder', 'R2_Improvement', 'AbsoluteImprovement']], 
        on='Confounder',
        how='left'
    )
    
    # 填充缺失的调整收益数据
    figure1e_data['R2_Improvement'] = figure1e_data['R2_Improvement'].fillna(0)
    figure1e_data['AbsoluteImprovement'] = figure1e_data['AbsoluteImprovement'].fillna(0)
    
    # 重命名列
    figure1e_data = figure1e_data.rename(columns={
        'ConfoundingStrength': 'Confounding_Strength',
        'R2_Improvement': 'Causal_Adjustment_Benefit',
        'AbsoluteImprovement': 'Absolute_R2_Improvement'
    })
    
    # 按混杂强度排序
    figure1e_data = figure1e_data.sort_values('Confounding_Strength', ascending=False)
    
    return confounder_analysis, adjustment_benefits, figure1e_data

def plot_real_figure1e_matplotlib(figure1e_data, save_path='Real_Figure1E_matplotlib.png'):
    """
    使用真实数据创建Figure 1E (matplotlib版本)
    """
    if figure1e_data is None or figure1e_data.empty:
        print("错误: 没有数据可以绘图")
        return
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(len(figure1e_data))
    confounders = figure1e_data['Confounder'].values
    
    # 左轴：混杂强度
    color1 = '#FF9F40'
    bars1 = ax1.bar(x_pos - 0.2, figure1e_data['Confounding_Strength'], 
                    width=0.4, label='Confounding Strength', 
                    color=color1, alpha=0.8, edgecolor='#FF6B35', linewidth=1.5)
    
    ax1.set_xlabel('Confounding Variables', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Confounding Strength', color=color1, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, max(figure1e_data['Confounding_Strength']) * 1.3)
    
    # 右轴：因果调整收益
    ax2 = ax1.twinx()
    color2 = '#4BC0C0'
    bars2 = ax2.bar(x_pos + 0.2, figure1e_data['Causal_Adjustment_Benefit'], 
                    width=0.4, label='Causal Adjustment Benefit',
                    color=color2, alpha=0.8, edgecolor='#36A2A2', linewidth=1.5)
    
    ax2.set_ylabel('Prediction Improvement (Fold Change)', color=color2, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(figure1e_data['Causal_Adjustment_Benefit']) * 1.3)
    
    # 设置x轴
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(confounders, rotation=45, ha='right', fontsize=11)
    
    # 添加数值标签
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        # 混杂强度标签
        ax1.annotate(f'{height1:.3f}',
                    xy=(bar1.get_x() + bar1.get_width() / 2, height1),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 调整收益标签
        if height2 > 0:
            ax2.annotate(f'+{height2:.1%}' if height2 < 1 else f'+{height2:.1f}x',
                        xy=(bar2.get_x() + bar2.get_width() / 2, height2),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 图表标题
    plt.title('Confounder Analysis & Causal Adjustment Effects\n' + 
              'Based on Real Patient Data (n=115 samples, 23 patients)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)
    
    # 美化
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_facecolor('#fafafa')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Figure 1E (matplotlib) 已保存至: {save_path}")

def plot_real_figure1e_plotly(figure1e_data, save_path='Real_Figure1E_plotly.html'):
    """
    使用真实数据创建交互式Figure 1E (plotly版本)
    """
    if figure1e_data is None or figure1e_data.empty:
        print("错误: 没有数据可以绘图")
        return
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 混杂强度柱状图
    fig.add_trace(
        go.Bar(
            x=figure1e_data['Confounder'],
            y=figure1e_data['Confounding_Strength'],
            name='Confounding Strength',
            marker=dict(
                color='rgba(255, 159, 64, 0.8)',
                line=dict(color='#FF6B35', width=2)
            ),
            text=[f'{val:.3f}' for val in figure1e_data['Confounding_Strength']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                         'Confounding Strength: %{y:.3f}<br>' +
                         'Corr with Sweat: %{customdata[0]:.3f}<br>' +
                         'Corr with Blood: %{customdata[1]:.3f}<extra></extra>',
            customdata=np.column_stack((
                figure1e_data['Corr_Confounder_Sweat'],
                figure1e_data['Corr_Confounder_Blood']
            ))
        ),
        secondary_y=False,
    )
    
    # 因果调整收益柱状图
    fig.add_trace(
        go.Bar(
            x=figure1e_data['Confounder'],
            y=figure1e_data['Causal_Adjustment_Benefit'],
            name='Causal Adjustment Benefit',
            marker=dict(
                color='rgba(75, 192, 192, 0.8)',
                line=dict(color='#4BC0C0', width=2)
            ),
            text=[f'+{val:.1%}' if val < 1 else f'+{val:.1f}x' 
                  for val in figure1e_data['Causal_Adjustment_Benefit']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                         'R² Improvement: %{y:.1%}<br>' +
                         'Absolute Improvement: %{customdata:.3f}<extra></extra>',
            customdata=figure1e_data['Absolute_R2_Improvement']
        ),
        secondary_y=True,
    )
    
    # 更新布局
    fig.update_xaxes(
        title_text="Confounding Variables", 
        tickangle=-45,
        title_font=dict(size=14, color='black')
    )
    
    fig.update_yaxes(
        title_text="Confounding Strength", 
        secondary_y=False,
        title_font=dict(size=14, color='#FF6B35')
    )
    
    fig.update_yaxes(
        title_text="Prediction Improvement", 
        secondary_y=True,
        title_font=dict(size=14, color='#4BC0C0')
    )
    
    fig.update_layout(
        title={
            'text': 'Confounder Analysis & Causal Adjustment Effects<br>' +
                   '<sub>Based on Real Patient Data (n=115 samples, 23 patients)</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': 'black'}
        },
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        height=600,
        width=1000,
        margin=dict(t=100, b=100, l=80, r=80)
    )
    
    fig.write_html(save_path)
    fig.show()
    
    print(f"Figure 1E (plotly) 已保存至: {save_path}")

def save_real_data_csv(df, confounder_analysis, adjustment_benefits, figure1e_data):
    """
    保存基于真实数据的所有分析结果
    """
    # 保存清洗后的原始数据
    df.to_csv('real_patient_data_cleaned.csv', index=False)
    print("清洗后的真实患者数据已保存至: real_patient_data_cleaned.csv")
    
    # 保存混杂因素详细分析
    if confounder_analysis is not None and not confounder_analysis.empty:
        confounder_analysis.to_csv('real_confounder_detailed_analysis.csv', index=False)
        print("真实数据混杂因素详细分析已保存至: real_confounder_detailed_analysis.csv")
    
    # 保存因果调整收益
    if adjustment_benefits is not None and not adjustment_benefits.empty:
        adjustment_benefits.to_csv('real_causal_adjustment_benefits.csv', index=False)
        print("真实数据因果调整收益已保存至: real_causal_adjustment_benefits.csv")
    
    # 保存Figure 1E绘图数据 - 这是最重要的文件
    if figure1e_data is not None and not figure1e_data.empty:
        figure1e_data.to_csv('Real_Figure1E_plot_data.csv', index=False)
        print("⭐ 真实数据Figure 1E绘图数据已保存至: Real_Figure1E_plot_data.csv")
        
        # 显示数据预览
        print("\n=== Figure 1E 绘图数据预览 ===")
        print(figure1e_data.round(4))
    
    # 创建数据说明文件
    data_description = f"""
# 基于真实数据的 Figure 1E 文件说明

## 数据来源
- 原始数据: merged_data.csv ({len(df)} 个样本, {df['PatientID'].nunique()} 名患者)
- 分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## 关键文件说明

### 1. Real_Figure1E_plot_data.csv ⭐⭐⭐ 最重要
**直接用于绘制Figure 1E的数据**
列说明：
- Confounder: 混杂因素名称 (x轴)
- Confounding_Strength: 混杂强度 (左y轴，橙色柱状图)
- Causal_Adjustment_Benefit: 因果调整收益 (右y轴，蓝绿色柱状图)
- Corr_Confounder_Sweat: 混杂因素与汗液指标的相关性
- Corr_Confounder_Blood: 混杂因素与血液指标的相关性
- Absolute_R2_Improvement: R²的绝对改善值

### 2. real_patient_data_cleaned.csv
清洗后的患者数据，移除了缺失关键变量的样本。

### 3. real_confounder_detailed_analysis.csv
详细的混杂因素分析，包含所有生物标志物组合的相关性。

### 4. real_causal_adjustment_benefits.csv
每个混杂因素的因果调整收益详细分析。

## 绘图说明
使用 Real_Figure1E_plot_data.csv 可以在任何软件中重现Figure 1E：

**图表类型**: 双Y轴柱状图
**X轴**: Confounder (混杂因素名称)
**左Y轴**: Confounding_Strength (橙色，混杂强度)
**右Y轴**: Causal_Adjustment_Benefit (蓝绿色，预测改善)

**建议颜色**:
- 左柱: #FF9F40 (橙色)
- 右柱: #4BC0C0 (蓝绿色)

## 主要发现
{figure1e_data['Confounder'].iloc[0] if not figure1e_data.empty else 'BMI'} 是最强的混杂因素 (强度: {figure1e_data['Confounding_Strength'].iloc[0]:.3f if not figure1e_data.empty else 'N/A'})
"""
    
    with open('real_data_description.txt', 'w', encoding='utf-8') as f:
        f.write(data_description)
    print("真实数据说明文件已保存至: real_data_description.txt")

def main():
    """
    主函数：使用真实数据生成Figure 1E
    """
    print("=== 基于真实数据的 Figure 1E 生成器 ===")
    print("正在分析您的真实患者数据...\n")
    
    # 加载真实数据
    df = load_real_data()
    if df is None:
        return
    
    # 清洗和准备数据
    df_clean = clean_and_prepare_data(df)
    if df_clean.empty:
        print("错误: 清洗后没有可用数据")
        return
    
    # 数据概览
    print(f"\n=== 数据概览 ===")
    print(f"患者数量: {df_clean['PatientID'].nunique()}")
    print(f"有效样本数: {len(df_clean)}")
    print(f"可用列: {list(df_clean.columns)}")
    
    # 基础统计
    if 'SweatCH' in df_clean.columns and 'BloodCH' in df_clean.columns:
        basic_corr = df_clean['SweatCH'].corr(df_clean['BloodCH'])
        print(f"汗液-血液胆固醇基础相关性: {basic_corr:.3f}")
    
    # 生成分析数据
    confounder_analysis, adjustment_benefits, figure1e_data = create_real_figure1e_data(df_clean)
    
    if figure1e_data is None or figure1e_data.empty:
        print("错误: 无法生成Figure 1E数据")
        return
    
    # 保存所有数据
    print("\n=== 保存分析结果 ===")
    save_real_data_csv(df_clean, confounder_analysis, adjustment_benefits, figure1e_data)
    
    # 创建图表
    print("\n=== 生成图表 ===")
    try:
        plot_real_figure1e_matplotlib(figure1e_data)
    except Exception as e:
        print(f"matplotlib绘图出错: {e}")
    
    try:
        plot_real_figure1e_plotly(figure1e_data)
    except Exception as e:
        print(f"plotly绘图出错: {e}")
    
    print("\n=== 完成！===")
    print("🎯 您的同事现在可以使用 'Real_Figure1E_plot_data.csv' 重现Figure 1E")
    print("📊 推荐绘图设置：")
    print("   - X轴: Confounder")
    print("   - 左Y轴 (橙色): Confounding_Strength") 
    print("   - 右Y轴 (蓝绿色): Causal_Adjustment_Benefit")
    print("   - 数据已按混杂强度降序排列")

if __name__ == "__main__":
    main()