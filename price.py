# 收入计算完整代码
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df_clean = pd.read_csv('D:/data/raw/bike_orders_cleaned.csv')
# 用户行为特征
user_stats = df_clean.groupby('USER_ID').agg({
    'ride_duration': ['count', 'mean', 'std'],
    'distance_km': ['mean', 'std']
}).round(2)
user_stats.columns = ['ride_count', 'avg_duration', 'std_duration', 'avg_distance', 'std_distance']
user_stats = user_stats.reset_index()

# 用户分群
def classify_user(row):
    if row['ride_count'] >= 10:
        return '高频用户'
    elif row['ride_count'] >= 5:
        return '中频用户'
    else:
        return '低频用户'

user_stats['user_segment'] = user_stats.apply(classify_user, axis=1)

class BikeRevenueCalculator:
    def __init__(self, df):
        self.df = df.copy()
        self.revenue_columns = []

    def calculate_ride_revenue(self, pricing_model='standard'):
        """
        计算单笔骑行订单收入
        支持多种定价模型
        """

        def single_ride_revenue(row, model):
            duration = row['ride_duration']
            distance = row['distance_km']

            if model == 'standard':
                # 标准模型：起步价 + 时长费 + 距离费
                start_fee = 1.5  # 起步价
                time_fee = max(0, (duration - 30) / 30) * 1.0  # 30分钟后每30分钟1元
                distance_fee = max(0, (distance - 3)) * 0.5  # 3公里后每公里0.5元
                return start_fee + time_fee + distance_fee

            elif model == 'time_based':
                # 纯时长计费模型
                if duration <= 30:
                    return 1.5
                else:
                    return 1.5 + ((duration - 30) // 15) * 0.5  # 每15分钟0.5元

            elif model == 'simple':
                # 简单模型：固定起步价 + 超时费
                return 1.5 + max(0, duration - 30) * 0.02  # 每分钟2分钱超时费

            elif model == 'premium':
                # 高端模型：考虑不同时段定价
                base_fee = 1.5
                # 高峰时段溢价
                if row['time_period'] in ['早高峰', '晚高峰']:
                    base_fee *= 1.2
                # 周末溢价
                if row['is_weekend'] == 1:
                    base_fee *= 1.1

                time_fee = max(0, (duration - 30) / 30) * 1.2
                return base_fee + time_fee

            elif model == 'realistic':
                # 更现实的模型，基于实际共享单车定价
                # 前15分钟1.5元，之后每15分钟1元
                if duration <= 15:
                    return 1.5
                else:
                    additional_blocks = np.ceil((duration - 15) / 15)
                    return 1.5 + additional_blocks * 1.0

            else:
                return 1.5  # 默认起步价

        # 应用定价模型
        col_name = f'revenue_{pricing_model}'
        self.df[col_name] = self.df.apply(
            lambda x: single_ride_revenue(x, pricing_model), axis=1
        )
        self.revenue_columns.append(col_name)

        total_revenue = self.df[col_name].sum()
        avg_revenue = self.df[col_name].mean()

        print(f"{pricing_model}模型:")
        print(f"  总收入: ¥{total_revenue:,.2f}")
        print(f"  平均订单收入: ¥{avg_revenue:.2f}")
        print(f"  总订单数: {len(self.df):,}")

        return total_revenue, avg_revenue

    def compare_pricing_models(self):
        """比较不同定价模型的收入结果"""
        print("=" * 50)
        print("不同定价模型收入对比")
        print("=" * 50)

        models = ['standard', 'time_based', 'simple', 'premium', 'realistic']
        results = []

        for model in models:
            total_rev, avg_rev = self.calculate_ride_revenue(model)
            results.append({
                'model': model,
                'total_revenue': total_rev,
                'avg_revenue': avg_rev
            })

        # 选择最合理的模型作为主要收入估算
        best_model = max(results, key=lambda x: x['total_revenue'])
        self.df['estimated_revenue'] = self.df[f'revenue_{best_model["model"]}']

        print(f"\n选择 {best_model['model']} 作为主要收入估算模型")
        return pd.DataFrame(results)

    def calculate_subscription_revenue(self, user_stats):
        """
        计算会员套餐收入
        user_stats: 用户行为统计DataFrame
        """
        print("\n" + "=" * 50)
        print("会员套餐收入估算")
        print("=" * 50)

        # 定义会员套餐假设
        pricing_tiers = {
            '月卡套餐': {
                'target_segment': '高频用户',
                'conversion_rate': 0.3,  # 30%转化率
                'monthly_price': 25,
                'months': 1,
                'description': '高频用户月卡'
            },
            '季卡套餐': {
                'target_segment': '中频用户',
                'conversion_rate': 0.15,  # 15%转化率
                'monthly_price': 20,
                'months': 3,
                'description': '中频用户季卡'
            },
            '次卡套餐': {
                'target_segment': '低频用户',
                'conversion_rate': 0.05,  # 5%转化率
                'monthly_price': 15,
                'months': 1,
                'description': '低频用户次卡'
            }
        }

        subscription_revenue = 0
        subscription_details = []

        for tier_name, tier_info in pricing_tiers.items():
            # 获取目标用户群体
            target_users = user_stats[
                user_stats['user_segment'] == tier_info['target_segment']
                ]

            # 估算购买人数
            estimated_buyers = len(target_users) * tier_info['conversion_rate']

            # 计算收入（按3天在月中的比例折算）
            daily_rate = tier_info['monthly_price'] / 30  # 每日费用
            tier_revenue = estimated_buyers * daily_rate * 3  # 3天收入

            subscription_revenue += tier_revenue

            detail = {
                '套餐类型': tier_name,
                '目标用户': tier_info['target_segment'],
                '估算购买人数': int(estimated_buyers),
                '月费': tier_info['monthly_price'],
                '三日收入': tier_revenue
            }
            subscription_details.append(detail)

            print(f"{tier_name}:")
            print(f"  目标用户: {tier_info['target_segment']}")
            print(f"  估算购买: {estimated_buyers:.0f} 人")
            print(f"  三日收入: ¥{tier_revenue:,.2f}")

        # 创建详细DataFrame
        subscription_df = pd.DataFrame(subscription_details)
        total_estimated = subscription_df['估算购买人数'].sum()

        print(f"\n会员套餐汇总:")
        print(f"  总估算购买人数: {total_estimated:.0f}")
        print(f"  会员套餐总收入: ¥{subscription_revenue:,.2f}")

        return subscription_revenue, subscription_df

    def analyze_revenue_breakdown(self):
        """分析收入构成和多维度分布"""
        print("\n" + "=" * 50)
        print("收入多维度分析")
        print("=" * 50)

        analysis_results = {}

        # 1. 按日期分析
        daily_revenue = self.df.groupby('date').agg({
            'estimated_revenue': ['sum', 'mean', 'count'],
            'USER_ID': 'nunique'
        }).round(2)

        daily_revenue.columns = ['日收入', '平均订单价值', '订单量', '独立用户数']
        daily_revenue['单用户价值'] = daily_revenue['日收入'] / daily_revenue['独立用户数']
        analysis_results['daily'] = daily_revenue

        print("每日收入分析:")
        for date, row in daily_revenue.iterrows():
            print(f"  {date}: ¥{row['日收入']:,.2f} ({row['订单量']}单, {row['独立用户数']}用户)")

        # 2. 按时段分析
        hourly_revenue = self.df.groupby('hour').agg({
            'estimated_revenue': ['sum', 'mean', 'count']
        }).round(2)
        hourly_revenue.columns = ['时段总收入', '平均订单价值', '订单量']
        analysis_results['hourly'] = hourly_revenue

        peak_hour = hourly_revenue['时段总收入'].idxmax()
        print(f"\n收入高峰时段: {peak_hour}点 (¥{hourly_revenue.loc[peak_hour, '时段总收入']:,.2f})")

        # 3. 按用户分群分析
        if 'user_segment' in self.df.columns:
            segment_revenue = self.df.groupby('user_segment').agg({
                'estimated_revenue': ['sum', 'mean', 'count'],
                'USER_ID': 'nunique'
            }).round(2)
            segment_revenue.columns = ['分群总收入', '平均订单价值', '总订单量', '用户数']
            segment_revenue['用户终身价值'] = segment_revenue['分群总收入'] / segment_revenue['用户数']
            analysis_results['segment'] = segment_revenue

            print(f"\n用户分群收入贡献:")
            for segment, row in segment_revenue.iterrows():
                contribution = row['分群总收入'] / segment_revenue['分群总收入'].sum() * 100
                print(f"  {segment}: ¥{row['分群总收入']:,.2f} ({contribution:.1f}%)")

        # 4. 按时段类型分析
        if 'time_period' in self.df.columns:
            period_revenue = self.df.groupby('time_period').agg({
                'estimated_revenue': ['sum', 'mean', 'count']
            }).round(2)
            period_revenue.columns = ['时段类型收入', '平均订单价值', '订单量']
            analysis_results['period'] = period_revenue

            print(f"\n时段类型收入:")
            for period, row in period_revenue.iterrows():
                print(f"  {period}: ¥{row['时段类型收入']:,.2f}")

        return analysis_results

    def calculate_unit_economics(self, subscription_revenue=0):
        """计算单位经济效益指标"""
        print("\n" + "=" * 50)
        print("单位经济效益分析")
        print("=" * 50)

        total_ride_revenue = self.df['estimated_revenue'].sum()
        total_business_revenue = total_ride_revenue + subscription_revenue
        total_users = self.df['USER_ID'].nunique()
        total_orders = len(self.df)

        # 付费用户数（有订单的用户）
        paying_users = self.df['USER_ID'].nunique()

        # 关键指标计算
        arpu = total_business_revenue / total_users  # 平均每用户收入
        arppu = total_business_revenue / paying_users  # 付费用户平均收入
        average_order_value = total_ride_revenue / total_orders

        metrics = {
            '总骑行收入': total_ride_revenue,
            '总商业收入': total_business_revenue,
            '总用户数': total_users,
            '付费用户数': paying_users,
            '总订单数': total_orders,
            'ARPU': arpu,
            'ARPPU': arppu,
            '平均订单价值': average_order_value,
            '会员收入占比': (subscription_revenue / total_business_revenue * 100) if total_business_revenue > 0 else 0
        }

        print(f"总骑行收入: ¥{metrics['总骑行收入']:,.2f}")
        print(f"会员套餐收入: ¥{subscription_revenue:,.2f}")
        print(f"总商业收入: ¥{metrics['总商业收入']:,.2f}")
        print(f"总用户数: {metrics['总用户数']:,}")
        print(f"付费用户数: {metrics['付费用户数']:,}")
        print(f"总订单数: {metrics['总订单数']:,}")
        print(f"ARPU (平均每用户收入): ¥{metrics['ARPU']:.2f}")
        print(f"ARPPU (付费用户平均收入): ¥{metrics['ARPPU']:.2f}")
        print(f"平均订单价值: ¥{metrics['平均订单价值']:.2f}")
        print(f"会员收入占比: {metrics['会员收入占比']:.1f}%")

        return metrics

    def identify_revenue_optimization(self, current_total_revenue):
        """识别收入优化机会"""
        print("\n" + "=" * 50)
        print("收入优化机会分析")
        print("=" * 50)

        optimization_opportunities = []

        # 机会1: 提升低频用户转化
        if 'user_segment' in self.df.columns:
            low_freq_users = self.df[self.df['user_segment'] == '低频用户']
            low_freq_revenue = low_freq_users['estimated_revenue'].sum()
            potential_revenue = low_freq_revenue * 0.2  # 提升20%
            optimization_opportunities.append({
                '机会点': '激活低频用户(提升20%)',
                '潜在收入': potential_revenue,
                '提升比例': (potential_revenue / current_total_revenue) * 100
            })

        # 机会2: 高峰时段动态调价
        peak_hours = [7, 8, 9, 17, 18, 19]
        peak_orders = self.df[self.df['hour'].isin(peak_hours)]
        peak_revenue = peak_orders['estimated_revenue'].sum()
        surge_pricing_revenue = peak_revenue * 0.15  # 15%溢价
        optimization_opportunities.append({
            '机会点': '高峰动态调价(15%溢价)',
            '潜在收入': surge_pricing_revenue,
            '提升比例': (surge_pricing_revenue / current_total_revenue) * 100
        })

        # 机会3: 减少车辆空置（提升运营效率）
        additional_revenue = current_total_revenue * 0.15  # 提升15%运营效率
        optimization_opportunities.append({
            '机会点': '调度优化减少空置',
            '潜在收入': additional_revenue,
            '提升比例': 15.0
        })

        # 机会4: 提升会员转化率
        member_revenue = current_total_revenue * 0.10  # 会员收入提升10%
        optimization_opportunities.append({
            '机会点': '提升会员转化率',
            '潜在收入': member_revenue,
            '提升比例': 10.0
        })

        opp_df = pd.DataFrame(optimization_opportunities)
        total_potential = opp_df['潜在收入'].sum()

        print(f"当前总收入: ¥{current_total_revenue:,.2f}")
        print(f"优化后预估收入: ¥{(current_total_revenue + total_potential):,.2f}")
        print(f"总收入提升空间: ¥{total_potential:,.2f} ({total_potential / current_total_revenue * 100:.1f}%)")
        print("\n具体优化机会:")
        for _, opp in opp_df.iterrows():
            print(f"- {opp['机会点']}: +¥{opp['潜在收入']:,.2f} (+{opp['提升比例']:.1f}%)")

        return opp_df

    def create_revenue_visualizations(self, analysis_results, unit_metrics, subscription_revenue=0):
        """创建收入分析可视化图表"""
        print("\n生成收入分析图表...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('共享单车收入分析仪表板', fontsize=16, fontweight='bold')

        # 1. 日收入趋势
        if 'daily' in analysis_results:
            daily_data = analysis_results['daily']
            axes[0, 0].plot(daily_data.index.astype(str), daily_data['日收入'],
                            marker='o', linewidth=2, color='#2E86AB', markersize=8)
            axes[0, 0].set_title('日收入趋势', fontweight='bold', fontsize=12)
            axes[0, 0].set_ylabel('收入(元)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)

        # 2. 时段收入分布
        if 'hourly' in analysis_results:
            hourly_data = analysis_results['hourly']
            axes[0, 1].bar(hourly_data.index, hourly_data['时段总收入'],
                           color='#A23B72', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('24小时收入分布', fontweight='bold', fontsize=12)
            axes[0, 1].set_xlabel('小时')
            axes[0, 1].set_ylabel('收入(元)')
            axes[0, 1].grid(True, alpha=0.3)

        # 3. 用户分群收入贡献
        if 'segment' in analysis_results:
            segment_data = analysis_results['segment']
            labels = segment_data.index
            sizes = segment_data['分群总收入']
            colors = ['#F18F01', '#C73E1D', '#3E92CC', '#4CB963']
            axes[0, 2].pie(sizes, labels=labels, autopct='%1.1f%%',
                           colors=colors[:len(labels)], startangle=90)
            axes[0, 2].set_title('用户分群收入贡献', fontweight='bold', fontsize=12)

        # 4. 订单价值分布
        # 重点1：缩小x轴范围（比如0到20元，覆盖“几块钱”的订单）
        x_range = (0, 20)
        # 重点2：减少分箱数（比如10个区间，避免区间过细）
        bins = 10

        # 绘制直方图（只显示x轴范围内的数据）
        filtered_revenue = self.df['estimated_revenue'][
            (self.df['estimated_revenue'] >= x_range[0]) & (self.df['estimated_revenue'] <= x_range[1])]
        axes[1, 0].hist(filtered_revenue, bins=bins, color='#4CB963', alpha=0.7, edgecolor='black')

        # 计算并绘制平均值
        mean_revenue = filtered_revenue.mean()
        axes[1, 0].axvline(mean_revenue, color='red', linestyle='--', label=f'平均: {mean_revenue:.2f}')

        # 调整x轴、y轴范围
        axes[1, 0].set_xlim(x_range)  # x轴限制在0-20元
        axes[1, 0].set_ylim(0, filtered_revenue.value_counts().max() * 1.2)  # y轴适配订单数量

        # 标题、标签保持不变
        axes[1, 0].set_title('订单价值分布', fontweight='bold', fontsize=12)
        axes[1, 0].set_xlabel('订单价值(元)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. 收入构成分析
        ride_revenue = unit_metrics['总骑行收入']
        total_revenue = unit_metrics['总商业收入']

        revenue_breakdown = {
            '骑行收入': ride_revenue,
            '会员收入': subscription_revenue
        }

        axes[1, 1].bar(revenue_breakdown.keys(), revenue_breakdown.values(),
                       color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('收入构成分析', fontweight='bold', fontsize=12)
        axes[1, 1].set_ylabel('收入(元)')

        # 在柱状图上添加数值标签
        for i, v in enumerate(revenue_breakdown.values()):
            axes[1, 1].text(i, v, f'¥{v:,.0f}',
                            ha='center', va='bottom', fontweight='bold')

        # 6. 单位经济指标
        metrics_to_show = {
            'ARPU': unit_metrics['ARPU'],
            'ARPPU': unit_metrics['ARPPU'],
            '平均订单价值': unit_metrics['平均订单价值']
        }

        axes[1, 2].bar(metrics_to_show.keys(), metrics_to_show.values(),
                       color=['#45B7D1', '#96CEB4', '#FEEAA5'], alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('单位经济指标', fontweight='bold', fontsize=12)
        axes[1, 2].set_ylabel('金额(元)')
        axes[1, 2].tick_params(axis='x', rotation=45)

        # 在柱状图上添加数值标签
        for i, v in enumerate(metrics_to_show.values()):
            axes[1, 2].text(i, v, f'¥{v:.2f}',
                            ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('收入分析图表.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("图表已保存为 '收入分析图表.png'")

    def generate_revenue_report(self, subscription_revenue, subscription_df,
                                analysis_results, unit_metrics, optimization_df):
        """生成完整的收入分析报告"""
        print("\n" + "=" * 50)
        print("生成收入分析报告")
        print("=" * 50)

        total_ride_revenue = unit_metrics['总骑行收入']
        total_business_revenue = unit_metrics['总商业收入']

        report = f"""
# 🚴 深圳市共享单车劳动节收入分析报告

## 📊 执行摘要

基于劳动节三天共享单车订单数据的深度分析，本项目估算了运营收入并识别了关键优化机会。

### 核心发现
- **预估总收入**: ¥{total_business_revenue:,.2f} (骑行: ¥{total_ride_revenue:,.2f} + 会员: ¥{subscription_revenue:,.2f})
- **平均订单价值**: ¥{unit_metrics['平均订单价值']:.2f}
- **用户价值**: ARPU ¥{unit_metrics['ARPU']:.2f} | ARPPU ¥{unit_metrics['ARPPU']:.2f}
- **优化潜力**: 通过四项关键措施可提升收入 {optimization_df['提升比例'].sum():.1f}%

## 💰 详细收入分析

### 收入构成
| 收入类型 | 金额(元) | 占比 |
|---------|----------|------|
| 骑行订单收入 | ¥{total_ride_revenue:,.2f} | {(total_ride_revenue / total_business_revenue * 100):.1f}% |
| 会员套餐收入 | ¥{subscription_revenue:,.2f} | {(subscription_revenue / total_business_revenue * 100):.1f}% |
| **总计** | **¥{total_business_revenue:,.2f}** | **100%** |

### 会员套餐详情
"""

        # 添加会员套餐详情
        for _, sub in subscription_df.iterrows():
            report += f"- **{sub['套餐类型']}**: {sub['估算购买人数']}用户 × ¥{sub['三日收入'] / sub['估算购买人数']:.2f} = ¥{sub['三日收入']:,.2f}\n"

        # 添加时间分析
        if 'daily' in analysis_results:
            daily_data = analysis_results['daily']
            best_day = daily_data.loc[daily_data['日收入'].idxmax()]
            report += f"""
### 时间分布特征
- **最高收入日**: {daily_data['日收入'].idxmax()} (¥{best_day['日收入']:,.2f})
- **日均收入**: ¥{daily_data['日收入'].mean():,.2f}
- **收入波动率**: {(daily_data['日收入'].std() / daily_data['日收入'].mean() * 100):.1f}%
"""

        # 添加用户分析
        if 'segment' in analysis_results:
            segment_data = analysis_results['segment']
            report += f"""
### 用户价值分层
| 用户类型 | 收入贡献 | 平均订单价值 | 用户价值 |
|---------|----------|-------------|----------|
"""
            for segment, row in segment_data.iterrows():
                contribution = row['分群总收入'] / segment_data['分群总收入'].sum() * 100
                report += f"| {segment} | {contribution:.1f}% | ¥{row['平均订单价值']:.2f} | ¥{row['用户终身价值']:.2f} |\n"

        # 添加优化建议
        report += f"""
## 🎯 收入优化机会

预计通过实施以下措施，可在现有基础上提升 **{optimization_df['提升比例'].sum():.1f}%** 的收入:

| 优化措施 | 潜在收入(元) | 提升幅度 |
|----------|-------------|----------|
"""
        for _, opp in optimization_df.iterrows():
            report += f"| {opp['机会点']} | +¥{opp['潜在收入']:,.2f} | +{opp['提升比例']:.1f}% |\n"

        report += f"""
**预期优化后收入**: ¥{(total_business_revenue + optimization_df['潜在收入'].sum()):,.2f}

## 📈 战略建议

### 短期行动 (1-3个月)
1. **实施高峰动态定价**: 在早晚高峰实施10-15%的价格溢价
2. **优化车辆调度**: 基于热点分析重新分配车辆，减少空置率
3. **启动用户激活活动**: 针对低频用户推出首单优惠

### 中期计划 (3-12个月)  
1. **建立会员体系**: 推出差异化套餐，提升用户粘性
2. **数据驱动定价**: 基于历史数据建立更精细的定价模型
3. **拓展服务场景**: 增加景区、商圈等特色服务

### 长期战略 (1年以上)
1. **生态体系建设**: 整合其他出行服务，打造综合出行平台
2. **国际化扩张**: 将成功模式复制到其他城市
3. **技术升级**: 引入AI调度和预测系统

## ⚠️ 风险提示

1. **政策风险**: 共享单车行业受政策影响较大
2. **竞争压力**: 需要持续创新保持竞争优势  
3. **季节性波动**: 收入受天气和季节因素影响
4. **用户留存**: 需要持续投入维持用户活跃度

---
*报告生成时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}*
*数据来源: 深圳市政府数据开放平台*
*分析周期: 劳动节三日订单数据*
"""

        # 保存报告
        with open('共享单车收入分析报告.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("完整收入分析报告已保存为 '共享单车收入分析报告.md'")
        return report


# 使用示例
def main():
    """
    收入计算主函数
    假设df_clean是已经清洗好的数据，包含以下字段：
    - ride_duration: 骑行时长(分钟)
    - distance_km: 骑行距离(公里)
    - hour: 开始小时
    - date: 日期
    - time_period: 时段分类
    - is_weekend: 是否周末
    - user_segment: 用户分群
    - USER_ID: 用户ID
    """

    # 初始化计算器
    calculator = BikeRevenueCalculator(df_clean)

    # 1. 比较不同定价模型
    model_comparison = calculator.compare_pricing_models()

    # 2. 计算会员收入（需要用户统计数据）
    # 假设我们已经有了user_stats DataFrame
    subscription_revenue, subscription_df = calculator.calculate_subscription_revenue(user_stats)

    # 3. 多维度收入分析
    revenue_analysis = calculator.analyze_revenue_breakdown()

    # 4. 单位经济效益分析
    unit_metrics = calculator.calculate_unit_economics(subscription_revenue)

    # 5. 识别优化机会
    current_total_revenue = unit_metrics['总商业收入']
    optimization_df = calculator.identify_revenue_optimization(current_total_revenue)

    # 6. 创建可视化
    calculator.create_revenue_visualizations(revenue_analysis, unit_metrics, subscription_revenue)

    # 7. 生成完整报告
    final_report = calculator.generate_revenue_report(
        subscription_revenue, subscription_df, revenue_analysis,
        unit_metrics, optimization_df
    )

    # 8. 保存关键数据供Power BI使用
    powerbi_data = {
        '收入KPI': pd.DataFrame([unit_metrics]),
        '每日收入': revenue_analysis['daily'].reset_index(),
        '优化机会': optimization_df,
        '会员详情': subscription_df
    }

    with pd.ExcelWriter('收入分析_PowerBI数据.xlsx') as writer:
        for sheet_name, data in powerbi_data.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    print("\nPower BI数据已保存为 '收入分析_PowerBI数据.xlsx'")

    return {
        'calculator': calculator,
        'model_comparison': model_comparison,
        'subscription_revenue': subscription_revenue,
        'revenue_analysis': revenue_analysis,
        'unit_metrics': unit_metrics,
        'optimization_df': optimization_df,
        'final_report': final_report
    }


# 如果直接运行此文件，执行示例
if __name__ == "__main__":
    # 这里需要先准备好df_clean和user_stats
    # results = main()
    print("收入计算模块已加载，请调用main()函数运行完整分析")

main()