# Context 通道语义核查报告

> 生成日期: 2026-04-15
> 数据源: `D:\bishe\WYB\{temperature,humidity,cloud_cover,component_of_wind}\trn.pkl`
> context shape: `(2300, 12, 2048, 8)` — 四个数据集完全一致

---

## 1. 各通道统计特征与语义判定

| Index | 语义 | min | max | mean | std | n_unique | 时间维恒定 | 站点维恒定 | 判断依据 |
|-------|------|-----|-----|------|-----|----------|-----------|-----------|----------|
| 0 | **year** | 2010 | 2016 | 2012.67 | 1.82 | 7 | ✅ | ✅ | 整数年份，所有站点同时刻一致 |
| 1 | **month** | 1 | 12 | 6.33 | 3.49 | 12 | ✅ | ✅ | 1-12 月份编码 |
| 2 | **day** | 1 | 31 | 15.68 | 8.79 | 31 | ✅ | ✅ | 1-31 日期编码 |
| 3 | **hour** | 0 | 11 | 5.50 | 3.45 | 12 | ❌ | ✅ | 沿时间步递增(0,1,2,...11)，唯一随时间变化的通道 |
| 4 | **region** | 0.0 | 1.0 | 0.34 | 0.46 | 524 | ✅ | ❌ | 空间静态，值域[0,1]，许多接近0的极小值(编码/归一化区域) |
| 5 | **altitude** | -28.95 | 4910.14 | 379.50 | 859.87 | 1828 | ✅ | ❌ | 空间静态，米为单位的海拔高度 |
| 6 | **latitude** | -87.19 | 87.19 | 0.00 | 51.94 | 32 | ✅ | ❌ | 空间静态，等间距纬度网格(5.625°步长) |
| 7 | **longitude** | 0.0 | 354.38 | 177.19 | 103.91 | 64 | ✅ | ❌ | 空间静态，等间距经度网格(5.625°步长) |

### 关键特征区分

- **时间特征** (0-3): channel 0/1/2 在同一样本内所有时间步相同; channel 3 沿时间步递增
- **空间特征** (4-7): 同一站点跨样本恒定, 不同站点取值不同
- **channel 4 (region)**: 值域 [0,1]，大量极小值(~1e-15)，少数值为1.0，分布极度偏斜 → 归一化/编码的区域特征

---

## 2. 四个数据集一致性

| 数据集 | context shape | 通道统计 | 结论 |
|--------|--------------|---------|------|
| temperature | (2300, 12, 2048, 8) | 与上表一致 | ✅ |
| humidity | (2300, 12, 2048, 8) | 与上表一致 | ✅ |
| cloud_cover | (2300, 12, 2048, 8) | 与上表一致 | ✅ |
| wind | (2300, 12, 2048, 8) | 与上表一致 | ✅ |

**结论: 四个数据集 context 通道顺序完全相同。**

---

## 3. 当前代码错位分析

### 代码中 mask 构造（train.py 第 102-110 行）:

```python
context_feature_mask = [
    context_features.get('use_longitude', True),   # mask[0] → channel 0 = YEAR    ❌
    context_features.get('use_latitude', True),    # mask[1] → channel 1 = MONTH   ❌
    context_features.get('use_altitude', True),    # mask[2] → channel 2 = DAY     ❌
    context_features.get('use_year', True),        # mask[3] → channel 3 = HOUR    ❌
    context_features.get('use_month', True),       # mask[4] → channel 4 = REGION  ❌
    context_features.get('use_day', True),         # mask[5] → channel 5 = ALTITUDE ❌
    context_features.get('use_hour', True),        # mask[6] → channel 6 = LATITUDE ❌
    context_features.get('use_region', True),      # mask[7] → channel 7 = LONGITUDE ❌
]
```

### 配置文件 (config.yaml):

```yaml
context_features:
    use_longitude: false  # 用户以为关闭经度，实际关闭了 year (channel 0)
    use_latitude: false   # 用户以为关闭纬度，实际关闭了 month (channel 1)
    use_altitude: true    # 用户以为开启海拔，实际开启了 day (channel 2)
    use_year: false       # 用户以为关闭年份，实际关闭了 hour (channel 3)
    use_month: true       # 用户以为开启月份，实际开启了 region (channel 4)
    use_day: false        # 用户以为关闭日期，实际关闭了 altitude (channel 5)
    use_hour: true        # 用户以为开启小时，实际开启了 latitude (channel 6)
    use_region: false     # 用户以为关闭区域，实际关闭了 longitude (channel 7)
```

### 实际效果

用户意图选择: **altitude, month, hour** (3 个通道)
模型实际收到: **day, region, latitude** (3 个完全不同的通道)

**8 个通道的名称-索引映射全部错位，严重影响模型训练效果。**

---

## 4. 修正方案

将 mask 构造顺序改为与真实数据一致:

```
index 0 → year
index 1 → month
index 2 → day
index 3 → hour
index 4 → region
index 5 → altitude
index 6 → latitude
index 7 → longitude
```

config.yaml 和代码中的 mask 列表都必须严格按此顺序排列。
