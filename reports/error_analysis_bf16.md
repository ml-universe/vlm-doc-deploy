# 错误分析报告 —— bf16

样本数：624 · 字段数：2496

## 总体分布

| 类别 | 数量 | 占比 |
|---|---|---|
| wrong | 123 | 4.93% |
| hallucination | 2 | 0.08% |
| missing | 1 | 0.04% |
| noise | 262 | 10.50% |
| correct | 2108 | 84.46% |

## 分字段分布

| 字段 | 完全正确 | 噪声 | 漏抽 | 误抽 | 语义错抽 |
|---|---|---|---|---|---|
| company | 538 | 51 | 0 | 0 | 35 |
| date | 559 | 49 | 1 | 0 | 15 |
| address | 422 | 135 | 0 | 1 | 66 |
| total | 589 | 27 | 0 | 1 | 7 |

## 典型错例

### 语义错抽
- `address`: pred=`10T 1851-A & 1851-B, JALAN KPB 6, KAWASA` ← gold=`LOT 1851-A & 1851-B, JALAN KPB 6, KAWASA`
- `address`: pred=`NO.2-1-1 JALAN SETIA PRIMA Q U 13/Q SETI` ← gold=`NO.2-1-1 JALAN SEITA PRIMA Q U 13/Q SEIT`
- `company`: pred=`TED HENG STATIONERY & BOOKS` ← gold=`TEO HENG STATIONERY & BOOKS`
- `address`: pred=`NO. 53, JALAN BESAR, 45600 BATANG BERJUN` ← gold=`NO.53, JALAN BESAR, 45600 BATANG BEJUNTA`
- `date`: pred=`1 J MAR 2018` ← gold=`14-03-18`

### 误抽
- `total`: pred=`$8.20` ← gold=``
- `address`: pred=`Chili's Grill & Bar KLCC` ← gold=``

### 漏抽
- `date`: pred=`` ← gold=`25032018`

### 噪声
- `address`: pred=`27,JALAN DEDAP 13, TAMAN JOHOR JAYA, 811` ← gold=`27, JALAN DEDAP 13, TAMAN JOHOR JAYA, 81`
- `total`: pred=`RM 33.90` ← gold=`33.90`
- `address`: pred=`NO 122,124 JALAN DEDAP 13 81100 JOHOR BA` ← gold=`NO 122.124. JALAN DEDAP 13 81100 JOHOR B`
- `total`: pred=`RM 30.90` ← gold=`30.90`
- `address`: pred=`NO.2&4,JALAN HARMONI 3/2, TAMAN DESA HAR` ← gold=`NO.2&4, JALAN HARMONI 3/2, TAMAN DESA HA`

