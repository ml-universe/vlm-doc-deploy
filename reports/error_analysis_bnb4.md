# 错误分析报告 —— bnb4

样本数：624 · 字段数：2496

## 总体分布

| 类别 | 数量 | 占比 |
|---|---|---|
| wrong | 165 | 6.61% |
| hallucination | 2 | 0.08% |
| missing | 2 | 0.08% |
| noise | 256 | 10.26% |
| correct | 2071 | 82.97% |

## 分字段分布

| 字段 | 完全正确 | 噪声 | 漏抽 | 误抽 | 语义错抽 |
|---|---|---|---|---|---|
| company | 511 | 51 | 0 | 0 | 62 |
| date | 560 | 45 | 2 | 0 | 17 |
| address | 408 | 137 | 0 | 1 | 78 |
| total | 592 | 23 | 0 | 1 | 8 |

## 典型错例

### 语义错抽
- `address`: pred=`LOT 1851-A & 1851-B, JALAN KPB 6, KAWASA` ← gold=`LOT 1851-A & 1851-B, JALAN KPB 6, KAWASA`
- `company`: pred=`MR D.I.Y. (H) SDN BHD` ← gold=`MR D.I.Y. (M) SDN BHD`
- `address`: pred=`10T 1851-A & 1851-B, JALAN KPB 6, KAWASA` ← gold=`LOT 1851-A & 1851-B, JALAN KPB 6, KAWASA`
- `company`: pred=`tan chay yee S.H.H. MOTOR ( SUNGAI RENGI` ← gold=`S.H.H. MOTOR (SUNGAI RENGIT) SDN. BHD.`
- `address`: pred=`No. 343, Jalan Kurau, Sungai Rengit, 816` ← gold=`NO. 343, JALAN KURAU, SUNGAI RENGIT, 816`

### 误抽
- `total`: pred=`$8.20` ← gold=``
- `address`: pred=`4 CHILIS3 23 Anis` ← gold=``

### 漏抽
- `date`: pred=`` ← gold=`23-01-2019`
- `date`: pred=`` ← gold=`25032018`

### 噪声
- `company`: pred=`BOOK TA.K(TAMAN DAYA) SDN BHD` ← gold=`BOOK TA .K (TAMAN DAYA) SDN BHD`
- `address`: pred=`27,JALAN DEDAP 13, TAMAN JOHOR JAYA, 811` ← gold=`27, JALAN DEDAP 13, TAMAN JOHOR JAYA, 81`
- `total`: pred=`RM 33.90` ← gold=`33.90`
- `address`: pred=`NO 122,124, JALAN DEDAP 13, 81100 JOHOR ` ← gold=`NO 122.124. JALAN DEDAP 13 81100 JOHOR B`
- `total`: pred=`RM 30.90` ← gold=`30.90`

