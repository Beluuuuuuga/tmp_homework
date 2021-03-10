## 概要
このレポートでは、以下について報告
- オートエンコーダ(以下AE)のモデル
- AEから特徴量を抽出した後に連結した畳み込み分類モデル(以下CNN)
- 不均衡データに対する施策
- データ拡張手法
- 実験結果のサマライズ
- 考察

## AEモデル
特徴量抽出のため、単一のAEとAEを3つ重ねたStacked AEを作成した。
- 単一AEとStacked AEのハイパーパラメータ最適化
- 単一AEとStacked AEの特徴量抽出のCNN結合モデルの性能

## CNNモデル
以下について報告する。
- 作成したモデル
- モデルのハイパーパラメータ最適化

#### 作成したモデル
AEで特徴量抽出後のモデルは主に4つ作成し、性能を比較した。
- モデル1: 単一AE(1層)+CNN(3層)+全結合(2層)の6層モデル
- モデル2: 単一AE(1層)+CNN(9層)+全結合(2層)の12層モデル
- モデル3: 単一AE(1層)+CNN(12層)+全結合(2層)の15層モデル
- モデル4: 単一AE(1層)+CNN(13層)+全結合(2層)のResNetベースの16層モデル

#### モデルのハイパーパラメータ最適化
ハイパーパラメータ最適化では精度が十分であると考えられたモデル2のパラメータの最適化を行った。
最適化したパラメータと候補は
- バッチサイズ: 32, 64, 128
- 最適化関数: Adam, RmSprop, SDG
である。  



## 不均衡データに対する施策
今回の課題では学習画像が10クラスそれぞれ5000枚存在しているが、鳥・鹿・トラックの3分類が学習画像では2500枚とその他のクラスの半分になっており、不均衡状態である。  
不均衡データに対する施策は
- アンダーサンプリング
- オーバーサンプリング
- オーバーサンプリング(SMOTE)
- クラス重み付け
- Out of fold 予測; 交差検証(Cross validation: CV)時にモデルを作成しアンサンブル
であり、それぞれについて報告する。

## データ拡張手法
`rotation_range`, `shear_range`, `horizontal_flip`, `vertical_flip`, `width_shift_range`, `height_shift_range`, `zoom_range`, `channel_shift_range`

## 実験結果サマライズ
| index | experiment                                  | accuracy | macro F1 | macro minority F1 | bird f1 | deer f1 | truck f1 | notebook name     | 
| ----- | ------------------------------------------- | -------- | -------- | ----------------- | ------- | ------- | -------- | ----------------- | 
| 1     | Stack AE + CNN 1                            | 47       | 43       | 44                | 36      | 18      | 77       | StackAE_2_CNN_001 | 
| 2     | Stack AE + CNN 1 +  Aug 1                   | 38       | 29       | 32                | 1       | 41      | 55       | StackAE_2_CNN_001 | 
| 3     | AE + CNN 1                                  | 71       | 70       | 66                | 54      | 65      | 78       | AE_2_CNN_002      | 
| 4     | AE + CNN 1 + CV                             | 70       | 68       | -                 | -       | -       | -        | AE_2_CNN_004      | 
| 5     | AE + CNN 1 +  Aug 1                         | 67       | 66       | 60                | 46      | 57      | 76       | AE_2_CNN_002      | 
| 6     | AE + CNN 1 +  Aug 1 + CV                    | 76       | 74       | -                 | -       | -       | -        | AE_2_CNN_004      | 
| 7     | AE + CNN 1 +  Aug 2                         | 79       | 78       | 75                | 63      | 76      | 86       | AE_2_CNN_002      | 
| 8     | AE + CNN 1 +  Aug 2 + CV                    | 78       | 76       | -                 | -       | -       | -        | AE_2_CNN_004      | 
| 9     | AE + CNN 1 + OOF                            | 78       | 78       | 73                | 62      | 74      | 84       | AE_2_CNN_004      | 
| 10    | AE + CNN 1 +  Aug 1 + OOF                   | 74       | 74       | 70                | 62      | 65      | 83       | AE_2_CNN_004      | 
| 11    | AE + CNN 1 +  Aug 2 + OOF                   | 82       | 82       | 79                | 71      | 77      | 88       | AE_2_CNN_004      | 
| 12    | AE + CNN 1 + UnderSamp                      | 68       | 68       | 65                | 56      | 65      | 75       | AE_2_CNN_003      | 
| 13    | AE + CNN 1 + UnderSamp + CV                 | 76       | 75       | -                 | -       | -       | -        | AE_2_CNN_007      | 
| 14    | AE + CNN 1 + UnderSamp + Aug 2              | 73       | 74       | 72                | 65      | 67      | 83       | AE_2_CNN_003      | 
| 15    | AE + CNN 1 + UnderSamp + Aug 2  + CV        | 74       | 72       | -                 | -       | -       | -        | AE_2_CNN_007      | 
| 16    | AE + CNN 1 + UnderSamp + OOF                | 76       | 76       | 74                | 65      | 71      | 85       | AE_2_CNN_007      | 
| 17    | AE + CNN 1 + UnderSamp + Aug 2  + OOF       | 78       | 77       | 76                | 69      | 75      | 85       | AE_2_CNN_007      | 
| 18    | AE + CNN 1 + ClassWeight                    | 72       | 72       | 71                | 62      | 67      | 83       | AE_2_CNN_005      | 
| 19    | AE + CNN 1 + ClassWeight + CV               | 72       | 71       | -                 | -       | -       | -        | AE_2_CNN_006      | 
| 20    | AE + CNN 1 + ClassWeight+ Aug 2             | 77       | 77       | 73                | 63      | 72      | 85       | AE_2_CNN_005      | 
| 21    | AE + CNN 1 + ClassWeight+ Aug 2 + CV        | 78       | 77       | -                 | -       | -       | -        | AE_2_CNN_006      | 
| 22    | AE + CNN 1 + ClassWeight + OOF              | 78       | 78       | 74                | 67      | 69      | 85       | AE_2_CNN_006      | 
| 23    | AE + CNN 1 + ClassWeight + Aug 2  + OOF     | 81       | 80       | 78                | 71      | 76      | 88       | AE_2_CNN_006      | 
| 24    | AE + CNN 1 + OverSamp                       | 75       | 75       | 73                | 64      | 71      | 83       | AE_2_CNN_008      | 
| 25    | AE + CNN 1 + OverSamp + CV                  | 87       | 87       | -                 | -       | -       | -        | AE_2_CNN_009      | 
| 26    | AE + CNN 1 + OverSamp + Aug 2               | 77       | 77       | 76                | 70      | 72      | 85       | AE_2_CNN_008      | 
| 27    | AE + CNN 1 + OverSamp + Aug 2  + CV         | 79       | 78       | -                 | -       | -       | -        | AE_2_CNN_009      | 
| 28    | AE + CNN 1 + OverSamp + OOF                 | 79       | 79       | 77                | 71      | 75      | 86       | AE_2_CNN_009      | 
| 29    | AE + CNN 1 + OverSamp + Aug 2  + OOF        | 78       | 77       | 76                | 69      | 75      | 85       | AE_2_CNN_009      | 
| 30    | AE + CNN 1 + OverSamp(SMOTE)                | 75       | 75       | 69                | 65      | 68      | 75       | AE_2_CNN_010      | 
| 31    | AE + CNN 1 + OverSamp(SMOTE) + CV           | 73       | 71       | -                 | -       | -       | -        | AE_2_CNN_011      | 
| 32    | AE + CNN 1 + OverSamp(SMOTE) + Aug 2        | 76       | 75       | 74                | 70      | 66      | 85       | AE_2_CNN_010      | 
| 33    | AE + CNN 1 + OverSamp(SMOTE) + Aug 2 + CV   | 76       | 75       | -                 | -       | -       | -        | AE_2_CNN_011      | 
| 34    | AE + CNN 1 + OverSamp(SMOTE) + OOF          | 77       | 77       | 74                | 67      | 74      | 82       | AE_2_CNN_011      | 
| 35    | AE + CNN 1 + OverSamp(SMOTE) + Aug 2  + OOF | 81       | 81       | 78                | 72      | 73      | 88       | AE_2_CNN_011      | 
| 36    | AE + CNN 2                                  | 78       | 78       | 74                | 70      | 70      | 81       | AE_2_CNN_012      | 
| 37    | AE + CNN 2 + CV                             | 80       | 79       | -                 | -       | -       | -        | AE_2_CNN_014      | 
| 38    | AE + CNN 3                                  | 78       | 78       | 76                | 65      | 75      | 87       | AE_2_CNN_012      | 
| 39    | AE + CNN 4                                  | 75       | 75       | 75                | 68      | 74      | 82       | AE_2_CNN_012      | 
| 40    | AE + CNN 2 + Aug 2                          | 80       | 80       | 75                | 66      | 72      | 88       | AE_2_CNN_012      | 
| 41    | AE + CNN 2 + Aug 2 + CV                     | 83       | 87       | -                 | -       | -       | -        | AE_2_CNN_014      | 
| 42    | AE + CNN 2 + Aug 2 + Batchsize64            | 77       | 77       | 73                | 66      | 68      | 86       | AE_2_CNN_013      | 
| 43    | AE + CNN 2 + Aug 2 + OOF                    | 82       | 82       | 78                | 69      | 79      | 87       | AE_2_CNN_014      | 
| 44    | AE + CNN 2 + Aug 2 + OOF           | 87       | 87       | 86                | 80      | 85      | 93       | AE_2_CNN_014      | 

## 考察



## ビルド
This environment is build in Windows(Anaconda) @ Python 3.8.
You make the conda environment and then push below commands:

```
$ python -m pip install --upgrade pip
$ pip install -r requirements.txt
$ conda install jupyter
```
