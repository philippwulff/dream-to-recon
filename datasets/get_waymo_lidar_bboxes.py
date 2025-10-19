"""
This script downloads bbox annotations for our Waymo dataset version. 
We use the bounding boxes to build GT occupancy for evaluation.
You do not need to run this script if you re-use the GT occupancy 
linked in our README.

pip install gcsfs waymo-open-dataset-tf-2-12-0==1.6.4

helpful links:
- https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto
"""

import os
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import dask.dataframe as dd
from waymo_open_dataset import v2

data_path_bbox = "../data/waymo_bbox"
data_path_calib = "../data/waymo_camera_calib"
out_path = "../data/waymo_bbox_out"

os.makedirs(data_path_bbox, exist_ok=True)
os.makedirs(data_path_calib, exist_ok=True)
os.makedirs(out_path, exist_ok=True)

os.system(
f"""
gsutil -m cp \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/10203656353524179475_7625_000_7645_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/1024360143612057520_3580_000_3600_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/10247954040621004675_2180_000_2200_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/10289507859301986274_4200_000_4220_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/10335539493577748957_1372_870_1392_870.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/10359308928573410754_720_000_740_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/10448102132863604198_472_000_492_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/10689101165701914459_2072_300_2092_300.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/1071392229495085036_1844_790_1864_790.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/10837554759555844344_6525_000_6545_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/10868756386479184868_3000_000_3020_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/11037651371539287009_77_670_97_670.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/11048712972908676520_545_000_565_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/1105338229944737854_1280_000_1300_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/11356601648124485814_409_000_429_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/11387395026864348975_3820_000_3840_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/11406166561185637285_1753_750_1773_750.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/11434627589960744626_4829_660_4849_660.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/11450298750351730790_1431_750_1451_750.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/11616035176233595745_3548_820_3568_820.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/11660186733224028707_420_000_440_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/11901761444769610243_556_000_576_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/12102100359426069856_3931_470_3951_470.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/12134738431513647889_3118_000_3138_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/12306251798468767010_560_000_580_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/12358364923781697038_2232_990_2252_990.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/12374656037744638388_1412_711_1432_711.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/12496433400137459534_120_000_140_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/12657584952502228282_3940_000_3960_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/12820461091157089924_5202_916_5222_916.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/12831741023324393102_2673_230_2693_230.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/12866817684252793621_480_000_500_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/12940710315541930162_2660_000_2680_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/13178092897340078601_5118_604_5138_604.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/13184115878756336167_1354_000_1374_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/13299463771883949918_4240_000_4260_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/1331771191699435763_440_000_460_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/13336883034283882790_7100_000_7120_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/13356997604177841771_3360_000_3380_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/13415985003725220451_6163_000_6183_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/13469905891836363794_4429_660_4449_660.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/13573359675885893802_1985_970_2005_970.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/13694146168933185611_800_000_820_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/13941626351027979229_3363_930_3383_930.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/13982731384839979987_1680_000_1700_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/1405149198253600237_160_000_180_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14081240615915270380_4399_000_4419_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14107757919671295130_3546_370_3566_370.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14127943473592757944_2068_000_2088_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14165166478774180053_1786_000_1806_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14244512075981557183_1226_840_1246_840.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14262448332225315249_1280_000_1300_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14300007604205869133_1160_000_1180_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14333744981238305769_5658_260_5678_260.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14383152291533557785_240_000_260_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14486517341017504003_3406_349_3426_349.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/1457696187335927618_595_027_615_027.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14624061243736004421_1840_000_1860_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/1464917900451858484_1960_000_1980_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14663356589561275673_935_195_955_195.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14687328292438466674_892_000_912_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14739149465358076158_4740_000_4760_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14811410906788672189_373_113_393_113.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14931160836268555821_5778_870_5798_870.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/14956919859981065721_1759_980_1779_980.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/15021599536622641101_556_150_576_150.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/15028688279822984888_1560_000_1580_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/1505698981571943321_1186_773_1206_773.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/15096340672898807711_3765_000_3785_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/15224741240438106736_960_000_980_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/15396462829361334065_4265_000_4285_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/15488266120477489949_3162_920_3182_920.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/15496233046893489569_4551_550_4571_550.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/15611747084548773814_3740_000_3760_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/15724298772299989727_5386_410_5406_410.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/15948509588157321530_7187_290_7207_290.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/15959580576639476066_5087_580_5107_580.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/16204463896543764114_5340_000_5360_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/16213317953898915772_1597_170_1617_170.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/16229547658178627464_380_000_400_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/16751706457322889693_4475_240_4495_240.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/16767575238225610271_5185_000_5205_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/16979882728032305374_2719_000_2739_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/17065833287841703_2980_000_3000_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/17135518413411879545_1480_000_1500_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/17136314889476348164_979_560_999_560.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/17152649515605309595_3440_000_3460_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/17244566492658384963_2540_000_2560_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/17344036177686610008_7852_160_7872_160.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/17539775446039009812_440_000_460_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/17612470202990834368_2800_000_2820_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/17626999143001784258_2760_000_2780_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/17694030326265859208_2340_000_2360_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/17703234244970638241_220_000_240_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/17763730878219536361_3144_635_3164_635.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/17791493328130181905_1480_000_1500_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/17860546506509760757_6040_000_6060_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/17962792089966876718_2210_933_2230_933.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/18024188333634186656_1566_600_1586_600.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/18045724074935084846_6615_900_6635_900.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/18252111882875503115_378_471_398_471.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/18305329035161925340_4466_730_4486_730.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/18331704533904883545_1560_000_1580_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/18333922070582247333_320_280_340_280.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/18446264979321894359_3700_000_3720_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/1906113358876584689_1359_560_1379_560.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/191862526745161106_1400_000_1420_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/1943605865180232897_680_000_700_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/2094681306939952000_2972_300_2992_300.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/2105808889850693535_2295_720_2315_720.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/2308204418431899833_3575_000_3595_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/2335854536382166371_2709_426_2729_426.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/2367305900055174138_1881_827_1901_827.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/2506799708748258165_6455_000_6475_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/2551868399007287341_3100_000_3120_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/260994483494315994_2797_545_2817_545.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/2624187140172428292_73_000_93_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/271338158136329280_2541_070_2561_070.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/272435602399417322_2884_130_2904_130.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/2736377008667623133_2676_410_2696_410.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/2834723872140855871_1615_000_1635_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/3015436519694987712_1300_000_1320_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/3039251927598134881_1240_610_1260_610.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/3077229433993844199_1080_000_1100_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/30779396576054160_1880_000_1900_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/3126522626440597519_806_440_826_440.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/346889320598157350_798_187_818_187.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/3577352947946244999_3980_000_4000_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/3651243243762122041_3920_000_3940_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/366934253670232570_2229_530_2249_530.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/3731719923709458059_1540_000_1560_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/3915587593663172342_10_000_30_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/4013125682946523088_3540_000_3560_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/4195774665746097799_7300_960_7320_960.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/4246537812751004276_1560_000_1580_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/4409585400955983988_3500_470_3520_470.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/4423389401016162461_4235_900_4255_900.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/4426410228514970291_1620_000_1640_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/447576862407975570_4360_000_4380_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/4490196167747784364_616_569_636_569.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/4575389405178805994_4900_000_4920_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/4612525129938501780_340_000_360_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/4690718861228194910_1980_000_2000_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/4759225533437988401_800_000_820_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/4764167778917495793_860_000_880_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/4816728784073043251_5273_410_5293_410.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/4854173791890687260_2880_000_2900_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/5183174891274719570_3464_030_3484_030.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/5289247502039512990_2640_000_2660_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/5302885587058866068_320_000_340_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/5372281728627437618_2005_000_2025_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/5373876050695013404_3817_170_3837_170.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/5574146396199253121_6759_360_6779_360.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/5772016415301528777_1400_000_1420_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/5832416115092350434_60_000_80_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/5847910688643719375_180_000_200_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/5990032395956045002_6600_000_6620_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/6001094526418694294_4609_470_4629_470.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/6074871217133456543_1000_000_1020_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/6161542573106757148_585_030_605_030.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/6183008573786657189_5414_000_5434_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/6324079979569135086_2372_300_2392_300.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/6491418762940479413_6520_000_6540_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/662188686397364823_3248_800_3268_800.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/6637600600814023975_2235_000_2255_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/6680764940003341232_2260_000_2280_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/6707256092020422936_2352_392_2372_392.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/7119831293178745002_1094_720_1114_720.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/7163140554846378423_2717_820_2737_820.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/7253952751374634065_1100_000_1120_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/7493781117404461396_2140_000_2160_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/7650923902987369309_2380_000_2400_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/7732779227944176527_2120_000_2140_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/7799643635310185714_680_000_700_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/7932945205197754811_780_000_800_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/7988627150403732100_1487_540_1507_540.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/8079607115087394458_1240_000_1260_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/8133434654699693993_1162_020_1182_020.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/8137195482049459160_3100_000_3120_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/8302000153252334863_6020_000_6040_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/8331804655557290264_4351_740_4371_740.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/8398516118967750070_3958_000_3978_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/8506432817378693815_4860_000_4880_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/8679184381783013073_7740_000_7760_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/8845277173853189216_3828_530_3848_530.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/8888517708810165484_1549_770_1569_770.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/8907419590259234067_1960_000_1980_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/89454214745557131_3160_000_3180_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/8956556778987472864_3404_790_3424_790.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/902001779062034993_2880_000_2900_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/9024872035982010942_2578_810_2598_810.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/9041488218266405018_6454_030_6474_030.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/9114112687541091312_1100_000_1120_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/9164052963393400298_4692_970_4712_970.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/9231652062943496183_1740_000_1760_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/9243656068381062947_1297_428_1317_428.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/9265793588137545201_2981_960_3001_960.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/933621182106051783_4160_000_4180_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/9443948810903981522_6538_870_6558_870.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/9472420603764812147_850_000_870_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/9579041874842301407_1300_000_1320_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/camera_calibration/967082162553397800_5102_900_5122_900.parquet" \
  {data_path_calib}

gsutil -m cp \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/10203656353524179475_7625_000_7645_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/1024360143612057520_3580_000_3600_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/10247954040621004675_2180_000_2200_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/10289507859301986274_4200_000_4220_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/10335539493577748957_1372_870_1392_870.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/10359308928573410754_720_000_740_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/10448102132863604198_472_000_492_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/10689101165701914459_2072_300_2092_300.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/1071392229495085036_1844_790_1864_790.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/10837554759555844344_6525_000_6545_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/10868756386479184868_3000_000_3020_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/11037651371539287009_77_670_97_670.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/11048712972908676520_545_000_565_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/1105338229944737854_1280_000_1300_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/11356601648124485814_409_000_429_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/11387395026864348975_3820_000_3840_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/11406166561185637285_1753_750_1773_750.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/11434627589960744626_4829_660_4849_660.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/11450298750351730790_1431_750_1451_750.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/11616035176233595745_3548_820_3568_820.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/11660186733224028707_420_000_440_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/11901761444769610243_556_000_576_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/12102100359426069856_3931_470_3951_470.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/12134738431513647889_3118_000_3138_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/12306251798468767010_560_000_580_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/12358364923781697038_2232_990_2252_990.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/12374656037744638388_1412_711_1432_711.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/12496433400137459534_120_000_140_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/12657584952502228282_3940_000_3960_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/12820461091157089924_5202_916_5222_916.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/12831741023324393102_2673_230_2693_230.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/12866817684252793621_480_000_500_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/12940710315541930162_2660_000_2680_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/13178092897340078601_5118_604_5138_604.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/13184115878756336167_1354_000_1374_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/13299463771883949918_4240_000_4260_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/1331771191699435763_440_000_460_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/13336883034283882790_7100_000_7120_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/13356997604177841771_3360_000_3380_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/13415985003725220451_6163_000_6183_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/13469905891836363794_4429_660_4449_660.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/13573359675885893802_1985_970_2005_970.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/13694146168933185611_800_000_820_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/13941626351027979229_3363_930_3383_930.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/13982731384839979987_1680_000_1700_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/1405149198253600237_160_000_180_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14081240615915270380_4399_000_4419_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14107757919671295130_3546_370_3566_370.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14127943473592757944_2068_000_2088_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14165166478774180053_1786_000_1806_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14244512075981557183_1226_840_1246_840.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14262448332225315249_1280_000_1300_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14300007604205869133_1160_000_1180_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14333744981238305769_5658_260_5678_260.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14383152291533557785_240_000_260_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14486517341017504003_3406_349_3426_349.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/1457696187335927618_595_027_615_027.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14624061243736004421_1840_000_1860_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/1464917900451858484_1960_000_1980_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14663356589561275673_935_195_955_195.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14687328292438466674_892_000_912_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14739149465358076158_4740_000_4760_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14811410906788672189_373_113_393_113.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14931160836268555821_5778_870_5798_870.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/14956919859981065721_1759_980_1779_980.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/15021599536622641101_556_150_576_150.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/15028688279822984888_1560_000_1580_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/1505698981571943321_1186_773_1206_773.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/15096340672898807711_3765_000_3785_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/15224741240438106736_960_000_980_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/15396462829361334065_4265_000_4285_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/15488266120477489949_3162_920_3182_920.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/15496233046893489569_4551_550_4571_550.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/15611747084548773814_3740_000_3760_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/15724298772299989727_5386_410_5406_410.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/15948509588157321530_7187_290_7207_290.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/15959580576639476066_5087_580_5107_580.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/16204463896543764114_5340_000_5360_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/16213317953898915772_1597_170_1617_170.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/16229547658178627464_380_000_400_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/16751706457322889693_4475_240_4495_240.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/16767575238225610271_5185_000_5205_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/16979882728032305374_2719_000_2739_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/17065833287841703_2980_000_3000_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/17135518413411879545_1480_000_1500_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/17136314889476348164_979_560_999_560.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/17152649515605309595_3440_000_3460_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/17244566492658384963_2540_000_2560_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/17344036177686610008_7852_160_7872_160.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/17539775446039009812_440_000_460_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/17612470202990834368_2800_000_2820_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/17626999143001784258_2760_000_2780_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/17694030326265859208_2340_000_2360_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/17703234244970638241_220_000_240_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/17763730878219536361_3144_635_3164_635.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/17791493328130181905_1480_000_1500_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/17860546506509760757_6040_000_6060_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/17962792089966876718_2210_933_2230_933.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/18024188333634186656_1566_600_1586_600.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/18045724074935084846_6615_900_6635_900.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/18252111882875503115_378_471_398_471.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/18305329035161925340_4466_730_4486_730.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/18331704533904883545_1560_000_1580_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/18333922070582247333_320_280_340_280.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/18446264979321894359_3700_000_3720_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/1906113358876584689_1359_560_1379_560.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/191862526745161106_1400_000_1420_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/1943605865180232897_680_000_700_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/2094681306939952000_2972_300_2992_300.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/2105808889850693535_2295_720_2315_720.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/2308204418431899833_3575_000_3595_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/2335854536382166371_2709_426_2729_426.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/2367305900055174138_1881_827_1901_827.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/2506799708748258165_6455_000_6475_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/2551868399007287341_3100_000_3120_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/260994483494315994_2797_545_2817_545.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/2624187140172428292_73_000_93_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/271338158136329280_2541_070_2561_070.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/272435602399417322_2884_130_2904_130.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/2736377008667623133_2676_410_2696_410.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/2834723872140855871_1615_000_1635_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/3015436519694987712_1300_000_1320_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/3039251927598134881_1240_610_1260_610.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/3077229433993844199_1080_000_1100_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/30779396576054160_1880_000_1900_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/3126522626440597519_806_440_826_440.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/346889320598157350_798_187_818_187.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/3577352947946244999_3980_000_4000_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/3651243243762122041_3920_000_3940_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/366934253670232570_2229_530_2249_530.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/3731719923709458059_1540_000_1560_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/3915587593663172342_10_000_30_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/4013125682946523088_3540_000_3560_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/4195774665746097799_7300_960_7320_960.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/4246537812751004276_1560_000_1580_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/4409585400955983988_3500_470_3520_470.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/4423389401016162461_4235_900_4255_900.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/4426410228514970291_1620_000_1640_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/447576862407975570_4360_000_4380_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/4490196167747784364_616_569_636_569.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/4575389405178805994_4900_000_4920_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/4612525129938501780_340_000_360_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/4690718861228194910_1980_000_2000_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/4759225533437988401_800_000_820_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/4764167778917495793_860_000_880_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/4816728784073043251_5273_410_5293_410.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/4854173791890687260_2880_000_2900_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/5183174891274719570_3464_030_3484_030.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/5289247502039512990_2640_000_2660_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/5302885587058866068_320_000_340_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/5372281728627437618_2005_000_2025_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/5373876050695013404_3817_170_3837_170.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/5574146396199253121_6759_360_6779_360.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/5772016415301528777_1400_000_1420_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/5832416115092350434_60_000_80_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/5847910688643719375_180_000_200_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/5990032395956045002_6600_000_6620_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/6001094526418694294_4609_470_4629_470.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/6074871217133456543_1000_000_1020_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/6161542573106757148_585_030_605_030.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/6183008573786657189_5414_000_5434_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/6324079979569135086_2372_300_2392_300.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/6491418762940479413_6520_000_6540_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/662188686397364823_3248_800_3268_800.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/6637600600814023975_2235_000_2255_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/6680764940003341232_2260_000_2280_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/6707256092020422936_2352_392_2372_392.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/7119831293178745002_1094_720_1114_720.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/7163140554846378423_2717_820_2737_820.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/7253952751374634065_1100_000_1120_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/7493781117404461396_2140_000_2160_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/7650923902987369309_2380_000_2400_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/7732779227944176527_2120_000_2140_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/7799643635310185714_680_000_700_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/7932945205197754811_780_000_800_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/7988627150403732100_1487_540_1507_540.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/8079607115087394458_1240_000_1260_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/8133434654699693993_1162_020_1182_020.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/8137195482049459160_3100_000_3120_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/8302000153252334863_6020_000_6040_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/8331804655557290264_4351_740_4371_740.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/8398516118967750070_3958_000_3978_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/8506432817378693815_4860_000_4880_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/8679184381783013073_7740_000_7760_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/8845277173853189216_3828_530_3848_530.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/8888517708810165484_1549_770_1569_770.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/8907419590259234067_1960_000_1980_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/89454214745557131_3160_000_3180_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/8956556778987472864_3404_790_3424_790.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/902001779062034993_2880_000_2900_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/9024872035982010942_2578_810_2598_810.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/9041488218266405018_6454_030_6474_030.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/9114112687541091312_1100_000_1120_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/9164052963393400298_4692_970_4712_970.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/9231652062943496183_1740_000_1760_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/9243656068381062947_1297_428_1317_428.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/9265793588137545201_2981_960_3001_960.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/933621182106051783_4160_000_4180_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/9443948810903981522_6538_870_6558_870.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/9472420603764812147_850_000_870_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/9579041874842301407_1300_000_1320_000.parquet" \
  "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/967082162553397800_5102_900_5122_900.parquet" \
  {data_path_bbox}
"""
)

file_names = os.listdir(data_path_bbox)
for file_name in tqdm(file_names):
    assert file_name.endswith(".parquet")
    scene_id = file_name.removeprefix("validation_lidar_box_").removesuffix(".parquet")
    fp_parquet = os.path.join(data_path_bbox, file_name)
    df = dd.read_parquet(fp_parquet)

    fp_parquet_calib = os.path.join(data_path_calib, f"{scene_id}.parquet")
    df_calib = dd.read_parquet(fp_parquet_calib)

    calib = v2.CameraCalibrationComponent.from_dict(df_calib)
    cam_to_car: list[np.ndarray] = list(calib.extrinsic.transform)
    cam0_to_car = cam_to_car[0].reshape(4, 4)

    lidar_boxes = defaultdict(list)
    for i, row in df.iterrows():
        lidar_box = v2.LiDARBoxComponent.from_dict(row)
        cx = lidar_box.box.center.x  # forward
        cy = lidar_box.box.center.y  # left
        cz = lidar_box.box.center.z  # up
        box_to_save = [
            cx - cam0_to_car[0, 3],
            cy - cam0_to_car[1, 3],
            cz - cam0_to_car[2, 3],
            lidar_box.box.size.x,
            lidar_box.box.size.y,
            lidar_box.box.size.z,
            lidar_box.box.heading,
        ]
        lidar_boxes[row["key.frame_timestamp_micros"]].append(box_to_save)

    # sort by key
    lidar_boxes_sorted = dict(sorted(lidar_boxes.items(), key=lambda x: x[0]))

    scene_path = os.path.join(out_path, scene_id, "lidar_box")
    os.makedirs(scene_path, exist_ok=True)
    for i, (t, boxes) in enumerate(lidar_boxes_sorted.items()):
        boxes_np = np.array(boxes)
        np.save(os.path.join(scene_path, f"{i:010d}"), boxes_np)
