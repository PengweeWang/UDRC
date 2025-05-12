# UDRC: Intra-pulse modulation recognition of radar via U-Net denoising and ResNet classification

UDRC is a radar in-pulse modulation recognition method based on time-spectrum. It has successfully improved the classification accuracy under low signal-to-noise ratio. The technical route is shown in the following figure.

![技术路线](https://stastic.s3.bitiful.net/tr.svg)

## Requirements

```bash
pip install -r requirements.txt
```



## Getting Started

### Generate Dataset

Generate time zone dataset for classification.

```bash
python data/generate4classifier.py 
```

Generate time zone dataset for denoising.

```bash
python data/generate4denoise.py 
```

Convert data to time-spectrum images. The processed data will be saved to `data/`.

```bash
python data/preprocess.py <time_zone_datapath>
```

### Train Model

Train resnet model for basic classification.

```bash
python train_resnet.py --dataset=<dataset_path> --batch_size=512 --learning_rate=2e-5 --epochs=100
```

Train unet model for denoising.

```bash
python train_unet.py --type=<the type of this train> --clean_data_path=<clean_TS_dataset_path> --noise_data_path=<noise_TS_dataset_path> --batch_size=512 --learning_rate=1e-4 --epochs=100
```

Train unet-resnet model for best classification

```bash
python train_unet_resnet.py --dataset=<dataset_path> --cls_model=<pretrained_resnet_path> --batch_size=512 --learning_rate=1e-4 --epochs=100
```

Train script in this paper can be found [here](./train.sh)

### Testing

```bash
python .\test.py --type=resnet --model=checkpoints/resnet.pth --dataset=data/data_stft.pkl # test resnet model

python .\test.py --type=unet-resnet --model=checkpoints/unet_resnet.pth  --dataset=data/data_stft.pkl # test unet-resnet model

python .\test.py --type=unet-noise --model=checkpoints/unet_noise.pth --cls_model=checkpoints/resnet.pth  --dataset=data/data_stft.pkl # test unet-noise model

python .\test.py --type=unet-signal --model=checkpoints/unet_signal.pth --cls_model=checkpoints/resnet.pth  --dataset=data/data_stft.pkl # test unet-signal model
```



## Dataset and Pretrained Model

The dataset and pretrained model for this paper are available [here](https://pan.baidu.com/s/1nuoInahausb1JfAwBodKQg?pwd=udrc).



