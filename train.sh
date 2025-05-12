# Train resnet model for basic classification.
python train_resnet.py --dataset=data/data_classifier_t_stft.pkl --batch_size=512 --learning_rate=2e-5 --epochs=100
# Train unet model for denoising.
python train_unet.py --type=signal --clean_data_path=data/data_denoise_y_t_stft.pkl --noise_data_path=data/data_denoise_x_t_stft.pkl --batch_size=512 --learning_rate=1e-4 --epochs=100
# another method
python train_unet.py --type=noise --clean_data_path=data/data_denoise_y_t_stft.pkl --noise_data_path=data/data_denoise_x_t_stft.pkl --batch_size=512 --learning_rate=1e-4 --epochs=100
# Train unet-resnet model for best classification
python train_unet_resnet.py --dataset=data/data_classifier_t_stft.pkl --cls_model=checkpoints/resnet.pth --batch_size=512 --learning_rate=1e-4 --epochs=100

