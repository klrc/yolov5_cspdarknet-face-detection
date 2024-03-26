# yolov5_cspdarknet-face-detection

## 运行demo
```shell
python demo.py --device mps --weight my_model.pt --camera
```

## 训练模型
```shell
python trainer.py --device cuda --batch_size 16 --max_epochs 200 --ema_enabled --wandb_enabled --autocast_enabled --early_stop
```

## 导出模型
```shell
python export.py --weight my_model.pt --input_shape 1 3 640 352 --input_names image --opset_version 13 --enable_onnxsim
```
