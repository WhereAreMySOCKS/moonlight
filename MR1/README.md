# K-折交叉验证

主要做了如下改动：
- 新增了run.py，增加日志功能，实现模型的训练和测试。
- 新增了utils/utils.py，实现了一些工具方法。
- 调整了文件存储路径：
  - swin_transformer预训练模型参数： MR1/data1/pretrain_model/swin_tiny
  - 训练日志和测试集保存路径：MR1/data1/logs
  - nii.gz解压并转换为png后的路径：MR1/data1/png
  - 训练完成的模型参数保存路径：MR1/model/model_pth

模型运行和评估：
- 训练swin-transformer并使用5折交叉验证：python run.py --model_type=2 --k=5 --gpu=0

主要参数说明：
- model_type：模型类型，0表示使用Unet，1表示使用UNext，2表示使用swin-transformer预训练模型。
- k：交叉验证的折数。
- gpu：使用的gpu编号。
- is_val: 训练阶段是否使用验证集。
- patience：early stopping的patience值。
- save_path：模型参数保存路径。
- png_data_path：nii.gz解压并转换为png后的路径。
- test_data_path：测试集路径。
