# nova-image-tagging

```bash
conda create -n nova python=3.10 -y
conda activate nova

pip install uv
uv pip install pandas boto3 sagemaker matplotlib
```

## 数据介绍

1. black_url_img_flag是原始的全量数据，图片已经全部下载到/imgs中，文件名在第三列
2. 基于全量数据，过滤掉了所有图片无法下载的记录，生成了下面的三个数据集
2. test_data是最新的用于评估标签PR的数据集，共400+条，35个左右标签
3. train_data是test_data的补集，但是标签很不平衡，刀具有近9000
4. train_data_balanced是train_data平衡后的数据，共1500+条
5. 数据上传到了 s3://687752207838-dify-files/shein_img_tagging/imgs/*.jpg