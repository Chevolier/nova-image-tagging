import pandas as pd
import boto3

account_id = "687752207838"
s3_bucket = "687752207838-dify-files"
s3_prefix = "shein_img_tagging/imgs"

s3_client = boto3.client('s3')

df = pd.read_excel('train_data_balanced.xlsx')

with open('nova_sft_trainset.jsonl', 'w', encoding='utf-8') as f:
    for idx, row in df.iterrows():
        filename = row['filename']
        s3_path = f"s3://{s3_bucket}/{s3_prefix}/{filename}"
        
        try:
            s3_client.head_object(Bucket=s3_bucket, Key=f"{s3_prefix}/{filename}")
        except:
            print(f"{s3_prefix}/{filename} is not existed")
            continue