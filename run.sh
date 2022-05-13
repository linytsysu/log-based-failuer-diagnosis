python code/preprocess.py -s final_b
python code/feature.py -s final_b
python code/main.py -s final_b

zip -j result.zip /prediction_result/final_pred_b.csv
