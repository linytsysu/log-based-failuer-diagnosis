python code/preprocess.py -s final_a
python code/feature.py -s final_a
python code/main.py -s final_a

zip -j result.zip /prediction_result/final_pred_a.csv
