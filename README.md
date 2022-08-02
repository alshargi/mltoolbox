# Toolbox

Lextoolbox is a Python package that contains handy functions. 


```bash
pip install git+https://github.com/alshargi/mltoolbox.git
```

## Usage
Features:



#### Demo of some of the features:
```python
# -*- coding: utf-8 -*-



from mltoolbox import read_json_originalText, log, save_file
from mltoolbox import load_modality_model,  Modality_labels_details
from mltoolbox import load_model, get_predict


#### setting
file_input  = 'example_TrainingData.json'
file_output = './result.txt'

  

s_labels_mod, full_name = Modality_labels_details()
print("short name", s_labels_mod)
print("full name",full_name)


# load modality model
loaded_model_mod,  loaded_cvec_mod , loaded_tfidf_transformer_mod =  load_model(load_modality_model())


load_fileToPredect = ['عام ألفين وثلاثة وعشرين', 'هذي السنة', 'عام 2023' ]

keepAllresult = get_predict(load_fileToPredect, loaded_model_mod, 
                       loaded_cvec_mod ,
                       loaded_tfidf_transformer_mod , 
                       full_name)

# save result
save_file(keepAllresult , file_output)
log("Saved" + str(file_output) )
log("#" *30)

>>>>
short name ['s', 'sw', 'w']
full name ['SpokenOnly', 'SpokenAndWritten', 'WrittenOnly']
Models , loaded 

2022-08-02 13:51:01.778928 1	WrittenOnly	عام ألفين وثلاثة وعشرين
2022-08-02 13:51:01.779970 2	WrittenOnly	هذي السنة
2022-08-02 13:51:01.781017 3	WrittenOnly	عام 2023
2022-08-02 13:51:01.781057 ##############################
2022-08-02 13:51:01.781067 # Entries: 3
2022-08-02 13:51:01.781075 
2022-08-02 13:51:01.781084 Modality >> 
2022-08-02 13:51:01.781100 100.0%	003		WrittenOnly
2022-08-02 13:51:01.781110 ##############################
2022-08-02 13:51:01.781565 Saved./result.txt
2022-08-02 13:51:01.781588 ##############################


 
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
