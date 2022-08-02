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

# pip install git+https://github.com/alshargi/mltoolbox.git





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


# from file
load_fileToPredect = read_json_originalText(file_input)


#load_fileToPredect = ['عام ألفين وثلاثة وعشرين', 'هذي السنة', 'عام 2023' ]
          
#
log('Json file loaded ' + str( len(load_fileToPredect)) +  ' Entries')

keepAllresult = get_predict(load_fileToPredect, loaded_model_mod, 
                       loaded_cvec_mod ,
                       loaded_tfidf_transformer_mod , 
                       full_name)



# save result
save_file(keepAllresult , file_output)
log("Saved" + str(file_output) )
log("#" *30)

 
 
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
