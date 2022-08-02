# This is so that you can import ppack or import average from ppack
# in stead of from ppack.functions import average


from .functions import save_file, count_now, read_json_originalText, get_predict, Modality_labels_details
from .functions import CodeSwitch_labels_details, Types_labels_details, Ranks_labels_details, load_modality_model
from .functions import read_csv_file_tab, read_csv_file_comma, CountFrequency_labeles,log, load_model
from .evaluations import show_report, chart_data, make_confusion_matrix, show_wrong_prred, log, check_mislabeled
  
