# This is so that you can import ppack or import average from ppack
# in stead of from ppack.functions import average

from .classify_func import classifyme
from .gtd_check_QA import check_GTD_for_CR, load_txt_file
from .functions import load_txt_file, save_file, count_now, read_json_originalText, get_predict, Modality_labels_details
from .functions import CodeSwitch_labels_details, Types_labels_details, Ranks_labels_details, load_modality_model, load_txt_file_ISO
from .functions import read_csv_file_tab, read_csv_file_comma, CountFrequency_labeles,log, load_model, load_txt_file_utf8

from .functions import create_new_auterance_MyMemoryTranslator, create_new_auterance_GoogleTranslator
from .evaluations import show_report, chart_data, make_confusion_matrix, show_wrong_prred, log, Create_model_max, save_model, check_mislabeled, save_file, Create_model_Synthatic
from .evaluations import Create_model_min

from .featurebasedmodality import Create_feature_based_modality
from .ara_functions import convert_ara_to_bw, convert_bw_to_ara

from .create_model import create_model

