#
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: A V O I R   D E E P   L E A R N I N G   N L P   S E T T I N G S   M A N A G E M E N T : :  :   :    :     :        :          :
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#

from pydantic import BaseModel, BaseSettings


class Settings(BaseSettings):
    invoice_tesseract_tsv_folder_path: str
    invoice_erp_data_filepath: str
    invoice_transformer_model_path: str
    invoice_transformer_ner_model_path: str
    invoice_transformer_max_lenght: int
    invoice_train_dataset_file: str
    invoice_validate_dataset_file: str
    invoice_test_dataset_file: str
    invoice_model_path: str
    invoice_model_temp_path: str
    invoice_training_logger_path: str
    invoice_prod_weights_filepath: str
    train_path: str
    validation_path: str
    test_path: str
    pdf_document_max: int
    validation_percent: int
    test_percent: int
    random_seed: int
    ner_label_n: int
    epoch_n: int
    batch_size: int

    class Config:
        env_file = "/Users/a20-100/repos/2012_avior_1/aviorml/.env"
        env_file_encoding = "utf-8"


settings = Settings()
