"""
AMC Snowflake ETL Configuration - Heart Failure with Complete Lab Panel
Includes all cardiac biomarkers and clinical lab values for heart failure prediction
"""

amc_hf_complete = {
    "name": "amc_hf_labs_process",
    "description": "Preprocessing of AMC data with complete heart failure lab panel",
    "SOURCE": "SNOWFLAKE",
    "preprocessing": "/tmp/preprocessing_hf_labs",

    "snowflake_config": {
        "input_schema": "RAW_DATA",
        "output_schema": "PROCESSED_DATA",
        "database": "HEALTHCARE_DB"
    },

    # Patient Demographics
    "Datatools4heart_Patient": {
        "table_name": "TBL_PATIENT",
        "columns": ["PSEUDO_ID", "GEBOORTEJAAR", "GEBOORTEMAAND", "GESLACHT", "OVERLIJDENSDATUM"]
    },
    
    # Smoking History
    "Datatools4heart_Tabakgebruik": {
        "table_name": "TBL_TABAKGEBRUIK",
        "columns": ["PSEUDO_ID", "ISHUIDIGEROKER", "ISVOORMALIGROKER", "PATIENTCONTACTID"]
    },
    
    # Admission Data
    "Datatools4heart_Opnametraject": {
        "table_name": "TBL_OPNAMETRAJECT", 
        "columns": ["PSEUDO_ID", "OPNAMEDATUM"]
    },
    
    # BMI Measurements
    "Datatools4heart_MetingBMI": {
        "table_name": "TBL_METINGBMI",
        "columns": ["PSEUDO_ID", "PATIENTCONTACTID", "BMI"]
    },
    
    # Blood Pressure Measurements
    "Datatools4heart_MetingBloeddruk": {
        "table_name": "TBL_METINGBLOEDDRUK",
        "columns": ["PSEUDO_ID", "PATIENTCONTACTID", "SYSTOLISCHEBLOEDDRUKWAARDE", "DIASTOLISCHEBLOEDDRUKWAARDE"]
    },
    
    # Medical History (Diagnoses)
    "Datatools4heart_VoorgeschiedenisMedisch": {
        "table_name": "TBL_VOORGESCHIEDENISMEDISCH",
        "columns": ["PSEUDO_ID", "DIAGNOSECODE_CLASSIFIED", 
                   {"DIAGNOSECODE": {"t2dm": "T2DM", "af": "AF", "acute_mi": "ACUTE_MI"}}, 
                   ["INDICATIEDATUMVASTSTELLING"]]
    },
    
    # Family History
    "Datatools4heart_VoorgeschiedenisFamilie": {
        "table_name": "TBL_VOORGESCHIEDENISFAMILIE",
        "columns": ["PSEUDO_ID", "FAMILIEVOORGESCHIEDENIS", "HF_FAM"]
    },
    
    # Medication Administration
    "Datatools4heart_MedicatieToediening": {
        "table_name": "TBL_MEDICATIETOEDIENING",
        "columns": ["PSEUDO_ID", "ATCCODE_CLASSIFIED", 
                   {"ATCCODE": {"ace": "ACE", "beta": "BETA"}}, 
                   ["TOEDIENINGSDATUM"]]
    },
    
    # Laboratory Results - Complete Heart Failure Panel
    "Datatools4heart_Labuitslag": {
        "table_name": "TBL_LABUITSLAG",
        "columns": ["PSEUDO_ID", "BEPALINGCODE", 
                   {"UITSLAGNUMERIEK": {
                       # Cardiac Biomarkers
                       "RKRE;BL": "CREATININE",
                       "EBNP;BL": "NT_PRO_BNP",
                       "K_BNP_HP": "NT_PRO_BNP_ALT",  # Alternative code
                       "ETRO;BL": "TROPONINE_T",
                       "K_CTNT_HP": "TROPONINE_T_ALT",
                       "K_TNTHS_HP": "TROPONINE_T_HS",
                       
                       # Renal Function
                       "CKD-EPI;BL": "EGFR",
                       "K_EGFRBer": "EGFR_ALT",
                       "RURE;BL": "UREUM",
                       "K_UR_HP": "UREUM_ALT",
                       
                       # Electrolytes
                       "RNAT;BL": "NATRIUM",
                       "K_NA_HP": "NATRIUM_ALT",
                       "RKAL;BL": "KALIUM",
                       "K_KA_HP": "KALIUM_ALT",
                       "RCAL;BL": "CALCIUM",
                       "K_CA_HP": "CALCIUM_ALT",
                       "RMAG;BL": "MAGNESIUM",
                       "K_MAGN_HP": "MAGNESIUM_ALT",
                       "DCHL;BL": "CHLORIDE",
                       "K_PHOS_HP": "FOSFAAT",
                       
                       # Hematology
                       "HHB;BL": "HEMOGLOBIN",
                       "K_HB_EB": "HEMOGLOBIN_ALT",
                       "DEHB;BL": "HEMOGLOBIN_NC",
                       "HTRO;BL": "TROMBOCYTEN",
                       "K_THR_EB": "TROMBOCYTEN_ALT",
                       "HLEU;BL": "LEUKOCYTEN",
                       "K_LEU_EB": "LEUKOCYTEN_ALT",
                       "K_ERY_EB": "ERYTHROCYTEN",
                       "K_HT_EB": "HEMATOCRIET",
                       "K_MCV_EB": "MCV",
                       "K_MCH_EB": "MCH",
                       "K_MCHC_EB": "MCHC",
                       "K_RDW_EB": "RDW",
                       
                       # Blood Gas & Acid-Base
                       "DLAC;BL": "LACTATE",
                       "P_LACT_HB": "LACTATE_HB",
                       "K_LACT_HB": "LACTATE_ALT",
                       "DBIC;BL": "BICARBONATE",
                       "CBIC;BL": "BICARBONATE_ALT",
                       "P_PCO2P_HB": "PCO2",
                       "K_PCO2P_HB": "PCO2_ALT",
                       "P_PH_HB": "PH",
                       "K_PH_HB": "PH_ALT",
                       "P_PO2P_HB": "PO2",
                       "K_PO2P_HB": "PO2_ALT",
                       "P_O2SAT_HB": "O2_SATURATIE",
                       "P_BE_HB": "BASE_EXCESS",
                       "P_AHCO3_HB": "BICARBONAAT_ACT",
                       
                       # Liver Function
                       "K_BILT_HP": "BILIRUBINE",
                       "K_ALAT_HP": "ALAT",
                       "K_ASAT_HP": "ASAT",
                       "K_AF_HP": "ALK_FOSFATASE",
                       "K_YGT_HP": "GAMMA_GT",
                       "K_LD_HP": "LDH",
                       "RLDH;BL": "LDH_ALT",
                       "K_ALBC_HP": "ALBUMINE",
                       "ALBI;BL": "ALBUMINE_ALT",
                       
                       # Lipids
                       "RHDL;BL": "HDL_CHOLESTEROL",
                       "RCHO;BL": "TOTAL_CHOLESTEROL",
                       "K_TRGL_HP": "TRIGLYCERIDEN",
                       
                       # Glucose Metabolism
                       "GLUCS;BL": "GLUCOSE",
                       "RGLU;BL": "GLUCOSE_ALT",
                       "K_GLUC_HP": "GLUCOSE_HP",
                       "P_GLUC_HB": "GLUCOSE_HB",
                       "K_HbA1c_EB": "HBA1C",
                       
                       # Inflammatory Markers
                       "K_CRP_HP": "CRP",
                       
                       # Coagulation
                       "SINR;BL": "INR",
                       "SPT;BL": "PT",
                       "SAPT;BL": "APTT",
                       
                       # Cardiac Enzymes
                       "K_CK_HP": "CREATINE_KINASE",
                       "K_MBM_HP": "CK_MB_MASSA",
                       "EMB;BL": "CKMB_MASSA_ALT",
                       
                       # Other
                       "K_TSH_HP": "TSH",
                       "K_FER_HP": "FERRITINE",
                       "K_D25_SR": "VITAMINE_D"
                   }}, 
                   ["MATERIAALAFNAMEDATUM"]]
    },

    "categories": {"GESLACHT": ["Man", "Vrouw"]},

    # Final columns to keep in the processed dataset
    "final_cols": [
        # Demographics & Clinical
        "GESLACHT", "AGEATOPNAME", "ISHUIDIGEROKER", "OPNAMEDATUM", "TIME",
        "BMI", "SYSTOLISCHEBLOEDDRUKWAARDE", "DIASTOLISCHEBLOEDDRUKWAARDE",
        
        # Medical History
        "T2DM", "ACUTE_MI", "AF", "HF_FAM",
        
        # Medications
        "ACE", "BETA",
        
        # Lipids
        "HDL_CHOLESTEROL", "TOTAL_CHOLESTEROL", "TRIGLYCERIDEN",
        
        # Cardiac Biomarkers
        "NT_PRO_BNP", "TROPONINE_T", "TROPONINE_T_HS", "CREATINE_KINASE", "CK_MB_MASSA",
        
        # Renal Function
        "CREATININE", "EGFR", "UREUM",
        
        # Electrolytes
        "NATRIUM", "KALIUM", "CALCIUM", "MAGNESIUM", "CHLORIDE", "FOSFAAT",
        
        # Hematology
        "HEMOGLOBIN", "HEMATOCRIET", "ERYTHROCYTEN", "LEUKOCYTEN", "TROMBOCYTEN",
        "MCV", "MCH", "MCHC", "RDW",
        
        # Blood Gas & Acid-Base
        "LACTATE", "BICARBONATE", "PCO2", "PH", "PO2", "O2_SATURATIE", "BASE_EXCESS",
        
        # Liver Function
        "BILIRUBINE", "ALAT", "ASAT", "ALK_FOSFATASE", "GAMMA_GT", "LDH", "ALBUMINE",
        
        # Glucose
        "GLUCOSE", "HBA1C",
        
        # Inflammatory
        "CRP",
        
        # Coagulation
        "INR", "PT", "APTT",
        
        # Other
        "TSH", "FERRITINE", "VITAMINE_D"
    ],

    # Temporal processing - values after reference date (admission)
    "tuple_vals_after_temp": [
        # Labs that should be measured close to admission
        "CREATININE"
    ],
    "tuple_vals_after": [
        # Labs that should be measured close to admission
        "HDL_CHOLESTEROL", "TOTAL_CHOLESTEROL", "TRIGLYCERIDEN",
        "CREATININE", "EGFR", "UREUM",
        "NT_PRO_BNP", "TROPONINE_T", "TROPONINE_T_HS", "CREATINE_KINASE", "CK_MB_MASSA",
        "NATRIUM", "KALIUM", "CALCIUM", "MAGNESIUM", "CHLORIDE", "FOSFAAT",
        "HEMOGLOBIN", "HEMATOCRIET", "ERYTHROCYTEN", "LEUKOCYTEN", "TROMBOCYTEN",
        "MCV", "MCH", "MCHC", "RDW",
        "LACTATE", "BICARBONATE", "PCO2", "PH", "PO2", "O2_SATURATIE", "BASE_EXCESS",
        "BILIRUBINE", "ALAT", "ASAT", "ALK_FOSFATASE", "GAMMA_GT", "LDH", "ALBUMINE",
        "GLUCOSE", "HBA1C", "CRP", "INR", "PT", "APTT",
        "TSH", "FERRITINE", "VITAMINE_D"
    ],
    
    # Conditions/medications before reference date
    "tuple_vals_anybefore": ["ACE", "BETA", "T2DM", "ACUTE_MI", "AF"],
    
    "ref_date": "OPNAMEDATUM",

    "scaler": "scaler_hf_labs.joblib",
    "imputer": "imputer_hf_labs.joblib",

    "InitTransforms": {
        "ATCCODE_CLASSIFIED": {
            "func": "classify",
            "kwargs": {
                "classification_map": ["ace", "beta"],
                "input_col": "ATCCODE",
                "out_col": "ATCCODE_CLASSIFIED",
                "id_col": "PSEUDO_ID"
            }
        },
        "DIAGNOSECODE_CLASSIFIED": {
            "func": "classify",
            "kwargs": {
                "classification_map": ["t2dm", "af", "acute_mi"],
                "input_col": "DIAGNOSECODE",
                "out_col": "DIAGNOSECODE_CLASSIFIED",
                "id_col": "PSEUDO_ID"
            }
        },
        "FAMILIEVOORGESCHIEDENIS": {
            "func": "pattern_match",
            "kwargs": {
                "id_col": "PSEUDO_ID",
                "search_col": "FAMILIEVOORGESCHIEDENIS",
                "pattern_dict": {
                    "HF_FAM": "hart|plots"
                },
                "case_sensitive": False,
                "group_by_id": True
            }
        }
    },
    
    "PreTransforms": {
        "OPNAMEDATUM": {
            "func": "datetime_keepfirst",
            "kwargs": {
                "col_to_date": "OPNAMEDATUM",
                "sort_col": "OPNAMEDATUM",
                "drop_col": "PSEUDO_ID"
            }
        },
        "OVERLIJDENSDATUM": {
            "func": "datetime",
            "kwargs": {
                "col_to_date": "OVERLIJDENSDATUM"
            }
        },
        "ISHUIDIGEROKER": {
            "func": "keepfirst",
            "kwargs": {
                "sort_col": "PATIENTCONTACTID",
                "drop_col": "PSEUDO_ID"
            }
        },
        "SYSTOLISCHEBLOEDDRUKWAARDE": {
            "func": "keepfirst",
            "kwargs": {
                "sort_col": "PATIENTCONTACTID",
                "drop_col": "PSEUDO_ID"
            }
        },
        "BMI": {
            "func": "keepfirst",
            "kwargs": {
                "sort_col": "PATIENTCONTACTID",
                "drop_col": "PSEUDO_ID"
            }
        }
    },

    "MergedTransforms": {
        "TIME": {
            "func": "diff",
            "kwargs": {
                "end": "OVERLIJDENSDATUM",
                "start": "OPNAMEDATUM"
            }
        },
        "AGEATOPNAME": {
            "func": "diff",
            "kwargs": {
                "end": "OPNAMEDATUM",
                "start": "GEBOORTEJAAR",
                "level": "year"
            }
        },
        "GESLACHT": {
            "func": "map",
            "kwargs": {
                "map": {"Man": 0, "Vrouw": 1}
            }
        },
        "ISHUIDIGEROKER": {
            "func": "map",
            "kwargs": {
                "map": {"Nee": 0, "Ja": 1}
            }
        },
        "FillNaN": {
            "func": "fillna",
            "kwargs": {
                "values": {"ISHUIDIGEROKER": 0}
            }
        }
    }
}


# Compact version with essential labs only (for faster processing)
amc_hf_essential = {
    "name": "amc_hf_essential_process",
    "description": "Preprocessing of AMC data with essential heart failure labs only",
    "SOURCE": "SNOWFLAKE",
    "preprocessing": "/tmp/preprocessing_hf_essential",

    "snowflake_config": {
        "input_schema": "RAW_DATA",
        "output_schema": "PROCESSED_DATA",
        "database": "HEALTHCARE_DB"
    },

    "Datatools4heart_Patient": {
        "table_name": "TBL_PATIENT",
        "columns": ["PSEUDO_ID", "GEBOORTEJAAR", "GEBOORTEMAAND", "GESLACHT", "OVERLIJDENSDATUM"]
    },
    "Datatools4heart_Tabakgebruik": {
        "table_name": "TBL_TABAKGEBRUIK",
        "columns": ["PSEUDO_ID", "ISHUIDIGEROKER", "ISVOORMALIGROKER", "PATIENTCONTACTID"]
    },
    "Datatools4heart_Opnametraject": {
        "table_name": "TBL_OPNAMETRAJECT", 
        "columns": ["PSEUDO_ID", "OPNAMEDATUM"]
    },
    "Datatools4heart_MetingBMI": {
        "table_name": "TBL_METINGBMI",
        "columns": ["PSEUDO_ID", "PATIENTCONTACTID", "BMI"]
    },
    "Datatools4heart_MetingBloeddruk": {
        "table_name": "TBL_METINGBLOEDDRUK",
        "columns": ["PSEUDO_ID", "PATIENTCONTACTID", "SYSTOLISCHEBLOEDDRUKWAARDE", "DIASTOLISCHEBLOEDDRUKWAARDE"]
    },
    "Datatools4heart_VoorgeschiedenisMedisch": {
        "table_name": "TBL_VOORGESCHIEDENISMEDISCH",
        "columns": ["PSEUDO_ID", "DIAGNOSECODE_CLASSIFIED", 
                   {"DIAGNOSECODE": {"t2dm": "T2DM", "af": "AF", "acute_mi": "ACUTE_MI"}}, 
                   ["INDICATIEDATUMVASTSTELLING"]]
    },
    "Datatools4heart_VoorgeschiedenisFamilie": {
        "table_name": "TBL_VOORGESCHIEDENISFAMILIE",
        "columns": ["PSEUDO_ID", "FAMILIEVOORGESCHIEDENIS", "HF_FAM"]
    },
    "Datatools4heart_MedicatieToediening": {
        "table_name": "TBL_MEDICATIETOEDIENING",
        "columns": ["PSEUDO_ID", "ATCCODE_CLASSIFIED", 
                   {"ATCCODE": {"ace": "ACE", "beta": "BETA"}}, 
                   ["TOEDIENINGSDATUM"]]
    },
    "Datatools4heart_Labuitslag": {
        "table_name": "TBL_LABUITSLAG",
        "columns": ["PSEUDO_ID", "BEPALINGCODE", 
                   {"UITSLAGNUMERIEK": {
                       # Essential cardiac & renal markers only
                       "EBNP;BL": "NT_PRO_BNP",
                       "RKRE;BL": "CREATININE",
                       "CKD-EPI;BL": "EGFR",
                       "RNAT;BL": "NATRIUM",
                       "RKAL;BL": "KALIUM",
                       "HHB;BL": "HEMOGLOBIN",
                       "RHDL;BL": "HDL_CHOLESTEROL",
                       "RCHO;BL": "TOTAL_CHOLESTEROL"
                   }}, 
                   ["MATERIAALAFNAMEDATUM"]]
    },

    "categories": {"GESLACHT": ["Man", "Vrouw"]},

    "final_cols": [
        "GESLACHT", "AGEATOPNAME", "ISHUIDIGEROKER", "HDL_CHOLESTEROL", "TOTAL_CHOLESTEROL", "OPNAMEDATUM",
        "BMI", "SYSTOLISCHEBLOEDDRUKWAARDE", "DIASTOLISCHEBLOEDDRUKWAARDE", "ACE", "TIME", "BETA", 
        "CREATININE", "T2DM", "ACUTE_MI", "AF", "HF_FAM",
        "NT_PRO_BNP", "EGFR", "NATRIUM", "KALIUM", "HEMOGLOBIN"
    ],

    "tuple_vals_after": [
        "HDL_CHOLESTEROL", "TOTAL_CHOLESTEROL", "CREATININE", "NT_PRO_BNP", "EGFR", 
        "NATRIUM", "KALIUM", "HEMOGLOBIN",
        "SYSTOLISCHEBLOEDDRUKWAARDE", "DIASTOLISCHEBLOEDDRUKWAARDE"
    ],
    "tuple_vals_anybefore": ["ACE", "BETA", "T2DM", "ACUTE_MI", "AF"],
    "ref_date": "OPNAMEDATUM",

    "scaler": "scaler_essential.joblib",
    "imputer": "imputer_essential.joblib",

    "InitTransforms": {
        "ATCCODE_CLASSIFIED": {
            "func": "classify",
            "kwargs": {
                "classification_map": ["ace", "beta"],
                "input_col": "ATCCODE",
                "out_col": "ATCCODE_CLASSIFIED",
                "id_col": "PSEUDO_ID"
            }
        },
        "DIAGNOSECODE_CLASSIFIED": {
            "func": "classify",
            "kwargs": {
                "classification_map": ["t2dm", "af", "acute_mi"],
                "input_col": "DIAGNOSECODE",
                "out_col": "DIAGNOSECODE_CLASSIFIED",
                "id_col": "PSEUDO_ID"
            }
        },
        "FAMILIEVOORGESCHIEDENIS": {
            "func": "pattern_match",
            "kwargs": {
                "id_col": "PSEUDO_ID",
                "search_col": "FAMILIEVOORGESCHIEDENIS",
                "pattern_dict": {
                    "HF_FAM": "hart|plots"
                },
                "case_sensitive": False,
                "group_by_id": True
            }
        }
    },
    "PreTransforms": {
        "OPNAMEDATUM": {
            "func": "datetime_keepfirst",
            "kwargs": {
                "col_to_date": "OPNAMEDATUM",
                "sort_col": "OPNAMEDATUM",
                "drop_col": "PSEUDO_ID"
            }
        },
        "OVERLIJDENSDATUM": {
            "func": "datetime",
            "kwargs": {
                "col_to_date": "OVERLIJDENSDATUM"
            }
        },
        "ISHUIDIGEROKER": {
            "func": "keepfirst",
            "kwargs": {
                "sort_col": "PATIENTCONTACTID",
                "drop_col": "PSEUDO_ID"
            }
        },
        "SYSTOLISCHEBLOEDDRUKWAARDE": {
            "func": "keepfirst",
            "kwargs": {
                "sort_col": "PATIENTCONTACTID",
                "drop_col": "PSEUDO_ID"
            }
        },
        "BMI": {
            "func": "keepfirst",
            "kwargs": {
                "sort_col": "PATIENTCONTACTID",
                "drop_col": "PSEUDO_ID"
            }
        }
    },
    "MergedTransforms": {
        "TIME": {
            "func": "diff",
            "kwargs": {
                "end": "OVERLIJDENSDATUM",
                "start": "OPNAMEDATUM"
            }
        },
        "AGEATOPNAME": {
            "func": "diff",
            "kwargs": {
                "end": "OPNAMEDATUM",
                "start": "GEBOORTEJAAR",
                "level": "year"
            }
        },
        "GESLACHT": {
            "func": "map",
            "kwargs": {
                "map": {"Man": 0, "Vrouw": 1}
            }
        },
        "ISHUIDIGEROKER": {
            "func": "map",
            "kwargs": {
                "map": {"Nee": 0, "Ja": 1}
            }
        },
        "FillNaN": {
            "func": "fillna",
            "kwargs": {
                "values": {"ISHUIDIGEROKER": 0}
            }
        }
    }
}
