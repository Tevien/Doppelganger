amc_v1_full = {
    "name": "amc_process",
    "description": "Preprocessing of AMC data",
    "SOURCE": "SNOWFLAKE",
    "preprocessing": "/tmp/preprocessing",

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
        "columns": ["PSEUDO_ID", "DIAGNOSECODE_CLASSIFIED", {"DIAGNOSECODE": {"t2dm": "T2DM", "af": "AF", "acute_mi": "ACUTE_MI"}}, ["INDICATIEDATUMVASTSTELLING"]]
    },
    "Datatools4heart_VoorgeschiedenisFamilie": {
        "table_name": "TBL_VOORGESCHIEDENISFAMILIE",
        "columns": ["PSEUDO_ID", "FAMILIEVOORGESCHIEDENIS", "HF_FAM"]
    },
    "Datatools4heart_MedicatieToediening": {
        "table_name": "TBL_MEDICATIETOEDIENING",
        "columns": ["PSEUDO_ID", "ATCCODE_CLASSIFIED", {"ATCCODE": {"ace": "ACE", "beta": "BETA"}}, ["TOEDIENINGSDATUM"]]
    },
    "Datatools4heart_Labuitslag": {
        "table_name": "TBL_LABUITSLAG",
        "columns": ["PSEUDO_ID", "BEPALINGCODE", {"UITSLAGNUMERIEK": {"RKRE;BL": "CREATININE", "RHDL;BL": "HDL_CHOLESTEROL", "RCHO;BL": "TOTAL_CHOLESTEROL"}}, ["MATERIAALAFNAMEDATUM"]]
    },

    "categories": {"GESLACHT": ["Man", "Vrouw"]},

    "final_cols": ["GESLACHT", "AGEATOPNAME", "ISHUIDIGEROKER", "HDL_CHOLESTEROL", "TOTAL_CHOLESTEROL", "OPNAMEDATUM",
                   "BMI", "SYSTOLISCHEBLOEDDRUKWAARDE", "DIASTOLISCHEBLOEDDRUKWAARDE", "ACE", "TIME", "BETA", "CREATININE", 
                   "T2DM", "ACUTE_MI", "AF", "HF_FAM"],

    "tuple_vals_after": ["HDL_CHOLESTEROL", "TOTAL_CHOLESTEROL", "CREATININE", 
                        "SYSTOLISCHEBLOEDDRUKWAARDE", "DIASTOLISCHEBLOEDDRUKWAARDE"],
    "tuple_vals_anybefore": ["ACE", "BETA", "T2DM", "ACUTE_MI", "AF"],
    "ref_date": "OPNAMEDATUM",

    "scaler": "scaler.joblib",
    "imputer": "imputer.joblib",

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