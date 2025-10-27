amc_v1 = {
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

    "categories": {"GESLACHT": ["Man", "Vrouw"]},

    "final_cols": ["GESLACHT", "AGEATOPNAME", "ISHUIDIGEROKER", "OPNAMEDATUM"],

    "InitTransforms": {
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
        "ISHUIDIGEROKER": {
            "func": "keepfirst",
            "kwargs": {
                "sort_col": "PATIENTCONTACTID",
                "drop_col": "PSEUDO_ID"
            }
        }
    },

    "MergedTransforms": {
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
