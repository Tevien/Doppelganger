######## MEDICATION DEFINITIONS ########

# ACE inhibitors
ace_atc = [f"C09AA{n:02d}" for n in range(1,17)]
ace_atc.extend([f"C09BA{n:02d}" for n in range(1,16)])
ace_atc.extend([f"C09BB{n:02d}" for n in range(2,13)])
ace_atc.extend([f"C09BX{n:02d}" for n in range(1,6)])

# beta_blockers
beta_atc = {
"Atenolol": "C07AB03",
"Bisoprolol": "C07AB07",
"Metoprolol": "C07AB02",
"Propranolol": "C07AA05",
"Carvedilol": "C07AG02",
"Nebivolol": "C07AB12"
}

######## DISEASE DEFINITIONS ########

af_icd_10 = [f"I48.{n:01d}" for n in range(1, 6)]
af_icd_10.append("I48.9")  # Adding unspecified AF

chron_ischemic_hd_icd_10 = ["I25.1", "I25.2", "I25.3", "I25.5", "I25.6", "I25.9"]

acute_mi_icd_10 = ["I21.0", "I21.1", "I21.2", "I21.3", "I21.4" "I21.9"]

# Type 2 Diabetes ICD-10 Codes
t2dm_codes = [
    "E11.9",   # without complications
    "E11.21",  # diabetic nephropathy
    "E11.22",  # diabetic CKD
    "E11.29",  # other kidney complication
    [f"E11.3{i}" for i in [1,2,3,4,5,6,9]],  # retinopathy/eye complications
    [f"E11.4{i}" for i in [0,1,2,3,4,9]],    # neuropathy complications
    [f"E11.5{i}" for i in [1,2,9]],          # circulatory complications
    "E11.65",  # hyperglycemia
    "E11.69",  # other specified complication
    "E11.8"    # unspecified complications
]



defs_map = {
    "ace": ace_atc,
    "beta": beta_atc.values(),
    "af": af_icd_10,
    "chronic_ischemic_hd": chron_ischemic_hd_icd_10,
    "acute_mi": acute_mi_icd_10,
    "t2dm": t2dm_codes
}

def make_classification_map(l_keys):
    """
    Create a classification map from a list of keys.
    The keys are used to create a dictionary where the key is the classification name
    and the value is a list of values that belong to that classification.
    """
    classification_map = {}
    for key in l_keys:
        if key in defs_map:
            classification_map[key] = defs_map[key]
        else:
            raise ValueError(f"Key {key} not found in defs_map")
    return classification_map