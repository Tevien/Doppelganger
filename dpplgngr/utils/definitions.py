######## MEDICATION DEFINITIONS ########

# ACE inhibitors - Plain and combinations
ace_atc = [f"C09AA{n:02d}" for n in range(1,17)]  # Plain ACE inhibitors
ace_atc.extend([f"C09BA{n:02d}" for n in range(1,16)])  # ACE + diuretics
ace_atc.extend([f"C09BB{n:02d}" for n in range(2,13)])  # ACE + CCB
ace_atc.extend([f"C09BX{n:02d}" for n in range(1,6)])   # ACE + other

# Angiotensin Receptor Blockers (ARB) - Plain and combinations
arb_atc = [f"C09CA{n:02d}" for n in range(1,11)]  # Plain ARB
arb_atc.extend([f"C09DA{n:02d}" for n in range(1,11)])  # ARB + diuretics
arb_atc.extend([f"C09DB{n:02d}" for n in range(1,10)])  # ARB + CCB
# ARB + other (excluding ARNI which is C09DX04)
arb_other = [f"C09DX{n:02d}" for n in range(1,8)]
arb_other.remove("C09DX04")  # Remove ARNI
arb_atc.extend(arb_other)

# Renin-Angiotensin System (RAS) inhibitors - broader category
ras_atc = ["C09AA", "C09B", "C09CA", "C09D", "C09X"]  # Parent codes

# Angiotensin Receptor-Neprilysin Inhibitor (ARNI)
arni_atc = ["C09DX04"]  # sacubitril/valsartan

# Beta-blockers - comprehensive list
beta_atc = {
    "Atenolol": "C07AB03",
    "Bisoprolol": "C07AB07",
    "Metoprolol": "C07AB02",
    "Propranolol": "C07AA05",
    "Carvedilol": "C07AG02",
    "Nebivolol": "C07AB12"
}

# Beta-blockers - all categories
beta_all_atc = ["C07AA", "C07AB", "C07AG"]  # Plain beta blockers
beta_all_atc.extend(["C07BA", "C07BB", "C07BG"])  # + thiazides
beta_all_atc.extend(["C07CA", "C07CB", "C07CG"])  # + diuretics
beta_all_atc.extend(["C07DA", "C07DB", "C07DG"])  # + other
beta_all_atc.extend(["C07FB", "C07FX"])  # + RAS

# Mineralocorticoid Receptor Antagonists (MRA)
mra_atc = [f"C03DA{n:02d}" for n in range(1,6)]
mra_atc.extend([f"C03EA{n:02d}" for n in range(1,15)])
mra_atc.extend(["C03EB01", "C03EB02"])  # combination potassium sparing

# Diuretics - comprehensive
diuretics_atc = ["C03A", "C03C", "C03D"]  # Main categories
# Loop diuretics
loop_diuretics = [f"C03CA{n:02d}" for n in range(1,5)]
loop_diuretics.extend(["C03CB01", "C03CB02", "C03CC01", "C03CC02", "C03CX01"])
# Thiazide diuretics
thiazide_diuretics = ["C03A"]
# Potassium-sparing agents
potassium_sparing = ["C03D"]

# SGLT-2 inhibitors
sglt2_hf_atc = ["A10BX09", "A10BK01", "A10BX12", "A10BK03"]  # For HF patients
sglt2_other_atc = ["A10BX11", "A10BK02", "A10BK04", "A10BK05", "A10BK06"]  # Other

# Anticoagulants
anticoagulant_atc = ["B01AA", "B01AB", "B01AX05", "B01AX04", "B01AF", "B01AE", 
                     "B01AX01", "B01AD12", "B01AB02"]

# Antiplatelet agents
antiplatelet_atc = ["B01AC13", "B01AC17", "B01AC16", "B01AC04", "B01AC22", "B01AC05",
                    "B01AC25", "B01AC24", "B01AC19", "B01AC11", "B01AC21", "B01AC06",
                    "B01AC15", "B01AC08", "B01AC10", "B01AC18", "B01AC07", "B01AC03",
                    "B01AC23", "B01AC01", "B01AC02", "B01AC26"]

# Statins
statins_atc = ["C10AA", "C10BA", "C10BX"]
# Specific statins
specific_statins = {
    "Atorvastatin": "C10AA05",
    "Simvastatin": "C10AA01",
    "Rosuvastatin": "C10AA07",
    "Pravastatin": "C10AA03"
}

# Digitalis glycosides
digitalis_atc = [f"C01AA{n:02d}" for n in range(1,9)]

# Ivabradine
ivabradine_atc = ["C01EB17"]

# Calcium Channel Blockers
ccb_atc = ["C08C", "C08D", "C08E", "C08GA"]

######## DISEASE DEFINITIONS ########

# Heart Failure - ICD-10
heart_failure_icd10 = ["I50", "I11.0", "I13.0", "I13.2", "I26.0", "I25.5", 
                       "I09.81", "I97.13"]

# Heart Failure - ICD-9
heart_failure_icd9 = ["428", "404.01", "404.03", "404.11", "404.13", "404.91", 
                      "404.93", "402.01", "402.11", "402.91", "415.0", "418.8", 
                      "398.91", "429.4"]

# Atrial Fibrillation/Flutter
af_icd_10 = ["I48"]
af_icd_9 = ["427.3", "427.31", "427.32"]

# Angina Pectoris
angina_icd10 = ["I20"]
angina_icd9 = ["411", "413"]

# Myocardial Infarction
mi_icd10 = ["I21", "I22"]
mi_icd9 = ["410", "412"]

# Chronic Ischemic Heart Disease
chron_ischemic_hd_icd_10 = ["I20", "I21", "I22", "I23", "I24", "I25"]
chron_ischemic_hd_icd_9 = ["410", "411", "412", "413", "414", "429"]

# Hypertension
hypertension_icd10 = ["I10", "I11", "I12", "I13", "I15"]
hypertension_icd9 = ["401", "402", "403", "404", "405"]

# Diabetes Mellitus
diabetes_icd10 = ["E10", "E11", "E12", "E13", "E14"]
diabetes_icd9 = ["249", "250"]

# Type 2 Diabetes - detailed
t2dm_codes = [
    "E11.9",   # without complications
    "E11.21",  # diabetic nephropathy
    "E11.22",  # diabetic CKD
    "E11.29",  # other kidney complication
    "E11.31", "E11.32", "E11.33", "E11.34", "E11.35", "E11.36", "E11.39",  # retinopathy
    "E11.40", "E11.41", "E11.42", "E11.43", "E11.44", "E11.49",  # neuropathy
    "E11.51", "E11.52", "E11.59",  # circulatory complications
    "E11.65",  # hyperglycemia
    "E11.69",  # other specified complication
    "E11.8"    # unspecified complications
]

# Chronic Kidney Disease
ckd_icd10 = ["N18", "N19"]
ckd_icd9 = ["585", "586"]

# Stroke
stroke_icd10 = ["I61", "I62", "I63", "I64", "I60"]
stroke_icd9 = ["430", "431", "432", "433.01", "433.11", "433.21", "433.31", 
               "433.81", "433.91", "434.01", "434.11", "434.91", "436"]

# Transient Ischemic Attack
tia_icd10 = ["G45.8", "G45.9", "I63.9"]
tia_icd9 = ["435", "435.8"]

# Peripheral Artery Disease
pad_icd10 = ["I73", "I70", "I71", "I72", "I74", "I75", "I77", "I78", "I79"]
pad_icd9 = ["443", "440", "441", "442", "444", "445", "446", "447", "448"]

# Cardiomyopathy
cardiomyopathy_icd10 = ["I42", "I43"]
cardiomyopathy_icd9 = ["425"]

# Dilated Cardiomyopathy
dilated_cm_icd10 = ["I42.0"]
dilated_cm_icd9 = ["425.4"]

# Valvular Disease
valvular_icd10 = ["A520", "I05", "I06", "I07", "I08", "I09.1", "I09.8", "I34", "I35", 
                  "I36", "I37", "I38", "I39", "Q23.0", "Q23.1", "Q23.2", "Q23.3", 
                  "Z95.2", "Z95.3", "Z95.4"]
valvular_icd9 = ["093.2", "746.3", "746.4", "746.5", "746.6", "V42.2", "V43.3", 
                 "394", "395", "396", "397", "424", "746"]

# COPD
copd_icd10 = ["J40", "J41", "J42", "J43", "J44"]
copd_icd9 = ["490", "491", "492", "494", "495", "496"]

# Depression
depression_icd10 = ["F20.4", "F31.3", "F31.4", "F31.5", "F32", "F33", "F34.1", 
                    "F41.2", "F43.2"]
depression_icd9 = ["296.2", "296.3", "296.5", "300.4", "309", "311"]

# Dementia
dementia_icd10 = ["F00", "F01", "F02", "F03", "F05.1", "G30", "G31.1"]
dementia_icd9 = ["290"]

# Liver Disease
liver_disease_icd10 = ["B18", "I85", "I86.4", "I98.2", "K70", "K71.1", "K71.3", 
                       "K71.4", "K71.5", "K71.7", "K72", "K73", "K74", "K76.0", 
                       "K76.2", "K76.3", "K76.4", "K76.5", "K76.6", "K76.7", 
                       "K76.8", "K76.9", "Z94.4"]
liver_disease_icd9 = ["070.22", "070.23", "070.32", "070.33", "070.44", "070.54", 
                      "070.6", "070.9", "456.0", "456.1", "456.2", "570", "571", 
                      "572.2", "572.3", "572.4", "572.5", "572.6", "572.7", "572.8", 
                      "573.3", "573.4", "573.8", "573.9", "V42.7"]

######## OUTCOME DEFINITIONS ########

# Heart Failure Hospitalization/ED Visit
hf_hosp_icd10 = ["I50", "I11", "I13.0", "I13.2", "I26.0", "I09.81", "I97.13"]
hf_hosp_icd9 = ["428", "404.01", "404.03", "404.11", "404.13", "404.91", "404.93", 
                "402.01", "402.11", "402.91", "415.0"]

# Major Adverse Cardiovascular Events (MACE) components
mace_components = {
    "acute_mi": mi_icd10,
    "stroke": stroke_icd10,
    "hf_hosp": hf_hosp_icd10
}

######## MAPPING DICTIONARY ########

defs_map = {
    # Medications
    "ace": ace_atc,
    "arb": arb_atc,
    "ras": ras_atc,
    "arni": arni_atc,
    "beta": list(beta_atc.values()),
    "beta_all": beta_all_atc,
    "mra": mra_atc,
    "diuretics": diuretics_atc,
    "loop_diuretics": loop_diuretics,
    "thiazide_diuretics": thiazide_diuretics,
    "sglt2_hf": sglt2_hf_atc,
    "sglt2_other": sglt2_other_atc,
    "anticoagulant": anticoagulant_atc,
    "antiplatelet": antiplatelet_atc,
    "statins": statins_atc,
    "digitalis": digitalis_atc,
    "ivabradine": ivabradine_atc,
    "ccb": ccb_atc,
    
    # Cardiovascular Conditions
    "heart_failure": heart_failure_icd10,
    "af": af_icd_10,
    "angina": angina_icd10,
    "acute_mi": mi_icd10,
    "chronic_ischemic_hd": chron_ischemic_hd_icd_10,
    "hypertension": hypertension_icd10,
    "stroke": stroke_icd10,
    "tia": tia_icd10,
    "pad": pad_icd10,
    "cardiomyopathy": cardiomyopathy_icd10,
    "dilated_cm": dilated_cm_icd10,
    "valvular": valvular_icd10,
    
    # Other Conditions
    "diabetes": diabetes_icd10,
    "t2dm": t2dm_codes,
    "ckd": ckd_icd10,
    "copd": copd_icd10,
    "depression": depression_icd10,
    "dementia": dementia_icd10,
    "liver_disease": liver_disease_icd10,
    
    # Outcomes
    "hf_hosp": hf_hosp_icd10,
    "mace": mace_components
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

def get_hf_medication_bundle():
    """
    Return the guideline-directed medical therapy (GDMT) bundle for heart failure.
    Based on current ESC guidelines.
    """
    return {
        "ace_arb_arni": ace_atc + arb_atc + arni_atc,
        "beta_blockers": beta_all_atc,
        "mra": mra_atc,
        "sglt2i": sglt2_hf_atc
    }

def get_ckd_stages():
    """
    Return eGFR thresholds for CKD staging as per protocol.
    """
    return {
        "normal": ">=90",
        "mild": "60-89", 
        "moderate": "15-59",
        "advanced": "<15"
    }

def get_hyperkalemia_stages():
    """
    Return potassium level thresholds for hyperkalemia staging as per protocol.
    """
    return {
        "mild": "5.0-5.5 mmol/L",
        "moderate": "5.5-6.0 mmol/L", 
        "severe": ">6.0 mmol/L"
    }