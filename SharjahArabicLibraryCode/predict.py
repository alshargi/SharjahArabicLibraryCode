
# -*- coding: utf-8 -*-
# Done By 
## Dr. Faisal ALSHARGI
## New York, US, May 06, 2024
##############################


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
import numpy as np
import pandas as pd

FinalResult = []


TOP_labels = {
    0: 'A  GENERAL WORKS',
    1: 'B  PHILOSOPHY. PSYCHOLOGY. RELIGION',
    2: 'C  AUXILIARY SCIENCES OF HISTORY',
    3: 'D  WORLD HISTORY AND HISTORY OF EUROPE, ASIA, AFRICA, AUSTRALIA, NEW ZEALAND, ETC.',
    4: 'E  HISTORY OF THE AMERICAS CONTENANT',
    5: 'F  HISTORY OF THE AMERICAS LOCAL',
    6: 'G  GEOGRAPHY. ANTHROPOLOGY. RECREATION',
    7: 'H  SOCIAL SCIENCES',
    8: 'J  POLITICAL SCIENCE',
    9: 'K  LAW',
    10: 'L  EDUCATION',
    11: 'M  MUSIC',
    12: 'N  FINE ARTS',
    13: 'P  LANGUAGE AND LITERATURE',
    14: 'Q  SCIENCE',
    15: 'R  MEDICINE',
    16: 'S  AGRICULTURE',
    17: 'T  TECHNOLOGY',
    18: 'U  MILITARY SCIENCE',
    19: 'V  NAVAL SCIENCE',
    20: 'W  MEDICINE AND RELATED SUBJECTS',
    21: 'Z  BIBLIOGRAPHY. LIBRARY SCIENCE. INFORMATION RESOURCES'
}



A_labels = {
    0 : "A2  BOOKS",
    1 : "AC  COLLECTIONS. SERIES. COLLECTED WORKS",
    2 : "AE  ENCYCLOPEDIAS",
    3 : "AG  DICTIONARIES AND OTHER GENERAL REFERENCE WORKS",
    4 : "AI  INDEXES",
    5 : "AM  MUSEUMS. COLLECTORS AND COLLECTING",
    6 : "AS  CADEMIES AND LEARNED SOCIETIES",
    7 : "AZ  HISTORY OF SCHOLARSHIP AND LEARNING. THE HUMANITIES"
}


B_labels = {
    0 : "B1  Philosophy (General)—Periodicals. Serials—English and American",
    1 : "B2  Philosophy (General)—Periodicals. Serials—French and Belgian",
    2 : "B3  Philosophy (General)—Periodicals. Serials—German",
    3 : "B4  Philosophy (General)—Periodicals. Serials—Italian",
    4 : "B5  Philosophy (General)—Periodicals. Serials—Spanish and Portuguese",
    5 : "B6  Philosophy (General)—Periodicals. Serials—Russian and other Slavic",
    6 : "B7  N/A",
    7 : "B8  Philosophy (General)—Periodicals. Serials—Other. By language, A-Z",
    8 : "B9  N/A",
    9 : "BC  LOGIC",
    10 :"BD  SPECULATIVE PHILOSOPHY",
    11 : "BF  PSYCHOLOGY",
    12 : "BH  AESTHETICS",
    13 : "BJ  ETHICS",
    14 : "BL  RELIGIONS. MYTHOLOGY. RATIONALISM",
    15 : "BM  JUDAISM",
    16 : "BP  ISLAM. BAHAISM. THEOSOPHY, ETC.",
    17 : "BQ  BUDDHISM",
    18 : "BR  CHRISTIANITY",
    19 : "BS  THE BIBLE",
    20 : "BT  DOCTRINAL THEOLOGY",
    21 : "BV  PRACTICAL THEOLOGY",
    22 : "BX  CHRISTIAN DENOMINATIONS"
}


C_labels = {
    0: 'CB  HISTORY OF CIVILIZATION',
    1: 'CC  ARCHAEOLOGY',
    2: 'CD  DIPLOMATICS. ARCHIVES. SEALS',
    3: 'CE  TECHNICAL CHRONOLOGY. CALENDAR',
    4: 'CJ  NUMISMATICS',
    5: 'CN  INSCRIPTIONS. EPIGRAPHY',
    6: 'CR  HERALDRY',
    7: 'CS  GENEALOGY',
    8: 'CT  BIOGRAPHY'
}




D_labels = {
    0 : 'D1    History (General)—Periodicals. Societies. Serials',
    1 : 'D2    History (General)—Registers',
    2 : 'D3    History (General)—Congresses, conferences, etc.',
    3 : 'D4    N/A',
    4 : 'D5    History (General)—Sources and documents. Collections—General works',
    5 : 'D6    History (General)—Collected works (Monographs, essays, etc.). Festschriften—Several authors',
    6 : 'D7    History (General)—Collected works (Monographs, essays, etc.). Festschriften—Individual authors',
    7 : 'D8    History (General)—Pamphlets, etc.',
    8 : 'D9    History (General)—Dictionaries',
    9 : 'DA    GREAT BRITAIN',
    10 :'DB     AUSTRIA - LIECHTENSTEIN - HUNGARY - CZECHOSLOVAKIA',
    11 :'DC     FRANCE - ANDORRA - MONACO',
    12 :'DD     GERMANY',
    13 :'DE     GRECO-ROMAN WORLD',
    14 :'DF     GREECE',
    15 :'DG     ITALY - MALTA',
    16 :'DJ     NETHERLANDS (HOLLAND)',
    17 :'DK      RUSSIA. SOVIET UNION. FORMER SOVIET REPUBLICS - POLAND',
    18 :'DL      NORTHERN EUROPE. SCANDINAVIA',
    19 :'DP      SPAIN - PORTUGAL',
    20 :'DR      BALKAN PENINSULA',
    21 :'DS      ASIA',
    22 :'DT      AFRICA',
    23 :'DU      OCEANIA (SOUTH SEAS)',
    24 :'DX      ROMANIES'
}


F_labels = {
    0: 'F1  United States local history—New England—Periodicals. Societies. Collections',
    1: 'F2  United States local history—New England—Gazetteers. Dictionaries. Geographic names'
}


E_labels = {
    0: 'E1  N/A',
    1: 'E2  N/A',
    2: 'E3  N/A',
    3: 'E4  N/A',
    4: 'E5  N/A',
    5: 'E7  N/A',
    6: 'E8  N/A',
    7: 'E9  N/A'
}


G_labels = {
0 : 'G1  Geography (General)—Periodicals. Serials',
1 : 'G2  Geography (General)—Societies—International',
2 : 'G3  Geography (General)—Societies—United States',
3 : 'G4  Geography (General)—Societies—Canada',
4 : 'G5  Geography (General)—Societies—Mexico. Central America. West Indies',
5 : 'G6  Geography (General)—Societies—South America',
6 : 'G7  Geography (General)—Societies—Great Britain',
7 : 'G8  Geography (General)—Societies—Czechoslovakia',
8 : 'G9  Geography (General)—Societies—Austria',
9 : 'GA  Mathematical geography. Cartography',
10 : 'GB  Physical geography',
11 : 'GC  Oceanography',
12 : 'GE  Environmental Sciences',
13 : 'GF  Human ecology. Anthropogeography',
14 : 'GN  Anthropology',
15 : 'GR  Folklore',
16 : 'GT  Manners and customs (General)',
17 : 'GV  Recreation. Leisure'
}

H_labels = {
    0 : 'H3  Social sciences (General)—Periodicals. Serials—French',
    1 : 'H4  N/A',
    2 : 'H5  Social sciences (General)—Periodicals. Serials—German',
    3 : 'H6  xx',
    4 : 'H8  Social sciences (General)—Periodicals. Serials—Other languages, A-Z',
    5 : 'H9  N/A see H1-8',
    6 : 'HA  STATISTICS',
    7 : 'HB  ECONOMIC THEORY. DEMOGRAPHY',
    8 : 'HC  ECONOMIC HISTORY AND CONDITIONS',
    9 : 'HD  INDUSTRIES. LAND USE. LABOR',
    10 : 'HE  TRANSPORTATION AND COMMUNICATIONS',
    11 : 'HF  COMMERCE',
    12 : 'HG  FINANCE',
    13 : 'HJ  PUBLIC FINANCE',
    14 : 'HM  SOCIOLOGY (GENERAL)',
    15 : 'HN  SOCIAL HISTORY AND CONDITIONS. SOCIAL PROBLEMS. SOCIAL REFORM',
    16 : 'HQ  THE FAMILY. MARRIAGE. WOMEN',
    17 : 'HS  SOCIETIES: SECRET, BENEVOLENT, ETC.',
    18 : 'HT  COMMUNITIES. CLASSES. RACES',
    19 : 'HV  SOCIAL PATHOLOGY. SOCIAL AND PUBLIC WELFARE. CRIMINOLOGY',
    20 : 'HX  Socialism. Communism. Utopias. Anarchism—Socialism. Communism'
}



J_labels = {
    0: 'J7  N/A see class K',
    1: 'JA  POLITICAL SCIENCE (GENERAL)',
    2: 'JC  POLITICAL THEORY',
    3: 'JF  POLITICAL INSTITUTIONS AND PUBLIC ADMINISTRATION',
    4: 'JK  POLITICAL INSTITUTIONS AND PUBLIC ADMINISTRATION (UNITED STATES)',
    5: 'JL  POLITICAL INSTITUTIONS AND PUBLIC ADMINISTRATION (CANADA, LATIN AMERICA, ETC.)',
    6: 'JN  POLITICAL INSTITUTIONS AND PUBLIC ADMINISTRATION (EUROPE)',
    7: 'JQ  POLITICAL INSTITUTIONS AND PUBLIC ADMINISTRATION (ASIA, AFRICA, ACIFIC AREA, ETC.)',
    8: 'JS  LOCAL GOVERNMENT. MUNICIPAL GOVERNMENT',
    9: 'JV  COLONIES AND COLONIZATION. EMIGRATION AND IMMIGRATION. INTERNATIONAL MIGRATION',
    10: 'JX INTERNATIONAL LAW, SEE JZ AND KZ (OBSOLETE)',
    11: 'JZ  INTERNATIONAL RELATIONS'
}



K_labels = {
0: 'K1  Law in general. Comparative and uniform law. Jurisprudence—Periodicals—By main entry—A',
1: 'K2  Law in general. Comparative and uniform law. Jurisprudence—Periodicals—By main entry—B',
2: 'K3  Law in general. Comparative and uniform law. Jurisprudence—Periodicals—By main entry—C',
3: 'K4  Law in general. Comparative and uniform law. Jurisprudence—Periodicals—By main entry—D',
4: 'K5  Law in general. Comparative and uniform law. Jurisprudence—Periodicals—By main entry—E',
5: 'K6  Law in general. Comparative and uniform law. Jurisprudence—Periodicals—By main entry—F',
6: 'K7  Law in general. Comparative and uniform law. Jurisprudence—Periodicals—By main entry—G',
7: 'K8  Law in general. Comparative and uniform law. Jurisprudence—Periodicals—By main entry—H',
8: 'K9  Law in general. Comparative and uniform law. Jurisprudence—Periodicals—By main entry—I',
9: 'KB  RELIGIOUS LAW IN GENERAL. COMPARATIVE RELIGIOUS LAW. JURISPRUDENCE',
10: 'KD  UNITED KINGDOM AND IRELAND',
11: 'KF  UNITED STATES',
12: 'KJ  EUROPE',
13: 'KK  Law of Germany Legal publications related to the laws of Germany',
14: 'KL  ASIA AND EURASIA, AFRICA, PACIFIC AREA, AND ANTARCTICA',
15: 'KM  Middle East.  Southwest Asia',
16: 'KN  South Asia.  Southeast Asia.  East Asia',
17: 'KP  South Asia.  Southeast Asia.  East Asia',
18: 'KQ  Law of the Air, Space and Antarctica Specialized laws governing air, space, and Antarctic territories/activities.',
19: 'KR  Law of Scotland Dedicated to legal sources pertaining to the laws of Scotland.',
20: 'KS  Law of Hispanic Countries Laws and legal guides related to Spanish-speaking countries in the Americas.',
21: 'KT  Law of Canada Legal publications focused on the laws and legislation of Canada.',
22: 'KZ  Law of Nations/International Law'
}


L_labels = {
    0: 'L1  Education (General)',
    1: 'L5  N/A',
    2: 'LA  HISTORY OF EDUCATION',
    3: 'LB  THEORY AND PRACTICE OF EDUCATION',
    4: 'LC  SPECIAL ASPECTS OF EDUCATION',
    5: 'LG  INDIVIDUAL INSTITUTIONS - ASIA, AFRICA, INDIAN OCEAN ISLANDS, AUSTRALIA, NEW ZEALAND, PACIFIC ISLAND'
}

M_labels = {
0: 'M1  XX',
1: 'ML  LITERATURE ON MUSIC',
2: 'MT  INSTRUCTION AND STUDY'
 }

N_labels = {
    0: 'N1  N1.A1. Visual arts—Periodicals—Polyglot. N1.A12-Z. Visual arts—Periodicals—American and English',
    1: 'N2  Visual arts—Periodicals—French',
    2: 'N3  Visual arts—Periodicals—German',
    3: 'N4  Visual arts—Periodicals—Italian',
    4: 'N5  Visual arts—Periodicals—Dutch and Flemish',
    5: 'N6  Visual arts—Periodicals—Russian. Slavic',
    6: 'N7  Visual arts—Periodicals—Spanish and Portuguese',
    7: 'N8  Visual arts—Periodicals—Other (including Oriental), A-Z',
    8: 'NA  ARCHITECTURE',
    9: 'NB  SCULPTURE',
    10:'NC  DRAWING. DESIGN. ILLUSTRATION',
    11:'ND  PAINTING',
    12:'NE  PRINT MEDIA',
    13:'NK  DECORATIVE ARTS',
    14:'NX  ARTS IN GENERAL'
}


P_labels = {
    0: 'P1  P1.A1. Philology. Linguistics—Periodicals. Serials—International or polyglot. P1.A3-Z. American and English',
    1: 'P2  Philology. Linguistics—Periodicals. Serials—French',
    2: 'P3  Philology. Linguistics—Periodicals. Serials—German',
    3: 'P4  N/A',
    4: 'P5  N/A',
    5: 'P6  N/A',
    6: 'P7  Philology. Linguistics—Periodicals. Serials—Scandinavian',
    7: 'P8  N/A',
    8: 'P9  N/A',
    9: 'PA  GREEK LANGUAGE AND LITERATURE. LATIN LANGUAGE AND LITERATURE',
    10: 'PB  MODERN LANGUAGES. CELTIC LANGUAGES',
    11: 'PC  ROMANCE LANGUAGES',
    12: 'PE  ENGLISH LANGUAGE',
    13: 'PF  WEST GERMANIC LANGUAGES',
    14: 'PG  SLAVIC LANGUAGES. BALTIC LANGUAGES. ALBANIAN LANGUAGE',
    15: 'PJ  ORIENTAL LANGUAGES AND LITERATURES',
    16: 'PK  INDO-IRANIAN LANGUAGES AND LITERATURES',
    17: 'PL  LANGUAGES AND LITERATURES OF EASTERN ASIA, AFRICA, OCEANIA',
    18: 'PN  LITERATURE (GENERAL)',
    19: 'PQ  FRENCH LITERATURE - ITALIAN LITERATURE - SPANISH LITERATURE - PORTUGUESE LITERATURE',
    20: 'PR  ENGLISH LITERATURE',
    21: 'PS  AMERICAN LITERATURE',
    22: 'PT  GERMAN LITERATURE - DUTCH LITERATURE - FLEMISH LITERATURE SINCE 1830 - AFRIKAANS LITERATURE - SCANDI',
    23: 'PZ  FICTION AND JUVENILE BELLES LETTRES'
}

Q_labels = {
    0: 'Q1  Q1.A1. Science (General)—Periodicals. By language of publication—Polyglot. Q1.A3-Z. English',
    1: 'Q2  Science (General)—Periodicals. By language of publication—French',
    2: 'Q3  Science (General)—Periodicals. By language of publication—German',
    3: 'Q8  N/A',
    4: 'QA  MATHEMATICS',
    5: 'QB  ASTRONOMY',
    6: 'QC  PHYSICS',
    7: 'QD  CHEMISTRY',
    8: 'QE  GEOLOGY',
    9: 'QH  NATURAL HISTORY - BIOLOGY',
    10: 'QK  BOTANY',
    11: 'QL  ZOOLOGY',
    12: 'QM  HUMAN ANATOMY',
    13: 'QP  PHYSIOLOGY',
    14: 'QR  MICROBIOLOGY',
    15: 'QS  HUMAN ANATOMY',
    16: 'QT  PHYSIOLOGY',
    17: 'QU  BIOCHEMISTRY',
    18: 'QV  PHARMACOLOGY',
    19: 'QW  MICROBIOLOGY',
    20: 'QX  PARASITOLOGY',
    21: 'QY  CLINICAL PATHOLOGY',
    22: 'QZ  PATHOLOGY'
}


R_labels = {
    0: 'R8  xx',
    1: 'RA  PUBLIC ASPECTS OF MEDICINE',
    2: 'RB  PATHOLOGY',
    3: 'RC  INTERNAL MEDICINE',
    4: 'RD  SURGERY',
    5: 'RJ  PEDIATRICS',
    6: 'RM  THERAPEUTICS. PHARMACOLOGY',
    7: 'RS  PHARMACY AND MATERIA MEDICA'
}



S_labels = {
0: 'S4  N/A',
1: 'S5  Agriculture (General)—Periodicals. By language of publication—French',
2: 'S6  N/A',
3: 'SB  PLANT CULTURE',
4: 'SD  FORESTRY',
5: 'SF  ANIMAL CULTURE',
6: 'SH  AQUACULTURE. FISHERIES. ANGLING',
7: 'SK  HUNTING SPORTS'
}


T_labels = {
    0: 'T1  Technology (General)—Periodicals and societies. By language of publication—English',
    1: 'T2  Technology (General)—Periodicals and societies. By language of publication—French',
    2: 'T3  Technology (General)—Periodicals and societies. By language of publication—German',
    3: 'T4  Technology (General)—Periodicals and societies. By language of publication—Other languages (not A-Z)',
    4: 'T5  (T5) NOT USED Yearbooks. See T1-4',
    5: 'T6  Technology (General)—Congresses',
    6: 'T9  Technology (General)—Dictionaries and encyclopedias—General works',
    7: 'TA  ENGINEERING (GENERAL). CIVIL ENGINEERING',
    8: 'TC  HYDRAULIC ENGINEERING. OCEAN ENGINEERING',
    9: 'TD  ENVIRONMENTAL TECHNOLOGY. SANITARY ENGINEERING',
    10: 'TE  HIGHWAY ENGINEERING. ROADS AND PAVEMENTS',
    11: 'TF  RAILROAD ENGINEERING AND OPERATION',
    12: 'TG  BRIDGE ENGINEERING',
    13: 'TH  BUILDING CONSTRUCTION',
    14: 'TJ  MECHANICAL ENGINEERING AND MACHINERY',
    15: 'TK  ELECTRICAL ENGINEERING. ELECTRONICS. NUCLEAR ENGINEERING',
    16: 'TL  MOTOR VEHICLES. AERONAUTICS. ASTRONAUTICS',
    17: 'TN  MINING ENGINEERING. METALLURGY',
    18: 'TP  CHEMICAL TECHNOLOGY',
    19: 'TR  PHOTOGRAPHY',
    20: 'TS  MANUFACTURES',
    21: 'TT  HANDICRAFTS. ARTS AND CRAFTS',
    22: 'TX  HOME ECONOMICS'
}

U_labels = {
    0: 'U1  Military science (General)—Periodicals and societies. By language of publication—English',
    1: 'U2  Military science (General)—Periodicals and societies. By language of publication—French',
    2: 'U3  Military science (General)—Periodicals and societies. By language of publication—German',
    3: 'U4  Military science (General)—Periodicals and societies. By language of publication—Other languages (not A-Z)',
    4: 'U5  N/A',
    5: 'U6  N/A',
    6: 'U7  Military science (General)—Congresses',
    7: 'U8  N/A',
    8: 'UA  ARMIES: ORGANIZATION, DISTRIBUTION, MILITARY SITUATION',
    9: 'UB  MILITARY ADMINISTRATION',
    10: 'UC  MAINTENANCE AND TRANSPORTATION',
    11: 'UD  INFANTRY',
    12: 'UE  CAVALRY. ARMOR',
    13: 'UF  ARTILLERY',
    14: 'UG  MILITARY ENGINEERING. AIR FORCES'
}

V_labels = {
    0: 'V1  Naval science (General)—Periodicals and societies. By language of publication—English',
    1: 'V2  Naval science (General)—Periodicals and societies. By language of publication—French',
    2: 'V4  Naval science (General)—Periodicals and societies. By language of publication—Italian',
    3: 'V5  Naval science (General)—Periodicals and societies. By language of publication—Other languages (not A-Z)',
    4: 'V6  N/A',
    5: 'VK  NAVIGATION. MERCHANT MARINE',
    6: 'VM  NAVAL ARCHITECTURE. SHIPBUILDING. MARINE ENGINEERING'
}


W_labels = {
    0: 'W1  General Medicine. Health Professions. Reference Works. General Works. Serials. Periodicals',
    1: 'W2  General Medicine. Health Professions. Reference Works. General Works. Serial documents (Table-G)',
    2: 'W3  General Medicine. Health Professions. Reference Works. General Works. Congresses',
    3: 'W4  General Medicine. Health Professions. Reference Works. General Works. Dissertations on clinical research by university.',
    4: 'W5  General Medicine. Health Professions. Reference Works. General Works. Collections by several authors',
    5: 'W6  General Medicine. Health Professions. Reference Works. General Works. Collected pamphlet vol. Miscellanous reprint vol.',
    6: 'W7  General Medicine. Health Professions. Reference Works. General Works. Collected works by individual authors',
    7: 'W8  N/A',
    8: 'WA  PUBLIC HEALTH',
    9: 'WB  PRACTICE OF MEDICINE',
    10: 'WC  COMMUNICABLE DISEASES',
    11: 'WD  DISORDERS OF SYSTEMIC, METABOLIC, OR ENVIRONMENTAL ORIGIN',
    12: 'WE  MUSCULOSKELETAL SYSTEM',
    13: 'WF  RESPIRATORY SYSTEM',
    14: 'WG  CARDIOVASCULAR SYSTEM',
    15: 'WH  HEMIC AND LYMPHATIC SYSTEMS',
    16: 'WI  DIGESTIVE SYSTEM',
    17: 'WJ  UROGENITAL SYSTEM',
    18: 'WK  ENDOCRINE SYSTEM',
    19: 'WL  NERVOUS SYSTEM',
    20: 'WM  PSYCHIATRY',
    21: 'WN  RADIOLOGY, NUCLEAR MEDICINE, AND MEDICAL IMAGING',
    22: 'WO  SURGERY',
    23: 'WP  GYNECOLOGY AND OBSTETRICS',
    24: 'WQ  VETERINARY SCIENCE',
    25: 'WR  DERMATOLOGY',
    26: 'WS  PEDIATRICS',
    27: 'WT  GERIATRICS; CHRONIC DISEASE',
    28: 'WU  DENTISTRY',
    29: 'WV  OTOLARYNGOLOGY (EAR, NOSE, THROAT)',
    30: 'WW  OPHTHALMOLOGY (EYE)',
    31: 'WX  HOSPITALS AND OTHER HEALTH FACILITIES',
    32: 'WY  NURSING',
    33: 'WZ  HISTORY OF MEDICINE, MEDICAL ETHICS, HEALTH POLICY'
}

Z_labels = {
    0: 'Z1  N/A',
    1: 'Z2  N/A',
    2: 'Z3  N/A',
    3: 'Z4  Books (General). Writing. Paleography—History of books and bookmaking—General works',
    4: 'Z5  Books (General). Writing. Paleography—History of books and bookmaking—By period—Early to 400',
    5: 'Z6  Books (General). Writing. Paleography—History of books and bookmaking—By period—400-1450',
    6: 'Z7  N/A',
    7: 'Z8  Books (General). Writing. Paleography—History of books and bookmaking—By region or country, A-Z',
    8: 'Z9  N/A',
    9: 'ZA  INFORMATION RESOURCES (GENERAL)'
}


k_3l_labels = {
    0: 'KBM  Islamic Theology - General' ,
    1: 'KBP  Islamic Theology' ,
    2: 'KJA  Islamic Jurisprudence - General' ,
    3: 'KJC  Islamic Jurisprudence - Maliki Legal Theory and Principles',
    4: 'KJE  Islamic Jurisprudence - Maliki Jurisprudence in Practice' ,
    5: 'KJK  Islamic Jurisprudence - Shafii',
    6: 'KJV  Islamic Jurisprudence' ,
    7: 'KKH  Islamic Law - Criminal',
    8: 'KKM  Islamic Law - Family',
    9: 'KKT  Islamic Law - Contracts',
    10: 'KKX  Islamic Law - Property',
    11: 'KMC  Islamic Culture and Civilization',
    12: 'KME  Islamic Jurisprudence - Shia' ,
    13: 'KMF  Islamic Ritual and Practice',
    14: 'KMH  Islamic Jurisprudence - Hanafi' ,
    15: 'KMJ  Islamic Jurisprudence - Hanbali' ,
    16: 'KMM  Islamic Moral and Ethical Thought' ,
    17: 'KMN  Islamic Jurisprudence' ,
    18: 'KMP  Islamic Ritual' ,
    19: 'KMQ  Islamic Ritual and Worship',
    20: 'KMS  Islamic Social Institutions' ,
    21: 'KMT  Islamic Theology and Doctrine' ,
    22: 'KMU  Islamic Mysticism',
    23: 'KMV  Islamic Sects and Movements',
    24: 'KMX  Islamic Intellectual and Cultural Life',
    25: 'KNX  Islamic Law - Procedure',
    26: 'KPL  Islamic Law - Personal Status' ,
    27: 'KQC  Islamic History - General',
    28: 'KQG  Islamic History' ,
    29: 'KRM  Islamic Law' ,
    30: 'KSP  Islamic Philosophy' ,
    31: 'KST  Islamic Intellectual History' ,
    32: 'KSW  Islamic Intellectual History' ,
    33: 'KTQ  Islamic Jurisprudence - Hanafi',
    34: 'KTV  Islamic Jurisprudence - Shafii',
    35: 'KZA  Islamic Law - General',
    36: 'KZD  Islamic Law - International'
}

# Download NLTK resources including the Arabic stopwords
nltk.download('stopwords')
nltk.download('punkt')
arabic_stopwords = set(stopwords.words('arabic'))

def remove_arabic_stopwords(text):
    arabic_stopwords = set(stopwords.words('arabic'))
    words = text.split()
    filtered_words = [word for word in words if word not in arabic_stopwords]
    return ' '.join(filtered_words)


def remove_tashkeel(text):
    tashkeel = "ًٌٍَُِّْ"
    for char in tashkeel:
        text = text.replace(char, '')
    return text



def check_subCategory(to_predict, model):
    # Preprocess the input text (remove diacritics)
    to_predict = remove_tashkeel(to_predict)
    
    # Transform input text using CountVectorizer and TfidfTransformer
    p_count = model[0].transform([to_predict])
    p_tfidf = model[1].transform(p_count)
    
    # Predict the subcategory
    subcategory_number = model[2].predict(p_tfidf)[0]
    
    # Get predicted probabilities for each subcategory
    probabilities = model[2].predict_proba(p_tfidf)[0] * 100
 
    return subcategory_number, probabilities


#.   start
def getlist(lstx):
    FinalResult.append("Top Predictions:")
    for i in lstx:
        #print(i)
        FinalResult.append(i)


   
# load model
# Load the consolidated model
def loadMosels():
    all_models = joblib.load('SharjahLibraryOfCongressModel.pkl')
    print("Models loaded")

def get_pred_label(to_predict):
    subcategory_numberTP, probabilities = check_subCategory(text, all_models["TP"])
    subcategory_nameTP = TOP_labels[subcategory_numberTP]
    resultTP = f"{subcategory_nameTP}  N#: {subcategory_numberTP}"
    topCode = resultTP.split("  ")[0]
    #print("Result:",topCode)
    FinalResult.append("Top Result:" + "\t" + topCode)
    if topCode == "A":
        subcategory_number, probabilities = check_subCategory(text, all_models["A"])
        subcategory_name = A_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), A_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)


    if topCode == "B":
        subcategory_number, probabilities = check_subCategory(text, all_models["B"])
        subcategory_name = B_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), B_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)


    if topCode == "C":
        subcategory_number, probabilities = check_subCategory(text, all_models["C"])
        subcategory_name = C_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), C_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)


    if topCode == "D":
        subcategory_number, probabilities = check_subCategory(text, all_models["D"])
        subcategory_name = D_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), D_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)


    if topCode == "E":
        subcategory_number, probabilities = check_subCategory(text, all_models["E"])
        subcategory_name = E_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), E_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)


    if topCode == "F":
        subcategory_number, probabilities = check_subCategory(text, all_models["F"])
        subcategory_name = F_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), F_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)


    if topCode == "G":
        subcategory_number, probabilities = check_subCategory(text, all_models["G"])
        subcategory_name = G_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), G_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)


    if topCode == "H":
        subcategory_number, probabilities = check_subCategory(text, all_models["H"])
        subcategory_name = H_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), H_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)


    if topCode == "J":
        subcategory_number, probabilities = check_subCategory(text, all_models["J"])
        subcategory_name = J_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), J_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)


    if topCode == "K":
        subcategory_number, probabilities = check_subCategory(text, all_models["K"])
        subcategory_name = K_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), K_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)


        subcategory_number, probabilities = check_subCategory(text, all_models["k_l3"])
        subcategory_name = k_3l_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), k_3l_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)


    if topCode == "L":
        subcategory_number, probabilities = check_subCategory(text, all_models["L"])
        subcategory_name = L_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), L_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)

    if topCode == "M":
        subcategory_number, probabilities = check_subCategory(text, all_models["M"])
        subcategory_name = M_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), M_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)

    if topCode == "N":
        subcategory_number, probabilities = check_subCategory(text, all_models["N"])
        subcategory_name = N_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), N_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)

    if topCode == "P":
        subcategory_number, probabilities = check_subCategory(text, all_models["P"])
        subcategory_name = P_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), P_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)

    if topCode == "Q":
        subcategory_number, probabilities = check_subCategory(text, all_models["Q"])
        subcategory_name = Q_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), Q_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)

    if topCode == "R":
        subcategory_number, probabilities = check_subCategory(text, all_models["R"])
        subcategory_name = R_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), R_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)

    if topCode == "S":
        subcategory_number, probabilities = check_subCategory(text, all_models["S"])
        subcategory_name = S_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), S_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)

    if topCode == "T":
        subcategory_number, probabilities = check_subCategory(text, all_models["T"])
        subcategory_name = T_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), T_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)

    if topCode == "U":
        subcategory_number, probabilities = check_subCategory(text, all_models["U"])
        subcategory_name = U_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), U_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)

    if topCode == "V":
        subcategory_number, probabilities = check_subCategory(text, all_models["V"])
        subcategory_name = V_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), V_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)


    if topCode == "W":
        subcategory_number, probabilities = check_subCategory(text, all_models["W"])
        subcategory_name = W_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), W_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)

    if topCode == "Z":
        subcategory_number, probabilities = check_subCategory(text, all_models["Z"])
        subcategory_name = Z_labels[subcategory_number]
        result = f"{subcategory_name}  N#: {subcategory_number}"
        sorted_indices = np.argsort(probabilities)[::-1]
        top_predictionsFinal = ['% {}  {}'.format(round(probabilities[i], 4), Z_labels[i]) for i in sorted_indices[:4]]
        FinalResult.append("Result:" + "\t" + result)
        #print("Result:", result)
        #print("Top Predictions:")
        getlist(top_predictionsFinal)


    return FinalResult
