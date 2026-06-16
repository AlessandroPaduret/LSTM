from build_vocabolary import crea_vocabolario_funzioni
from label_data import etichetta_syscall_ransomware
from extractor import estrai_features_per_processo

if __name__ == "__main__":
    CARTELLA_INPUT = "./raw"
    FILE_VOCABOLARIO = "./vocab/vocabolario_global_syscalls.json"
    CARTELLA_LABELED_DATA = "./data"
    CARTELLA_FEATURES = "./features"

    etichetta_syscall_ransomware(CARTELLA_INPUT, CARTELLA_LABELED_DATA)

    crea_vocabolario_funzioni(CARTELLA_INPUT, FILE_VOCABOLARIO)

    estrai_features_per_processo(
        CARTELLA_LABELED_DATA, CARTELLA_FEATURES, FILE_VOCABOLARIO
    )

    # ora puoi allenare il modello
