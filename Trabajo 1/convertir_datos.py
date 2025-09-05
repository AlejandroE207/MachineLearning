import pandas as pd

# 1️⃣ Cargar CSV
df = pd.read_csv(r"Trabajo 1\dataset_spam_700_300.csv")

# 2️⃣ Fecha → 5 rangos
def convertir_fecha_5r(fecha):
    hora = pd.to_datetime(fecha, errors="coerce").hour
    if pd.isna(hora):
        return 3
    if 0 <= hora <= 3:
        return 1
    elif 4 <= hora <= 7:
        return 2
    elif 8 <= hora <= 11:
        return 3
    elif 12 <= hora <= 17:
        return 4
    else:
        return 5

df["Fecha_Num"] = df["Fecha"].apply(convertir_fecha_5r)
df = df.drop(columns=["Fecha"])

# 3️⃣ Dominio → 5 rangos
def clasificar_dominio_5r(dominio):
    muy_confiables = ["gmail.com", "yahoo.com"]
    confiables = ["hotmail.com", "outlook.com", "protonmail.com"]
    neutrales = ["zoho.com", "tutanota.com", "icloud.com", "aol.com", "mail.com", "empresa.org", "exito.com"]
    sospechosos = ["secure-mail.net", "fakebank.co", "securepay.net"]
    muy_sospechosos = ["freemail.xyz", "ofertas123.com", "trabajo-rápido.org", "noticiaslive.co", "exit0.com"]

    if dominio in muy_confiables:
        return 1
    elif dominio in confiables:
        return 2
    elif dominio in neutrales:
        return 3
    elif dominio in sospechosos:
        return 4
    else:
        return 5

df["Dominio_Num"] = df["Dominio"].apply(clasificar_dominio_5r)
df = df.drop(columns=["Dominio"])

# 4️⃣ IP → 5 rangos por percentiles
def ip_a_numero(ip):
    try:
        octetos = list(map(int, ip.split(".")))
        return octetos[0]*256**3 + octetos[1]*256**2 + octetos[2]*256 + octetos[3]
    except:
        return 0

df["IP_num"] = df["IP"].apply(ip_a_numero)
p20, p40, p60, p80 = df["IP_num"].quantile([0.2, 0.4, 0.6, 0.8])

def rango_ip_5r(num):
    if num <= p20:
        return 1
    elif num <= p40:
        return 2
    elif num <= p60:
        return 3
    elif num <= p80:
        return 4
    else:
        return 5

df["Rango_IP"] = df["IP_num"].apply(rango_ip_5r)
df = df.drop(columns=["IP_num", "IP"])

# 5️⃣ URL → 5 rangos
def clasificar_url_5r(url):
    url = str(url).lower()
    muy_sospechosos = ["phishing", "malicious", "freegift", "getprize", "lottery", ".xyz", ".biz", ".io", ".co", ".win"]
    sospechosos = ["secure-login", "freemoney", "discountzone", "investcrypto", "promo"]
    neutro = [".com", ".org", ".net", "bank", "offer"]

    if any(s in url for s in muy_sospechosos):
        return 5
    elif any(s in url for s in sospechosos):
        return 4
    elif any(s in url for s in neutro):
        return 3
    elif len(url) < 15:
        return 2
    else:
        return 1

df["URL_Num"] = df["UrlsIndexados"].apply(clasificar_url_5r)
df = df.drop(columns=["UrlsIndexados"])

# 6️⃣ Booleanos → 0/1
columnas_bool = ["CorreoConTLS", "ArchivosAdjuntosPeligrosos", "OfertasIrreales",
                 "ImagenesCodigoOculto", "HeaderRemitenteFalso", "ContenidoSensible"]

for col in columnas_bool:
    df[col] = df[col].astype(int)

# 7️⃣ Etiqueta → 0/1
df["Etiqueta_Num"] = df["Etiqueta"].map({1:1, 0:0})
df = df.drop(columns=["Etiqueta"])

# 8️⃣ Guardar dataset listo
df.to_csv(r"Trabajo 1\dataset_datos_convertidos.csv", index=False)
print("✅ Dataset convertido y listo para regresión logística")
print(df.head())
