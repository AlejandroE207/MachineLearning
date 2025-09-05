import pandas as pd
import random
import string

# Número de filas
n_real = 700
n_aleatorio = 300

# ------------------ Funciones Realistas ------------------
def generar_dominio_real(etiqueta):
    dominios_ham = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "protonmail.com"]
    dominios_spam = ["exit0.com", "freemail.xyz", "ofertas123.com", "trabajo-rápido.org",
                     "noticiaslive.co", "fakebank.co", "securepay.net"]
    dominios_neutrales = ["zoho.com", "tutanota.com", "icloud.com", "aol.com", "mail.com",
                          "empresa.org", "exito.com", "secure-mail.net"]
    if etiqueta == 0:
        poblacion = dominios_ham + dominios_neutrales*2 + dominios_spam
        pesos = [5]*len(dominios_ham) + [2]*len(dominios_neutrales)*2 + [1]*len(dominios_spam)
    else:
        poblacion = dominios_spam + dominios_neutrales*2 + dominios_ham
        pesos = [5]*len(dominios_spam) + [2]*len(dominios_neutrales)*2 + [1]*len(dominios_ham)
    return random.choices(poblacion, weights=pesos)[0]

def generar_url_real(etiqueta):
    base = ["http://", "https://"]
    sitios_ham = ["oferta.com", "promo.net", "secure.org", "bank.com"]
    sitios_spam = ["fakeprize.xyz", "clickhere.biz", "freegift.io", "malicious.co", "superdeal.shop",
                   "lottery.win", "investcrypto.pro", "phishing-page.net", "couponfast.info",
                   "secure-login.co", "freemoney.biz", "discountzone.org", "getprize.store"]
    if etiqueta == 0:
        poblacion = sitios_ham + sitios_spam
        pesos = [5]*len(sitios_ham) + [1]*len(sitios_spam)
    else:
        poblacion = sitios_spam + sitios_ham
        pesos = [5]*len(sitios_spam) + [1]*len(sitios_ham)
    sitio = random.choices(poblacion, weights=pesos)[0]
    return random.choice(base) + "".join(random.choices(string.ascii_lowercase, k=5)) + "." + sitio

def generar_bool_real(etiqueta, tipo):
    prob = random.random()
    if etiqueta == 0:
        if tipo == 'tls': return prob < 0.85
        elif tipo in ['adjunto','oferta','imagen','remitente']: return prob < 0.1
        elif tipo == 'contenido': return prob < 0.2
    else:
        if tipo == 'tls': return prob < 0.3
        elif tipo in ['adjunto','oferta','imagen','remitente']: return prob < 0.6
        elif tipo == 'contenido': return prob < 0.5

def generar_texto_real(etiqueta):
    return round(random.uniform(0.4,0.95),2) if etiqueta==0 else round(random.uniform(0.1,0.7),2)

def generar_ip():
    return ".".join(str(random.randint(0,255)) for _ in range(4))

def generar_fecha():
    return pd.Timestamp("2023-01-01") + pd.to_timedelta(random.randint(0,365), unit="d")

# ------------------ Crear etiquetas ------------------
etiquetas_real = [0]*(n_real//2) + [1]*(n_real - n_real//2)
random.shuffle(etiquetas_real)
etiquetas_aleatorio = [random.choice([0,1]) for _ in range(n_aleatorio)]

# ------------------ Crear datasets ------------------
# Dataset realista
data_real = {
    "Fecha": [generar_fecha() for _ in range(n_real)],
    "Dominio": [generar_dominio_real(e) for e in etiquetas_real],
    "PorcentajeTexto": [generar_texto_real(e) for e in etiquetas_real],
    "IP": [generar_ip() for _ in range(n_real)],
    "CorreoConTLS": [generar_bool_real(e,"tls") for e in etiquetas_real],
    "ArchivosAdjuntosPeligrosos": [generar_bool_real(e,"adjunto") for e in etiquetas_real],
    "OfertasIrreales": [generar_bool_real(e,"oferta") for e in etiquetas_real],
    "UrlsIndexados": [generar_url_real(e) for e in etiquetas_real],
    "ImagenesCodigoOculto": [generar_bool_real(e,"imagen") for e in etiquetas_real],
    "HeaderRemitenteFalso": [generar_bool_real(e,"remitente") for e in etiquetas_real],
    "ContenidoSensible": [generar_bool_real(e,"contenido") for e in etiquetas_real],
    "Etiqueta": etiquetas_real
}

# Dataset aleatorio
data_aleatorio = {
    "Fecha": [generar_fecha() for _ in range(n_aleatorio)],
    "Dominio": [random.choice(["gmail.com","yahoo.com","hotmail.com","exit0.com","freemail.xyz","empresa.org"]) for _ in range(n_aleatorio)],
    "PorcentajeTexto": [round(random.uniform(0.1,0.95),2) for _ in range(n_aleatorio)],
    "IP": [generar_ip() for _ in range(n_aleatorio)],
    "CorreoConTLS": [random.choice([True,False]) for _ in range(n_aleatorio)],
    "ArchivosAdjuntosPeligrosos": [random.choice([True,False]) for _ in range(n_aleatorio)],
    "OfertasIrreales": [random.choice([True,False]) for _ in range(n_aleatorio)],
    "UrlsIndexados": ["http://"+''.join(random.choices(string.ascii_lowercase,k=5))+".com" for _ in range(n_aleatorio)],
    "ImagenesCodigoOculto": [random.choice([True,False]) for _ in range(n_aleatorio)],
    "HeaderRemitenteFalso": [random.choice([True,False]) for _ in range(n_aleatorio)],
    "ContenidoSensible": [random.choice([True,False]) for _ in range(n_aleatorio)],
    "Etiqueta": etiquetas_aleatorio
}

# ------------------ Combinar ------------------
df = pd.concat([pd.DataFrame(data_real), pd.DataFrame(data_aleatorio)], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)  # Mezclar todo

# Guardar
df.to_csv("dataset_spam_700_300.csv", index=False)
print("✅ Dataset combinado (700 reales, 300 aleatorios) generado")
print(df['Etiqueta'].value_counts())
print(df.head())
