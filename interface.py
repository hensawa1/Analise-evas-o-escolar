import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler

# Carregar modelo e scaler
modelo = joblib.load("modelo_treinado.pkl")
scaler = joblib.load("scaler.pkl")

# Carregar dados para estatísticas adicionais
df = pd.read_csv("MICRODADOS_ENEM_ESCOLA.csv", sep=";", encoding="latin-1", low_memory=False)

# Mapear os códigos para nomes de tipo de escola
def classificar_escola(row):
    if row['TP_LOCALIZACAO_ESCOLA'] == 2:
        return 'Rural'
    elif row['TP_DEPENDENCIA_ADM_ESCOLA'] == 1:
        return 'Federal'
    elif row['TP_DEPENDENCIA_ADM_ESCOLA'] == 2:
        return 'Estadual'
    elif row['TP_DEPENDENCIA_ADM_ESCOLA'] == 4:
        return 'Privada'
    else:
        return 'Outros'

df['TIPO_ESCOLA'] = df.apply(classificar_escola, axis=1)
df = df[df['TIPO_ESCOLA'].isin(['Federal', 'Estadual', 'Privada', 'Rural'])]

# Remover nulos
colunas_taxas = [
    'NU_TAXA_APROVACAO', 'NU_TAXA_REPROVACAO',
    'NU_TAXA_ABANDONO', 'NU_TAXA_PERMANENCIA'
]
df = df.dropna(subset=[
    'NU_MEDIA_CN', 'NU_MEDIA_CH', 'NU_MEDIA_LP',
    'NU_MEDIA_MT', 'NU_MEDIA_RED', *colunas_taxas
])

# Corrigir taxas fora de faixa (0-100%)
for col in colunas_taxas:
    df = df[(df[col] >= 0) & (df[col] <= 100)]

# Interface interativa
print("=== Previsão de Desempenho Escolar ===")
tipos = ['Federal', 'Estadual', 'Privada', 'Rural']
print("Tipos disponíveis:", ", ".join(tipos))
tipo_escola = input("Digite o tipo de escola: ").capitalize()

if tipo_escola not in tipos:
    print("❌ Tipo de escola inválido.")
    exit()

# Entradas do usuário
try:
    cn = float(input("Média de Ciências da Natureza: "))
    ch = float(input("Média de Ciências Humanas: "))
    lp = float(input("Média de Linguagens e Códigos: "))
    mt = float(input("Média de Matemática: "))
    red = float(input("Nota da Redação: "))
except ValueError:
    print("❌ Entrada inválida. Use números.")
    exit()

# Montar o vetor de entrada
entrada = pd.DataFrame([[cn, ch, lp, mt, red]], columns=[
    'NU_MEDIA_CN', 'NU_MEDIA_CH', 'NU_MEDIA_LP', 'NU_MEDIA_MT', 'NU_MEDIA_RED'])

# Escalar a entrada
entrada_scaled = scaler.transform(entrada)

# Prever
previsao = modelo.predict(entrada_scaled)[0]
print(f"\n✅ Previsão da Média Total: {previsao:.2f}")


# Estatísticas do tipo de escola escolhido
grupo = df[df['TIPO_ESCOLA'] == tipo_escola]
print(f"\n📊 Estatísticas médias para escolas do tipo {tipo_escola}:")
print(f"Taxa de Aprovação: {grupo['NU_TAXA_APROVACAO'].mean():.2f}%")
print(f"Taxa de Reprovação: {grupo['NU_TAXA_REPROVACAO'].mean():.2f}%")
print(f"Taxa de Abandono: {grupo['NU_TAXA_ABANDONO'].mean():.2f}%")
print(f"Taxa de Permanência: {grupo['NU_TAXA_PERMANENCIA'].mean():.2f}%")

# Gráfico comparativo de todas as taxas por tipo de escola
taxas_corrigidas = df.groupby('TIPO_ESCOLA')[colunas_taxas].mean().round(2)
taxas_corrigidas.plot(kind='bar', figsize=(10, 6), colormap='Set2', edgecolor='black')
plt.title("Comparação de Taxas Educacionais por Tipo de Escola")
plt.ylabel("Percentual (%)")
plt.xlabel("Tipo de Escola")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
