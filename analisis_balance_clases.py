import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV de registros
csv_path = 'outputRecords.csv'
df = pd.read_csv(csv_path)

# Mostrar el conteo de ejemplos por clase
print('Conteo de ejemplos por actividad:')
print(df['activity'].value_counts())

# Graficar el balance de clases
plt.figure(figsize=(8,5))
df['activity'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Balance de clases en outputRecords.csv')
plt.xlabel('Actividad')
plt.ylabel('Cantidad de ejemplos')
plt.tight_layout()
plt.show()

# Balancear el dataset por downsampling
min_count = df['activity'].value_counts().min()
print(f"\nCada clase se reducirá a {min_count} ejemplos para balancear el dataset.")

balanced_df = (
    df.groupby('activity', group_keys=False)
    .apply(lambda x: x.sample(min_count, random_state=42))
    .reset_index(drop=True)
)

# Guardar el nuevo dataset balanceado
balanced_df.to_csv('outputRecords_balanceado.csv', index=False)
print('\nNuevo archivo balanceado guardado como outputRecords_balanceado.csv')

# Graficar el balance de clases balanceado
plt.figure(figsize=(8,5))
balanced_df['activity'].value_counts().plot(kind='bar', color='limegreen')
plt.title('Balance de clases (balanceado)')
plt.xlabel('Actividad')
plt.ylabel('Cantidad de ejemplos')
plt.tight_layout()
plt.show()
