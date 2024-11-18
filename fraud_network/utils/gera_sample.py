import pandas as pd
import numpy as np
import random

def gerar_dataframe_amostra(num_rows):
    """Gera um DataFrame simulando transações.

    Args:
        num_rows: Número de linhas (transações) a serem geradas.
        sender_pool: Lista de possíveis remetentes.
        receiver_pool: Lista de possíveis destinatários.
        amount_range: Tupla indicando o intervalo mínimo e máximo do valor da transação.
        date_range: Tupla indicando a data inicial e final do intervalo de datas.

    Returns:
        Um DataFrame Pandas com as colunas 'sender', 'receiver', 'amount' e 'timestamp'.
    """
    # Exemplo de uso:
    sender_pool = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    receiver_pool = sender_pool.copy()
    amount_range = (1000, 50000)
    start_date, end_date = '2023-01-01', '2023-12-31'

    # Gerando dados aleatórios
    senders = [random.choice(sender_pool) for _ in range(num_rows)]
    receivers = [random.choice(receiver_pool) for _ in range(num_rows)]
    amounts = [random.randint(*amount_range) for _ in range(num_rows)]
    timestamps = pd.to_datetime(np.random.choice(pd.date_range(start=start_date, end=end_date), size=num_rows))
    fraud_label = np.zeros(num_rows)

    

    # Criando o DataFrame
    df = pd.DataFrame({
        'sender': senders,
        'receiver': receivers,
        'amount': amounts,
        'timestamp': timestamps,
        'fraud_label': fraud_label
    })

    for i in range(num_rows):
        while df.loc[i, 'sender'] == df.loc[i, 'receiver']:
            df.loc[i, 'receiver'] = random.choice(receivers)

    for _ in range(int(num_rows*0.3)):
        i = np.random.choice(num_rows)
        df.loc[i, 'amount'] = random.randint(100, 350)
        df.loc[i, 'fraud_label'] = 1
    
    return df

