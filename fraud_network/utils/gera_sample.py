import pandas as pd
import random

def gerar_dataframe_amostra(num_rows, sender_pool, receiver_pool, amount_range, date_start):
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

    # Gerando dados aleatórios
    senders = [random.choice(sender_pool) for _ in range(num_rows)]
    receivers = [random.choice(receiver_pool) for _ in range(num_rows)]
    amounts = [random.randint(*amount_range) for _ in range(num_rows)]
    timestamps = pd.date_range(start=date_start, periods=num_rows, freq='D')
    fraud_label = [random.randint(0, 1) for _ in range(num_rows)]

    # Criando o DataFrame
    df = pd.DataFrame({
        'sender': senders,
        'receiver': receivers,
        'amount': amounts,
        'timestamp': timestamps,
        'fraud_label': fraud_label
    })

    df = df.loc[df.sender != df.receiver]
    return df
