import os
#import shutil
#import cv2
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
import spacy
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from torch import combinations

# Carregar modelo de linguagem em português do spaCy
nlp = spacy.load("pt_core_news_md")

#Caminho do arquivo csv que armazena as informações extraidas dos posts
caminho_base_posts = "dados/processados/base_posts.csv"

# Colunas que serão usadas
colunas = ['Content','Url','Posted At']


# Classe para transformar texto em embeddings com spaCy
class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nlp = spacy.load("pt_core_news_md")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.nlp(text).vector for text in X]

# Função definida para unir vários arquivos em um só
def une_csv():
    caminho_csv = os.path.join(os.path.dirname(__file__), '..', '..', 'Dados', 'Brutos', 'csv')
    caminho_csv = os.path.abspath(caminho_csv)

    arquivos = [os.path.join(caminho_csv, f) for f in os.listdir(caminho_csv) if f.endswith('.csv')]

    df_unido = pd.concat([pd.read_csv(arquivo) for arquivo in arquivos], ignore_index=True)

    df_unido = df_unido[colunas]

    df_unido.to_csv(caminho_base_posts, index=False)
    
    return



def cria_base_rotulada_texto():
    # Caminhos dos arquivos
    caminho_relacionado = "dados/brutos/txt/relacionado.txt"
    caminho_nao_relacionado = "dados/brutos/txt/nao-relacionado.txt"
    caminho_saida = "dados/processados/base_rotulada.csv"

    # Lê os arquivos como listas de frases
    with open(caminho_relacionado, encoding='utf-8') as f:
        relacionados = [linha.strip() for linha in f if linha.strip()]

    with open(caminho_nao_relacionado, encoding='utf-8') as f:
        nao_relacionados = [linha.strip() for linha in f if linha.strip()]

    # Monta o DataFrame
    textos = relacionados + nao_relacionados
    rotulos = ['relacionado'] * len(relacionados) + ['nao-relacionado'] * len(nao_relacionados)

    df = pd.DataFrame({
        'texto': textos,
        'rotulo': rotulos
    })

    # Salva o CSV
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    df.to_csv(caminho_saida, index=False)
    print(f"Base salva em: {caminho_saida}")


# def cria_base_rotulada_texto():
#     etiquetas = ['relacionado','nao-relacionado']

#     termos_relacionados = ['cachorro encontrado','cachorro desaparecido','cachorro perdido', 'cachorro sumiu', 'cachorra encontrada'
#                            'viu','cade o dono?','procurando?','fugiu', 'escapou','procura-se','encontrar o dono','ele sumiu','ela sumiu'
#                            'sumiu','with a Facebook','with a Photo','procuro tutor','procuro dono','procurando','desapareceu'
#                            'escapou', 'saiu', 'está perdido','está perdida','alguém conhece?' ]
#     termos_nao_relacionados = ['gato','gatinho' ,'pássaro','tartaruga','calopsita','ração','hospedagem','vendo','alugo','compro',
#                                'doação','adotar','lar temporário','resgate']

#     textos = []
#     rotulos = []

#     for texto in termos_relacionados:
#         textos.append(texto)
#         rotulos.append(etiquetas[0])
#     for texto in termos_nao_relacionados:
#         textos.append(texto)
#         rotulos.append(etiquetas[1])

#     df = pd.DataFrame({
#     'texto': textos,
#     'rotulo': rotulos
#         })
    
#     caminho_salvar = os.path.join("dados/processados", "base_rotulada.csv")
#     df.to_csv(caminho_salvar, index=False)



def cria_modelo():

    # Exemplo com base rotulada
    df_train = pd.read_csv('dados/processados/base_rotulada.csv')

    # Treinamento com divisão de treino/teste
    X_train, X_test, y_train, y_test = train_test_split(df_train['texto'], df_train['rotulo'], test_size=0.2, random_state=42)

    # Pipeline
    pipeline = Pipeline([
        ("vect", SpacyVectorTransformer()),
        ("clf", LogisticRegression())
    ])

    pipeline.fit(X_train, y_train)

    # Avaliação
    accuracy = pipeline.score(X_test, y_test)

    # Salvar o modelo
    joblib.dump(pipeline, "dados/modelos/spacy_model_relacionado.pkl")


class SentenceTransformerVectorizer:
    def __init__(self, model_name='neuralmind/bert-base-portuguese-cased'):
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.encode(X.tolist(), convert_to_numpy=True)




def cria_modelo_transformer():

    df = pd.read_csv('dados/processados/base_rotulada.csv')

    X_train, X_test, y_train, y_test = train_test_split(df['texto'], df['rotulo'], test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("bert", SentenceTransformerVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Relatório de classificação:\n", classification_report(y_test, y_pred))

    joblib.dump(pipeline, "dados/modelos/spacy_model_relacionado.pkl")



# Função definida para criar um arquivo csv apenas com postagens relacionados a cachorros desaparecidos/encontrados
#e postagens que não possuem texto
def filtra_content():

    # 1. Carrega o CSV com as postagens
    df = pd.read_csv(caminho_base_posts) 

    # Carregar modelo treinado
    modelo = joblib.load("dados/modelos/spacy_model_relacionado.pkl")

    df['Content'] = df['Content'].fillna('').astype(str)

    # Previsão
    df['relacionado'] = modelo.predict(df['Content'])

    # Filtrando
    df_filtrado = df[df['relacionado'] == 'relacionado']
   
    df_filtrado = df_filtrado[colunas]

    # 5. Salva em um novo arquivo CSV
    df_filtrado.to_csv(caminho_base_posts, index=False)


def postsComURL():

    #Unindo os arquivos csv que foram baixados com a extensão ESUIT
    une_csv()
    # #Cria uma base rotulada com o contexto de cachorros desaparecido ou encontrado
    cria_base_rotulada_texto()
    # # Cria um modelo pré treina com a base rotulada. Modelo este que será utilizado para filtrar apenas posts que estejam no contexto de cachorro encontrado/desaparecido
    #e os posts que não possuem texto, só uma URL tbm serão mantidos
    cria_modelo_transformer()

    filtra_content()

    # Carregando o CSV
    caminho_csv = os.path.join(os.path.dirname(__file__), '..', '..', 'dados', 'processados')
    caminho_csv = os.path.abspath(caminho_csv)
    # 1. Carrega o CSV com as postagens
    df = pd.read_csv(caminho_base_posts) 

    # Expressão regular para detectar URL
    padrao_url = r'https?://[^\s]+'

    # Filtra linhas que possuem URL na coluna 'Content'
    df_com_url = df[df['Content'].astype(str).str.contains(padrao_url, na=False)]


    df_com_url.to_csv(caminho_base_posts, index=False)




def manipula_base_processada(url, content=None, posted_at=None, nome_imagem=None):

    csv_caminho = Path(caminho_base_posts)

    # Se já existir, lê o arquivo
    if csv_caminho.exists():
        df = pd.read_csv(csv_caminho)
    # Se não existir, inicia um DataFrame
    else:
        df = pd.DataFrame(columns=[colunas,"nome_imagem"])

    # Verifica se a URL já está registrada
    if url in df["url"].values:
        # Atualiza os campos fornecidos
        index = df[df["url"] == url].index[0]
        if content is not None:
            df.at[index, "Content"] = content
        if url is not None:
            df.at[index, "Url"] = url
        if posted_at is not None:
            df.at[index, "Posted At"] = posted_at
        if nome_imagem is not None:
            df.at[index, "nome_imagem"] = nome_imagem
        
    else:
        # Cria novo registro
        novo_registro = {
            "url":url,
            "Content": content if content else "",
            "Posted At":posted_at if posted_at else "",            
            "nome_imagem": nome_imagem if nome_imagem else "",
        }
        df = pd.concat([df, pd.DataFrame([novo_registro])], ignore_index=True)
        

    # Salva de volta
    df.to_csv(caminho_base_posts, index=False)





def unifica_registros_por_imagem(imagem_mantida, imagem_removida):
    
    # Verifica se o CSV existe
    if not Path(caminho_base_posts).exists():
        print(f"Arquivo CSV não encontrado: {caminho_base_posts}")
        return

    # Carrega o CSV
    df = pd.read_csv(caminho_base_posts)

    # Verifica se ambas as imagens existem no CSV
    if imagem_mantida not in df["nome_imagem"].values or imagem_removida not in df["nome_imagem"].values:
        print("Uma ou ambas as imagens não estão no CSV.")
        print("Nome 1: ",imagem_mantida)
        print("Nome 2: ",imagem_removida)
        return

    # Recupera os registros
    registro_mantido = df[df["nome_imagem"] == imagem_mantida].iloc[0]
    registro_removido = df[df["nome_imagem"] == imagem_removida].iloc[0]

    # Unifica as características e o rótulo (evita duplicar informações)
    caracteristicas_mantidas = str(registro_mantido.get("Content", ""))
    caracteristicas_removidas = str(registro_removido.get("Content", ""))
    novas_caracteristicas = "; ".join(filter(None, {caracteristicas_mantidas, caracteristicas_removidas}))

    
    # Atualiza o registro mantido
    df.loc[df["nome_imagem"] == imagem_mantida, "Content"] = novas_caracteristicas
    
    # Remove o registro duplicado
    df = df[df["nome_imagem"] != imagem_removida]

    # Salva o CSV atualizado
    df.to_csv(caminho_base_posts, index=False)
   



