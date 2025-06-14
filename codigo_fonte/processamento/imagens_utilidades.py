import pandas as pd
import os
from ultralytics import YOLO
from PIL import Image
import shutil
import imagehash
import cv2
from skimage.metrics import structural_similarity as ssim
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pytesseract
from pathlib import Path
import joblib
from itertools import combinations
from codigo_fonte.processamento.csv_utilidades import unifica_registros_por_imagem

# Caminho do modelo salvo
modelo_path_spacy = "dados/modelos/spacy_model_relacionado.pkl"


# Caminhos
caminho_base_posts = "dados/processados/base_posts.csv"
caminho_imagem_post =Path("dados/processados/imagem")
caminho_descarte_sem_cachorro = Path("dados/processados/imagem/descartadas/semCachorro")
caminho_descarte_sem_contexto = Path("dados/processados/imagem/descartadas/semContexto")

# Criando as pastas, caso não existam
caminho_imagem_post.mkdir(parents=True, exist_ok=True)
caminho_descarte_sem_cachorro.mkdir(parents=True, exist_ok=True)
caminho_descarte_sem_contexto.mkdir(parents=True, exist_ok=True)

# Carregar modelo pré-treinado YOLOv8
model = YOLO("yolov8n.pt")  # ou yolov8s.pt para maior precisão


def contem_cachorro(caminho_imagem, conf_minima=0.3):
    """
    Verifica se há um cachorro na imagem usando YOLOv8.
    
    Retorna True se houver, False caso contrário.
    """

    results = model.predict(caminho_imagem, conf=conf_minima, verbose=False)
    
    for result in results:
        for box in result.boxes:
            classe_id = int(box.cls[0])
            nome_classe = model.names[classe_id]
            if nome_classe.lower() == "dog":
                return True
    return False


def filtrar_imagens_com_cachorro():
    origem = "dados/brutos/imagens"

    df = pd.read_csv(caminho_base_posts)

    for nome_arquivo in os.listdir(origem):
        caminho_imagem = os.path.join(origem, nome_arquivo)

        if caminho_imagem.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                if contem_cachorro(caminho_imagem):
                    shutil.move(caminho_imagem, os.path.join(caminho_imagem_post, nome_arquivo))
                else:
                    shutil.move(caminho_imagem, os.path.join(caminho_descarte_sem_cachorro, nome_arquivo))
                    # Remove a linha correspondente do CSV
                    df = df[df['nome_imagem'] != nome_arquivo]

            except Exception as e:
                print(f"Erro ao processar {nome_arquivo}: {e}")

    # Salva o CSV atualizado
    df.to_csv(caminho_base_posts, index=False)



def imagens_sao_semelhantes(img1_path, img2_path, limiar=0.75):
    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        # Redimensiona as duas imagens para o menor tamanho entre elas
        altura = min(img1.shape[0], img2.shape[0])
        largura = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (largura, altura))
        img2 = cv2.resize(img2, (largura, altura))

        score, _ = ssim(img1, img2, full=True)
        return score >= limiar
    except Exception as e:
        # print(f"Erro ao comparar imagens: {e}")
        return False


def recortar_cachorro_da_imagem():

    filtrar_imagens_com_cachorro()
    seleciona_imagem_com_texto_cachorro()

    modelo = YOLO("yolov8n.pt")  # ou yolov8s.pt para melhor precisão
    pasta_origem = Path("dados/processados/imagem")
    pasta_destino = Path("dados/processados/imagem/recorte")
    pasta_destino.mkdir(parents=True, exist_ok=True)

    for imagem_path in pasta_origem.glob("*.*"):
        # Carrega a imagem
        imagem = cv2.imread(str(imagem_path))

        # Detecta objetos na imagem
        resultados = modelo(imagem, verbose=False)[0]

        # Verifica se há cachorro (classe 16 no COCO)
        for i, det in enumerate(resultados.boxes.data):
            classe = int(det[5])
            if classe == 16:  # Classe 16 = dog no COCO
                x1, y1, x2, y2 = map(int, det[:4])
                recorte = imagem[y1:y2, x1:x2]

                # Salva recorte com o mesmo nome
                caminho_salvar = pasta_destino / imagem_path.name
                cv2.imwrite(str(caminho_salvar), recorte)
                break  



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extrair_texto_da_imagem(caminho_imagem):
    # Lê e converte a imagem para tons de cinza
    img = cv2.imread(str(caminho_imagem))
    if img is None:
        return ""

    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binariza para melhorar OCR
    _, binarizada = cv2.threshold(cinza, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Converte para formato compatível com pytesseract
    imagem_pil = Image.fromarray(binarizada)

    # Extrai texto com OCR
    texto = pytesseract.image_to_string(imagem_pil, lang="por")

    return texto.lower().strip()




# Função para extrair texto e classificar
def extrair_texto_e_classificar(caminho_imagem):

    # Carrega o modelo
    modelo_spacy = joblib.load(modelo_path_spacy)
    
    try:
        imagem = Image.open(caminho_imagem)
        texto = pytesseract.image_to_string(imagem)

        # Caso não tenha texto na imagem, o registro é mantido
        if texto.strip() == "":
            return "relacionado",""

        return modelo_spacy.predict([texto])[0],texto  # "relacionado" ou "nao-relacionado" e o texto extraído

    except Exception as e:
        # print(f"Erro ao processar {caminho_imagem.name}: {e}")
        return "nao-relacionado",None

def seleciona_imagem_com_texto_cachorro():

    # Lê o CSV
    df = pd.read_csv(caminho_base_posts)

    # Percorre todas as imagens
    for imagem_path in caminho_imagem_post.glob("*.*"):
        classificacao,texto = extrair_texto_e_classificar(imagem_path)

        if classificacao == "relacionado":
            # Atualiza o texto extraído na coluna 'Content'
            df.loc[df['nome_imagem'] == imagem_path.name, 'Content'] += " " + texto
        else:
            # Move para descartadas
            destino = caminho_descarte_sem_contexto / imagem_path.name
            shutil.move(str(imagem_path), str(destino))

            # Remove do DataFrame
            df = df[df['nome_imagem'] != imagem_path.name]

    # Salva o CSV atualizado
    df.to_csv(caminho_base_posts, index=False)





def comparar_todas_imagens_recorte():
    
    recortar_cachorro_da_imagem()

    pasta_recorte="dados/processados/imagem/recorte"
    pasta_descartadas="dados/processados/imagem/descartadas/iguais"
    limiar_similaridade=0.85

    imagens = list(Path(pasta_recorte).glob("*.*"))
    imagens_comparadas = set()
    df = pd.read_csv(caminho_base_posts)

    Path(pasta_descartadas).mkdir(parents=True, exist_ok=True)

    for img1, img2 in combinations(imagens, 2):
        nome1, nome2 = img1.name, img2.name

        # Evita comparar imagens já descartadas
        if nome1 in imagens_comparadas or nome2 in imagens_comparadas:
            continue

        imagem1 = cv2.imread(str(img1), cv2.IMREAD_GRAYSCALE)
        imagem2 = cv2.imread(str(img2), cv2.IMREAD_GRAYSCALE)

        if imagem1 is None or imagem2 is None:
            continue

        # Redimensiona para mesmo tamanho
        altura = min(imagem1.shape[0], imagem2.shape[0])
        largura = min(imagem1.shape[1], imagem2.shape[1])
        imagem1 = cv2.resize(imagem1, (largura, altura))
        imagem2 = cv2.resize(imagem2, (largura, altura))

        # Similaridade estrutural
        score, _ = ssim(imagem1, imagem2, full=True)

        if score > limiar_similaridade:
            print(f"Imagens semelhantes: {nome1} e {nome2} (SSIM: {score:.2f})")

            unifica_registros_por_imagem(nome1, nome2)
        
            # Move imagem duplicada
            shutil.move(str(img2), os.path.join(pasta_descartadas, nome2))
            imagens_comparadas.add(nome2)

    # Salva CSV atualizado
    df.to_csv(caminho_base_posts, index=False)
    print("Comparação concluída e CSV atualizado.")